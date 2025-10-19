"""Flamingo version of PoET model, with flash attention and
ESM-IF1 encoder for structure"""

import copy
import re
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from poet.alphabets import Uniprot21
from poet.models.modules.embedding import RotaryEmbedding
from poet.models.modules.packed_sequence import (
    PackedTensorSequences,
    get_mask,
    pad_input,
    unpad_input,
)
from poet.models.modules.structure_adapter import (
    AdaptedTransformerEncoder,
    EncoderProjector,
    GeometricAttentionProjector,
    TieredGatedTransformerEncoderLayer,
)
from poet.models.poet import (
    LogitsAllocateMemoryMixin,
    _compute_attn_memory,
    _packed_sequence_append,
    _update_causal_prefix_memory,
    top_k_top_p_filtering,
)


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)


class PoETS(nn.Module, LogitsAllocateMemoryMixin):
    def __init__(
        self,
        n_vocab: int,
        structure_encoder: Optional[nn.Module] = None,
        encoder_dim: int = 512,
        hidden_dim: int = 768,
        ff_dim: Optional[int] = None,
        num_layers: int = 6,
        nhead: int = 12,
        dropout: float = 0,
        use_multi_rotary: bool = True,
        norm: bool = False,
        alpha_init: float = 1e-5,
        projector: str = "mlp",
        add_gaussian_noise: bool = True,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.token_embed = nn.Embedding(n_vocab, hidden_dim)
        # kept just to maintain compatability with old models
        self.rotary_emb = RotaryEmbedding(hidden_dim // nhead)

        ff_dim = ff_dim or 4 * hidden_dim
        self.struct_encoder = structure_encoder

        if projector == "mlp":
            projector_type = EncoderProjector
        elif projector == "geom_att":
            projector_type = GeometricAttentionProjector
        else:
            raise ValueError(f"Invalid projector type: {projector}")

        self.encoder_proj = projector_type(
            d_model=hidden_dim,
            d_encoder=encoder_dim,
            dropout=dropout,
            add_gaussian_noise=add_gaussian_noise,
        )

        self.decoder = AdaptedTransformerEncoder(
            encoder_layer=TieredGatedTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                use_multi_rotary=use_multi_rotary,
                batch_first=True,
                alpha_init=alpha_init,
                causal=True,
            ),
            num_layers=num_layers,
        )

        if norm:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = nn.Identity()

        self.linear = nn.Linear(hidden_dim, n_vocab)

        if self.training:
            self._freeze_pretrained_layers()  # freeze the pretrained layers for tranining

    def _freeze_pretrained_layers(self):
        freeze_model_and_make_eval_(self.token_embed)
        if self.struct_encoder is not None:
            freeze_model_and_make_eval_(self.struct_encoder)
        for layer in self.decoder.layers:
            freeze_model_and_make_eval_(layer.tiered_block)
        freeze_model_and_make_eval_(self.norm)
        freeze_model_and_make_eval_(self.linear)

    @classmethod
    def from_poet_pretrained(cls, poet_ckpt_path: str, **kwargs):
        # load model
        ckpt = torch.load(poet_ckpt_path, weights_only=False)
        # encoder = EsmIF1().get_encoder()  # load the ESM-IF1 encoder, with already trained weights
        # No need to load struct encoder for training since we use precomputed embeddings
        init_args = ckpt["hyper_parameters"]["model_spec"]["init_args"]
        init_args.update(**kwargs)
        print("Loading model with the following arguments:", init_args)
        model = cls(**init_args)
        # set dropout
        updated_state_dict = {
            re.sub(
                r"decoder\.layers\.(\d+)",
                r"decoder.layers.\1.tiered_block",
                k.split(".", 1)[1],
            ): v
            for k, v in ckpt["state_dict"].items()
        }
        missing_keys, unexpected_keys = model.load_state_dict(
            updated_state_dict, strict=False
        )
        assert not unexpected_keys, f"Unexpected keys: {unexpected_keys}"
        # remove known missing keys from the missing keys list
        missing_keys = [
            k for k in missing_keys if not "encoder" in k and "gated_block" not in k
        ]
        assert not missing_keys, f"Missing keys: {missing_keys}"
        return model

    def load_pretrained(self, poet_ckpt_path: str):
        ckpt = torch.load(poet_ckpt_path, weights_only=False)
        updated_state_dict = {
            re.sub(
                r"decoder\.layers\.(\d+)",
                r"decoder.layers.\1.tiered_block",
                k.split(".", 1)[1],
            ): v
            for k, v in ckpt["state_dict"].items()
        }
        missing_keys, unexpected_keys = self.load_state_dict(
            updated_state_dict, strict=False
        )
        assert not unexpected_keys, f"Unexpected keys: {unexpected_keys}"
        # remove known missing keys from the missing keys list
        missing_keys = [
            k for k in missing_keys if not "encoder" in k and "gated_block" not in k
        ]
        assert not missing_keys, f"Missing keys: {missing_keys}"

    def _prepare_inputs_for_decoder(
        self,
        input_ids: torch.Tensor,
        segment_sizes: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.LongTensor] = None,
        confidence: Optional[torch.Tensor] = None,
        structure_embeds: Optional[torch.Tensor] = None,
    ):
        seqs_seqlens = segment_sizes.sum(dim=1).type(torch.int32)
        xs, indices, _, _ = unpad_input(input_ids.unsqueeze(2), ~get_mask(seqs_seqlens))
        xs = xs.squeeze(1)
        h = self.token_embed(xs)

        segment_sizes_cpu = segment_sizes.cpu()
        seqs_seqlens_cpu = segment_sizes_cpu.sum(dim=1).type(torch.int32)
        nonzero_segment_sizes_cpu = (
            segment_sizes_cpu[segment_sizes_cpu > 0].flatten().type(torch.int32)
        )
        cu_seqlens_cpu = F.pad(
            nonzero_segment_sizes_cpu.cumsum(
                dim=0, dtype=nonzero_segment_sizes_cpu.dtype
            ),
            (1, 0),
        )

        cu_seqlens = cu_seqlens_cpu.to(xs.device)
        h = PackedTensorSequences(
            packed_tensor=h,
            positions=torch.cat(
                [
                    torch.arange(segment_size, dtype=xs.dtype, device=xs.device)
                    for segment_size in nonzero_segment_sizes_cpu
                ]
            ),
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_s=nonzero_segment_sizes_cpu.max(),
            # only needed for unpadding (used in standard attn)
            to_paddedable=False,
            indices=None,
            batch_size=None,
        )

        if coords is not None and structure_embeds is None:
            with torch.no_grad():
                structure_embeds = self.struct_encoder(
                    coords, padding_mask, confidence
                )[
                    "encoder_out"
                ]  # may need to be changed

        if structure_embeds is not None:
            structure_embeds = self.encoder_proj(structure_embeds, coords=coords)
            s = self._pack_structure_embeds(structure_embeds, segment_sizes)

        return h, s, seqs_seqlens, seqs_seqlens_cpu, indices

    def _pack_structure_embeds(
        self, structure_embeds: torch.Tensor, segment_sizes: torch.Tensor
    ) -> PackedTensorSequences:
        # convert structure embeds to packed tensor sequences for the decoder
        # If not on cuda device, will raise an error

        # according to tests, it seems that each structure must repeated for all
        # its corresponding sequences in the batch
        # get number of repeats for each structure
        repeats = torch.sum(segment_sizes != 0, dim=1)
        struct_sizes = torch.sum(
            structure_embeds[..., 0] != 0, dim=1
        )  # 0 is padding index

        s = torch.cat(
            [
                structure_embeds[i, : struct_sizes[i]].repeat(repeats[i], 1)
                for i in range(len(structure_embeds))
            ]
        )
        # repeat struct_sizes[i] for repetitions[i] times
        repeats_indices = torch.repeat_interleave(
            torch.arange(
                len(struct_sizes), dtype=torch.int64, device=struct_sizes.device
            ),  # torch.gather() requires dtype torch.int64
            repeats,
        )
        struct_sizes = torch.gather(struct_sizes, 0, repeats_indices)
        struct_sizes_cpu = struct_sizes.flatten().cpu().type(torch.int32)
        cu_seqlens_cpu = F.pad(
            struct_sizes_cpu.cumsum(dim=0, dtype=struct_sizes_cpu.dtype),
            (1, 0),
        )
        cu_seqlens = cu_seqlens_cpu.to(s.device)
        repeats_indices = repeats_indices.to(torch.int32)

        s = PackedTensorSequences(
            packed_tensor=s,
            positions=torch.cat(
                [
                    torch.arange(struct_size, dtype=s.dtype, device=s.device)
                    for struct_size in struct_sizes_cpu
                ]
            ),
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_s=struct_sizes_cpu.max(),
            to_paddedable=False,
            indices=repeats_indices,
            batch_size=None,
        )
        return s

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_sizes: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.LongTensor] = None,
        confidence: Optional[torch.Tensor] = None,
        structure_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the next token probability distributions.

        Examples:
          Example input with batch size 1

          xs: [$ A B * $ A B C * $ E F]
          segment_sizes: [[4, 5, 3]]

          Note that the last sequence in a sequence of sequences does not need to have a
          stop token.

        Args:
          xs:
            (B, L) sequence of sequences of tokens
          segment_sizes:
            (B, N) the lengths of each sequence in the sequence of sequences

        Returns:
          (B, L, V) logits of the next token probability distributions. Here, V is
          the vocabulary size

        """
        B, L = input_ids.size()
        (
            h,
            s,
            seqs_seqlens,
            seqs_seqlens_cpu,
            indices,
        ) = self._prepare_inputs_for_decoder(
            input_ids, segment_sizes, coords, padding_mask, confidence, structure_embeds
        )

        h = self.decoder(
            h,
            s,
            seqs_cu_seqlens=F.pad(
                seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
            ),
            seqs_cu_seqlens_cpu=F.pad(
                seqs_seqlens_cpu.cumsum(dim=0, dtype=seqs_seqlens_cpu.dtype),
                (1, 0),
            ),
        )

        logits = self.linear(self.norm(h.x))
        logits, _ = pad_input(logits, indices, B, L)  # (B,L,num_tokens)

        # add loss computation here for integration with Trainer
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.n_vocab),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            return loss, logits

        return logits

    def embed(
        self,
        xs: torch.Tensor,
        s: Optional[torch.Tensor],
        coords: Optional[torch.Tensor],
        segment_sizes: torch.Tensor,
        allow_cpu_offload: bool = False,
        pbar_position: Optional[int] = None,
    ) -> list[PackedTensorSequences]:
        """
        Returns the memory of each layer in a list. The memory is the input to the
        multi-sequence attention.

        Args:
          xs:
            (B, L) sequence of sequences
          segment_sizes:
            (B, N) the lengths of each sequence in the sequence of sequences
          allow_cpu_offload:
            whether or not memory should be offloaded to cpu if CUDA OOMs
          pbar_position:
            position of a tqdm progress bar if not None

        Returns:
          The memory. If allow_cpu_offload and there is insufficient GPU memory to
          store the tensors, the tensors will be stored in CPU memory instead.
        """
        h, s, seqs_seqlens, seqs_seqlens_cpu, _ = self._prepare_inputs_for_decoder(
            xs, segment_sizes, coords, structure_embeds=s
        )

        memory = []
        output_device: Optional[torch.device] = None
        if pbar_position is None:
            layers = self.decoder.layers
        else:
            layers = tqdm(
                self.decoder.layers,
                desc=f"[{pbar_position}] encoding",
                leave=False,
                position=pbar_position,
            )
        for layer in layers:
            layer: TieredGatedTransformerEncoderLayer
            try:
                h, _, (key, value) = layer.forward(
                    h,
                    s,
                    seqs_cu_seqlens=F.pad(
                        seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
                    ),
                    seqs_cu_seqlens_cpu=F.pad(
                        seqs_seqlens_cpu.cumsum(dim=0, dtype=seqs_seqlens.dtype),
                        (1, 0),
                    ),
                    return_memory=True,
                )
                if output_device is not None:
                    key.x = key.x.to(output_device)
                    value.x = value.x.to(output_device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and allow_cpu_offload:
                    if pbar_position is not None:
                        tqdm.write(
                            "OOMed during encoding, retrying by offloading to cpu"
                        )
                    torch.cuda.empty_cache()
                    output_device = torch.device("cpu")
                    for this_memory in memory:
                        this_memory.x = this_memory.x.to(output_device)
                    torch.cuda.empty_cache()
                    h, (_, _), (key, value) = layer.forward(
                        h,
                        seqs_cu_seqlens=F.pad(
                            seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
                        ),
                        seqs_cu_seqlens_cpu=F.pad(
                            seqs_seqlens_cpu.cumsum(dim=0, dtype=seqs_seqlens.dtype),
                            (1, 0),
                        ),
                        return_memory=True,
                    )
                    key.x = key.x.to(output_device)
                    value.x = value.x.to(output_device)
                else:
                    raise e
            memory.append(key)
            memory.append(value)
        return memory

    def logits(
        self,
        x: torch.Tensor,
        s: Optional[torch.Tensor],
        coords: Optional[torch.Tensor],
        memory: Optional[list[PackedTensorSequences]],
        preallocated_memory: bool = False,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the next token probability distributions given a precomputed memory
        (see self.embed and/or self.logits_allocate_memory).

        Args
          x:
            (B, L) sequence of sequences of tokens
          memory:
            output of self.embed
            if not preallocated_memory, has batch size 1 (it will be expanded if necessary)
            if memory is not on the same device as x, a copy of memory will be made to the
            device of x as necessary
          preallocated_memory:
            whether or not additional memory needed for this method was preallocated
            using self.logits_allocate_memory

        Returns:
          logits:
            (B, L, V) logits of the next token probability distributions. Here, V is
            the vocabulary size
        """
        B, L_x = x.size()

        x: PackedTensorSequences = PackedTensorSequences.pack_input(x.unsqueeze(2))
        x.x = self.token_embed.forward(x.x.squeeze(1))

        if s is not None:
            s = self.encoder_proj(s, coords=coords)
            # expand s along the batch dimension
            s = s.expand(B, -1, -1)
            s: PackedTensorSequences = PackedTensorSequences.pack_input(s)

        # print("x.x.shape", x.x.shape)
        # print("s.x.shape", s.x.shape)
        # print("x.cu_seqlens", x.cu_seqlens)
        # print("s.cu_seqlens", s.cu_seqlens)
        # print("x.batch_size", x.batch_size)
        # print("s.batch_size", s.batch_size)

        x = _apply_causal_prefix_attention(
            decoder=self.decoder,
            x=x,
            s=s,
            batch_size=B,
            length=L_x,
            self_memory=None,
            memory=memory,
            preallocated_memory=preallocated_memory,
        )

        embeddings = self.norm(x.x)
        logits = self.linear.forward(embeddings).view(B, L_x, -1)
        if not return_embeddings:
            return logits
        else:
            return logits, embeddings.view(B, L_x, -1)

    def sample(
        self,
        xs: torch.Tensor,
        segment_sizes: torch.Tensor,
        s: Optional[torch.Tensor] = None,  # structure embeddings
        temperature: float = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        maxlen: int = 1000,
        alphabet: Uniprot21 = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        ),
        remove_invalid: bool = True,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, float]:
        """Sample batch_size sequences.

        Note: this implementation is out of date
        """
        memory = self.embed(xs, s, segment_sizes)
        return self.sample_given_memory(
            memory=memory,
            s=s,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            maxlen=maxlen,
            alphabet=alphabet,
            remove_invalid=remove_invalid,
            batch_size=batch_size,
        )

    @torch.inference_mode()
    def sample_given_memory(
        self,
        memory: Optional[list[PackedTensorSequences]],
        s: Optional[torch.Tensor] = None,
        temperature: float = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        maxlen: int = 1000,
        alphabet: Uniprot21 = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        ),
        remove_invalid: bool = True,
        batch_size: int = 1,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Sample batch_size sequences from memory.

        Assumes memory represents one prompt, and samples each sequence from that one
        prompt.

        Note: this implementation is out of date

        Args:
          memory:
            Output of self.embed
            Must only describe one sequence of sequences i.e. have a batch size of 1
          temperature:
            Controls the randomness of the sampling by dividing the logits
          top_k:
            Controls the number of most probable tokens to consider at each step of
            sampling
            Default is None, which means all tokens are considered
          top_p:
            Controls the cumulative probability of the most probable tokens to consider
            at each step of sampling as in nucleus sampling
            Default is None, which is equivalent to the behavior with top_p=1
          maxlen:
            Maximum sequence length to sample, not including start and stop tokens
            Thus, returned sequences with have length up to maxlen+2, where the first
            token is the start token, and the last token is the stop token if the
            sequence terminates within maxlen tokens.
          alphabet:
            The alphabet encoding the sequence.
          remove_invalid:
            Whether or not to avoid sampling non-amino acids within a sequence.
          batch_size:
            Number of sequences to sample in parallel

        Returns:
          A tuple (sample_xs, sample_scores), where sample_xs is a list containing the
          sampled sequences as tensors encoded by alphabet, and sample_scores is a
          tensor containing the negative log likelihood of each sampled sequence.
        """
        criteria = nn.CrossEntropyLoss(
            ignore_index=alphabet.mask_token, reduction="none"
        )
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        invalid_tokens = torch.tensor(
            [alphabet.mask_token, alphabet.start_token, alphabet.gap_token],
            device=device,
        )
        nhead = self.decoder.layers[0].num_heads
        head_dim = self.decoder.layers[0].dim // nhead

        # initialize memory buffer
        buffer_size = (batch_size, maxlen + 2, nhead, head_dim)
        self_buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]
        buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]

        # initialize x
        current_token = (
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
            * alphabet.start_token
        )
        current_x = current_token
        current_position = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        current_position_int = 0
        current_logits: Optional[torch.Tensor] = None

        # sample rest of x
        sampled_xs, sampled_scores = [], []
        while True:
            # get logits for current x
            x: PackedTensorSequences = PackedTensorSequences.pack_input(
                current_token.unsqueeze(2),
                positions=current_position,
            )
            x.x = self.token_embed.forward(x.x.squeeze(1))
            x = _apply_causal_prefix_attention_buffered(
                decoder=self.decoder,
                x=x,
                s=s,
                memory=memory,
                self_buffer=[buf[:, : current_position_int + 1] for buf in self_buffer],
                buffer=[buf[:, : current_position_int + 1] for buf in buffer],
            )
            embeddings = self.norm(x.x)
            logits = self.linear.forward(embeddings).unsqueeze(1)

            # sample the next token
            next_token_logits = logits[:, -1].log_softmax(dim=1)
            if remove_invalid:
                next_token_logits[:, invalid_tokens] += -torch.inf
            next_token_logits /= temperature
            next_token_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            next_token = torch.multinomial(
                next_token_logits.float().softmax(dim=-1), 1
            ).flatten()

            # update state
            current_token = next_token.unsqueeze(1)
            current_x = torch.cat([current_x, current_token], dim=1)
            current_position = current_position + 1
            current_position_int += 1
            if current_logits is None:
                current_logits = logits
            else:
                current_logits = torch.cat([current_logits, logits], dim=1)

            # apply sampling termination conditions
            is_stop_batch_filter = (
                (next_token == alphabet.stop_token)
                if current_x.size(1) < maxlen + 2
                else torch.ones((current_x.size(0),), dtype=torch.bool, device=device)
            )
            if is_stop_batch_filter.sum() > 0:
                is_stop_batch_idxs = torch.where(is_stop_batch_filter)[0]
                not_is_stop_batch_idxs = torch.where(~is_stop_batch_filter)[0]

                sampled_xs.extend(current_x[is_stop_batch_idxs].unbind())
                sampled_scores.append(
                    -criteria.forward(
                        current_logits[is_stop_batch_idxs].transpose(1, 2),
                        current_x[is_stop_batch_idxs, 1:].cuda(),
                    )
                    .float()
                    .sum(dim=1)
                )
                if is_stop_batch_idxs.numel() == current_x.size(0):
                    break
                else:
                    # remove terminated sequences from state
                    _filter = not_is_stop_batch_idxs
                    current_token = current_token[_filter]
                    current_x = current_x[_filter]
                    current_position = current_position[_filter]
                    current_logits = current_logits[_filter]
                    for idx in range(len(self_buffer)):
                        self_buffer[idx] = self_buffer[idx][_filter]
                    for idx in range(len(buffer)):
                        buffer[idx] = buffer[idx][_filter]
        return sampled_xs, torch.hstack(sampled_scores)


def _apply_causal_prefix_attention(
    decoder: AdaptedTransformerEncoder,
    x: PackedTensorSequences,
    s: Optional[PackedTensorSequences],
    batch_size: int,
    length: int,
    self_memory: Optional[list[PackedTensorSequences]],
    memory: Optional[list[PackedTensorSequences]],
    preallocated_memory: bool,
) -> tuple[
    PackedTensorSequences,
    Optional[list[PackedTensorSequences]],
    Optional[list[PackedTensorSequences]],
]:
    B, L_x = batch_size, length

    for layer_idx, layer in enumerate(decoder.layers):
        layer: TieredGatedTransformerEncoderLayer

        # apply cross attention layer, without the need for memory, since
        # the memory correesponds to keys, values extracted from tiered attention
        # block of for retrieved sequences, which is not needed for the structure
        if s is not None:
            x = layer.gated_block(x, s, return_attention=False)[0]

        # not the best approach to rename layer, but allows for easy access to the children modules
        layer = layer.tiered_block
        # apply the self attention layer on the sequences independently
        x_norm = copy.copy(x)
        x_norm.x = layer.norm1.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.self_attn)
        if self_memory is not None:
            key_memory, value_memory = (
                copy.copy(self_memory[2 * layer_idx]),
                copy.copy(self_memory[2 * layer_idx + 1]),
            )
            key_memory.x, value_memory.x = (
                key_memory.x.to(x.x.device),
                value_memory.x.to(x.x.device),
            )
            key_memory, value_memory = _update_causal_prefix_memory(
                x_norm=x_norm,
                x_norm_km=x_norm_key.x,
                x_norm_vm=x_norm_value.x,
                key_memory=key_memory,
                value_memory=value_memory,
                batch_size=B,
                length=L_x,
                preallocated_memory=preallocated_memory,
            )
        else:
            key_memory, value_memory = x_norm_key, x_norm_value
        try:
            layer.self_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.self_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.self_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout1.forward(x2.x)

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        x_norm.x = layer.norm2.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.multihead_attn)
        if memory is not None:
            key_memory, value_memory = (
                copy.copy(memory[2 * layer_idx]),
                copy.copy(memory[2 * layer_idx + 1]),
            )
            key_memory.x, value_memory.x = (
                key_memory.x.to(x.x.device),
                value_memory.x.to(x.x.device),
            )
            key_memory, value_memory = _update_causal_prefix_memory(
                x_norm=x_norm,
                x_norm_km=x_norm_key.x,
                x_norm_vm=x_norm_value.x,
                key_memory=key_memory,
                value_memory=value_memory,
                batch_size=B,
                length=L_x,
                preallocated_memory=preallocated_memory,
            )
        else:
            key_memory, value_memory = x_norm_key, x_norm_value
        try:
            layer.multihead_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.multihead_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.multihead_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout2.forward(x2.x)

        x2 = layer.linear2(layer.dropout(F.gelu(layer.linear1(layer.norm3(x.x)))))
        x.x = x.x + layer.dropout3(x2)
    return x


def _apply_causal_prefix_attention_buffered(
    decoder: AdaptedTransformerEncoder,
    x: PackedTensorSequences,
    s: Optional[PackedTensorSequences],
    memory: Optional[list[PackedTensorSequences]],
    self_buffer: list[torch.Tensor],
    buffer: list[torch.Tensor],
) -> PackedTensorSequences:
    """
    does not implement self_memory b/c we won't be testing that code path atm
    also, it technically requires more calculations relating to position to make the
    code "look right", even though it is not necessary to do for RoPE
    """
    for layer_idx, layer in enumerate(decoder.layers):
        layer: TieredGatedTransformerEncoderLayer

        # apply cross attention layer, without the need for memory buffer
        if s is not None:
            x = layer.gated_block(x, s, return_attention=False)[0]

        layer = layer.tiered_block
        # apply the self attention layer on the sequences independently
        x_norm = copy.copy(x)
        x_norm.x = layer.norm1.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.self_attn)
        key_buffer, value_buffer = (
            self_buffer[2 * layer_idx],
            self_buffer[2 * layer_idx + 1],
        )
        key_buffer[:, -1], value_buffer[:, -1] = x_norm_key.x, x_norm_value.x
        key_memory = PackedTensorSequences.pack_input(key_buffer)
        value_memory = PackedTensorSequences.pack_input(value_buffer)
        key_memory.x = key_memory.x.unflatten(1, (x_norm_key.x.size(1), -1))
        value_memory.x = value_memory.x.unflatten(1, (x_norm_value.x.size(1), -1))
        try:
            layer.self_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.self_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.self_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout1.forward(x2.x)

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        x_norm.x = layer.norm2.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.multihead_attn)
        key_buffer, value_buffer = (
            buffer[2 * layer_idx],
            buffer[2 * layer_idx + 1],
        )
        key_buffer[:, -1], value_buffer[:, -1] = x_norm_key.x, x_norm_value.x
        if memory is not None:
            key_memory, value_memory = (
                copy.copy(memory[2 * layer_idx]),
                copy.copy(memory[2 * layer_idx + 1]),
            )
            key_memory.x, value_memory.x = (
                key_memory.x.to(x.x.device),
                value_memory.x.to(x.x.device),
            )
            _packed_sequence_append(key_memory, x=key_buffer)
            _packed_sequence_append(value_memory, x=value_buffer)
        else:
            # TODO: this code path may be untested
            key_memory = PackedTensorSequences.pack_input(key_buffer)
            value_memory = PackedTensorSequences.pack_input(value_buffer)
            key_memory.x = key_memory.x.unflatten(1, (x_norm_key.x.size(1), -1))
            value_memory.x = value_memory.x.unflatten(1, (x_norm_value.x.size(1), -1))
        try:
            layer.multihead_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.multihead_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.multihead_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout2.forward(x2.x)

        x2 = layer.linear2(layer.dropout(F.gelu(layer.linear1(layer.norm3(x.x)))))
        x.x = x.x + layer.dropout3(x2)
    return x
