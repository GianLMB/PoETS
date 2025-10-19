"""Flash attention implementation of Flamingo layers attending to PoET representations."""

import copy
from typing import Optional, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

from poet.inverse_folding.esm3_utils import build_affine3d_from_coordinates
from poet.inverse_folding.modules.geometric_attention import (
    GeometricReasoningOriginalImpl,
)
from poet.models.modules.attention_flash import FlashMultiheadAttention
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.modules.transformer import TransformerEncoder
from poet.models.modules.transformer_rotary import TieredRotaryTransformerEncoderLayer

T = TypeVar("T", Tensor, PackedTensorSequences)


class GatedXattentionBlock(nn.Module):
    """Flamingo-like cross attention block with tanh gating, designed in PoET layers fashion."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,
        dropout=0,
        use_qkv_bias=False,
        batch_first=False,
        alpha_init=1e-5,
    ):
        super().__init__()

        assert batch_first, (
            "Flash Attention requires batch first, make sure your transformer uses"
            " batch_first=True."
        )

        self.dim = d_model
        if dim_feedforward is None:
            dim_feedforward = (
                4 * d_model
            )  # maybe can be reduced to 2 to reduce complexity
        self.dim_feedforward = dim_feedforward
        self.num_heads = nhead

        self.cross_attn = self._init_cross_mha_module(
            d_model,
            nhead,  # might be modified
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
        )

        self.alpha_cross_attn = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_ffwd = nn.Parameter(torch.full((1,), alpha_init))

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        nn.init.constant_(self.linear2.weight, 0.0)  # ?
        nn.init.constant_(self.linear2.bias, 0.0)  # ?

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _init_cross_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
    ):
        """
        Initialize the multi-head attention module used for each sequence independently.
        """
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=False,  # cross attention
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=False,
        )

    def reset_parameters(self):
        self.cross_attn.reset_parameters()
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward_packed(
        self,
        x: PackedTensorSequences,
        s: PackedTensorSequences,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[PackedTensorSequences, tuple[PackedTensorSequences, Tensor]]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: PackedTensorSequences of the individual sequences.
        seqs_cu_seqlens: (B+1,) the cumulative lengths of the sequences-of-sequences.
        src_key_padding_mask: B x N x L x K where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences, and K is the hidden dim
        """

        # ln1
        x_norm = copy.copy(x)
        x_norm.x = self.norm1(x.x)
        # Apply cross attention with q from x and k, v from s on sequences independently
        x2, attn_cross = self.cross_attn(
            query=x_norm,
            key=s,
            value=s,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_attention,
        )
        # # tanh gating
        x = copy.copy(x)
        x.x = x.x + self.dropout1(x2.x) * F.tanh(self.alpha_cross_attn)
        # ln2
        x_norm = copy.copy(x)
        x_norm.x = self.norm2(x.x)
        # feedforward, Sq ReLU found to be better than GELU in Flamingo
        # x2 = self.linear2(self.dropout(F.gelu(self.linear1(x_norm.x))))
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x_norm.x)) ** 2))
        # tanh gating
        x = copy.copy(x)
        x.x = x.x + self.dropout2(x2) * F.tanh(self.alpha_ffwd)

        output = (x,)
        if return_attention:
            output += (attn_cross,)
        return output

    def forward_padded(
        self,
        x: Tensor,
        s: Tensor = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: Tensor of the individual sequences. Size B x N x L x K
        s: Tensor of the structure encodings. Size B x L_struct x K
        src_key_padding_mask: B x N x L where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences
        """
        B, N, L, K = x.size()
        # sequence-independent attention
        x = x.view(B * N, L, K)
        x_norm = self.norm1(x)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(B * N, L)
        # Apply cross attention with q from x and k, v from s on sequences independently
        x2, attn_cross = self.cross_attn(
            q=x_norm,
            k=s,
            v=s,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_attention,
        )
        # # tanh gating
        x = x + self.dropout1(x2) * F.tanh(self.alpha_cross_attn)
        # ln2 and reshape
        x_norm = self.norm2(x)
        x = x.view(B, N, L, K)
        # feedforward
        x2 = self.linear2(self.dropout(F.gelu(self.linear1(x_norm))))
        # tanh gating
        x = x + self.dropout2(x2) * F.tanh(self.alpha_ffwd)

        output = (x,)
        if return_attention:
            output += (attn_cross,)
        return output

    def forward(
        self,
        x: T,  # sequences
        s: T,  # structure encodings
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[T, tuple[T, Tensor]]:
        """
        See self.forward_padded and self.forward_packed for information about x,
        seqs_cu_seqlens, src_mask, and src_key_padding_mask.

        By default, only returns the output of the layer: (out)

        If return_attention=True, additionally returns the cross-attention matrix: (out, attn_cross)
        """
        assert type(x) == type(
            s
        ), "x and s must be of the same type, either Tensor or PackedTensorSequences"
        fn = self.forward_padded
        if type(x) is PackedTensorSequences:
            fn = self.forward_packed
        return fn(
            x,
            s=s,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_attention=return_attention,
        )


class AdaptedTransformerEncoder(TransformerEncoder):
    def __init__(
        self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False
    ):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor)

    def forward(
        self,
        x,
        s,
        src_mask=None,
        src_key_padding_mask=None,
        return_attention=False,
        activation_checkpointing=False,
        **kwargs,
    ):
        attn = []
        for layer in self.layers:
            if not activation_checkpointing:
                x = layer(
                    x,
                    s,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=return_attention,
                    **kwargs,
                )
            else:
                x = checkpoint.checkpoint(
                    layer,
                    x,
                    s,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=return_attention,
                    **kwargs,
                    use_reentrant=False,
                )
            if return_attention:
                x, a = x
                attn.append(a)

        if return_attention:
            return x, attn

        return x


class TieredGatedTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that operates on sequences-of-sequences. Processes sequences
    in two attention blocks analogously to transformer decoder layers. The first attention
    layer only attends within each sequence. The second attention layer also attends to
    other sequences within each sequence-of-sequences.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,
        activation=nn.GELU(),
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=True,
        use_multi_rotary=True,
        alpha_init=1e-5,
    ):
        super().__init__()

        assert batch_first, (
            "Flash Attention requires batch first, make sure your transformer uses"
            " batch_first=True."
        )

        self.gated_block = GatedXattentionBlock(
            d_model,
            nhead,
            dim_feedforward=d_model // 2,  # keep same dimension to reduce complexity
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            alpha_init=alpha_init,
        )

        self.tiered_block = TieredRotaryTransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=None,
            rotary_force_fp32=None,
            use_multi_rotary=use_multi_rotary,
        )

    def reset_parameters(self):
        self.gated_block.reset_parameters()
        self.tiered_block.reset_parameters()

    def forward(
        self,
        x: T,
        s: Optional[T],
        seqs_cu_seqlens: Tensor,
        seqs_cu_seqlens_cpu: Optional[Tensor],
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        PackedTensorSequences,
        tuple[PackedTensorSequences, tuple[Tensor, Tensor]],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            tuple[PackedTensorSequences, PackedTensorSequences],
        ],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            tuple[Optional[PackedTensorSequences], Optional[PackedTensorSequences]],
            PackedTensorSequences,
        ],
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: PackedTensorSequences of the individual sequences.
        seqs_cu_seqlens: (B+1,) the cumulative lengths of the sequences-of-sequences.
        src_key_padding_mask: B x N x L x K where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences, and K is the hidden dim
        """

        if s is not None:
            cross_att_output = self.gated_block(
                x,
                s,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=return_attention,
            )
            x = cross_att_output[0]

        output_tiered = self.tiered_block(
            x,
            seqs_cu_seqlens=seqs_cu_seqlens,
            seqs_cu_seqlens_cpu=seqs_cu_seqlens_cpu,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_attention=return_attention,
            return_memory=return_memory,
        )

        if return_attention:
            # prepend cross-attention to attention tuple
            output_tiered[1] = (cross_att_output[1],) + output_tiered[1]

        return output_tiered


class TieredMultiGatedTransformerEncoderLayer(TieredRotaryTransformerEncoderLayer):
    """
    Transfoermer Encoder layer that applies gated cross attention
    with structure encodings before the multi-attention block.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,
        activation=nn.GELU(),
        dropout=0,
        use_qkv_bias=False,
        batch_first=False,
        causal=True,
        self_causal: Optional[bool] = None,
        alpha_init: float = 1e-5,
        rotary_scale=None,
        rotary_force_fp32=None,
        use_multi_rotary=True,
        **kwargs,
    ):
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        self.use_multi_rotary = use_multi_rotary
        self.alpha_init = alpha_init

        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            self_causal=self_causal,
            **kwargs,
        )

        self.gated_block = GatedXattentionBlock(
            d_model,
            nhead,
            dim_feedforward=d_model // 2,  # keep same dimension to reduce complexity
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            alpha_init=alpha_init,
        )

    def _transform_tiered_input(
        self,
        x: PackedTensorSequences,
        seqs_cu_seqlens: Tensor,
        seqs_cu_seqlens_cpu: Optional[Tensor],
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> PackedTensorSequences:
        """
        Return input to the tiered attention block, but without the norm applied.
        Also, do not consider the patch case (that is not used in PoETS model)
        """
        x2 = copy.copy(x)
        x2.x = self.norm2(x.x)
        x2.cu_seqlens = seqs_cu_seqlens  # "reshape" the packed sequences
        if seqs_cu_seqlens_cpu is not None:
            x2.cu_seqlens_cpu = seqs_cu_seqlens_cpu
        else:
            x2.cu_seqlens_cpu = seqs_cu_seqlens.cpu()
        x2.max_s = x2.cu_seqlens_cpu.max()
        if x2.to_paddedable:
            seqs_seqlens = seqs_cu_seqlens.diff()
            x2.indices = x2.compute_indices(seqs_seqlens)
            x2.batch_size = seqs_seqlens.numel()
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(
                -1, src_key_padding_mask.size(-1)
            )
        return x2, src_key_padding_mask

    def forward_packed(
        self,
        x: PackedTensorSequences,
        s: Optional[PackedTensorSequences],
        seqs_cu_seqlens: Tensor,
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
        return_memory: bool = False,
    ):
        if return_multi_attention:
            return_attention = True
        if return_self_attention:
            return_attention = True

        # apply the self attention layer on the sequences independently
        x_norm = copy.copy(x)
        x_norm.x = self.norm1(x.x)
        x2, attn_self = self.self_attn(
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_self_attention,
        )
        x = copy.copy(x)
        x.x = x.x + self.dropout1(x2.x)

        # apply transformation to self attentionoutput to prepare for tiered
        # attention block
        # apply the sequence-of-sequence attention layer on the reshaped sequences

        # apply cross attention layer, no norm applied
        x2, src_key_padding_mask = self._transform_tiered_input(
            x, seqs_cu_seqlens, seqs_cu_seqlens_cpu, src_key_padding_mask
        )
        if s is not None:
            output_gated = self.gated_block(
                x2,
                s,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=return_attention,
            )
            x = output_gated[0]

        # apply tiered attention block
        x_norm = copy.copy(x2)  # copy attributes of x2, except for x attribute
        x_norm.x = self.norm2(x.x)

        if not return_memory:
            x2, attn = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
            )
            key, value = None, None
        else:
            x2, attn, (_, key, value) = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
                return_projs=return_memory,
            )

        x = copy.copy(x)
        x.x = x.x + self.dropout2(x2.x)

        x2 = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(x.x)))))
        x.x = x.x + self.dropout3(x2)

        if return_attention:
            return x, (attn_self, attn)
        if return_memory:
            return x, (attn_self, attn), (key, value)
        return x

    def forward_padded(
        self,
        x: Tensor,
        s: Optional[Tensor] = None,
        seqs_cu_seqlens: Optional[Tensor] = None,
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        Tensor,
        tuple[Tensor, tuple[Tensor, Tensor]],
        tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]], Tensor],
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: Tensor of the individual sequences. Size B x N x L x K
        src_key_padding_mask: B x N x L where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences
        """
        if return_multi_attention:
            return_attention = True
        if return_self_attention:
            return_attention = True

        B, N, L, K = x.size()
        # sequence-independent attention
        x = x.view(B * N, L, K)
        x_norm = self.norm1(x)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(B * N, L)
        x2, attn_self = self.self_attn(
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_self_attention,
        )
        x = x + self.dropout1(x2)

        # prepare input for tiered attention block
        x = x.view(B, N * L, K)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(B, N * L)

        if s is not None:
            # apply cross attention layer, no norm applied
            x, _ = self.gated_block(
                x,
                s,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=return_attention,
            )

        # apply tiered attention block
        x_norm = self.norm2(x)
        if not return_memory:
            x2, attn = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
            )
        else:
            x2, attn, (_, key, value) = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
                return_projs=return_memory,
            )
        x = x + self.dropout2(x2)

        # reshape x back
        x = x.view(B, N, L, K)

        x2 = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(x)))))
        x = x + self.dropout3(x2)

        if return_attention:
            return x, (attn_self, attn)
        if return_memory:
            return x, (attn_self, attn), (key, value)
        return x

    def forward(
        self,
        x: T,
        s: Optional[T] = None,
        seqs_cu_seqlens: Optional[Tensor] = None,
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        T,
        tuple[T, tuple[Tensor, Tensor]],
        tuple[T, tuple[Optional[Tensor], Optional[Tensor]], T],
    ]:
        """
        See self.forward_padded and self.forward_packed for information about x,
        seqs_cu_seqlens, src_mask, and src_key_padding_mask.

        By default, only returns the output of the layer: (out)

        If return_attention=True, additionally returns the self and multi-sequence
        attention matrices: (out, (attn_self, attn))

        If return_memory=True, additionally returns the "memory" (input to multi-
        sequence attention): (out, (attn_self, attn), memory)
        Here, attn_self and attn may be None depending on the value of
        return_attention.
        """
        fn = self.forward_padded
        if type(x) is PackedTensorSequences:
            assert seqs_cu_seqlens is not None
            fn = self.forward_packed
        return fn(
            x,
            s,
            seqs_cu_seqlens=seqs_cu_seqlens,
            seqs_cu_seqlens_cpu=seqs_cu_seqlens_cpu,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_self_attention=return_attention,
            return_multi_attention=return_attention,
            return_memory=return_memory,
        )


class EncoderProjector(nn.Module):
    """FFW projector for the encoder output to the decoder cross-attention."""

    def __init__(self, d_encoder, d_model, dropout=0, add_gaussian_noise=True):
        super().__init__()
        self.linear1 = nn.Linear(d_encoder, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.add_gaussian_noise = add_gaussian_noise

    def forward(self, s: Tensor, **kwargs) -> Tensor:  # B x L x K
        if self.add_gaussian_noise and self.training:
            s = s + self._compute_noise(s)

        s = self.linear1(s)
        s = self.dropout(F.gelu(s))
        s = self.linear2(s)
        return s  # to be packed later

    def _compute_noise(self, s: Tensor) -> Tensor:
        # add gaussian noise to non-zeros elements (padding) to avoid overfitting
        padding_mask = torch.all(s == 0, dim=-1)
        # set variance to 5% of variance of the tensor
        std = (s[~padding_mask].var() * 0.05) ** 0.5
        noise = torch.randn_like(s) * std
        return noise * padding_mask.unsqueeze(-1)


class GeometricAttentionProjector(EncoderProjector):
    """Geometric transfoemr layer."""

    def __init__(self, d_encoder, d_model, dropout=0, add_gaussian_noise=True):
        super().__init__(d_encoder, d_model, dropout, add_gaussian_noise)
        self.norm = nn.LayerNorm(d_encoder)
        # In esm3, the number of head is set to 128, for an hidden dimension of 1024
        # If we use encodings from ESM-IF1, we might also set it to 64
        self.geom_att = GeometricReasoningOriginalImpl(d_encoder, v_heads=64)

    def forward(self, s: Tensor, coords: Tensor) -> Tensor:  # B x L x K
        if self.add_gaussian_noise and self.training:
            s = s + self._compute_noise(s)

        # pad coordinates with inf if it wasn't done before
        if coords.size(1) != s.size(1):
            coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)

        # Apply geometric attention
        s2 = self.norm(s)
        affine, affine_mask = build_affine3d_from_coordinates(coords)
        # affine.trans = affine.trans.to(s2.dtype)
        # affine_mask = affine_mask.to(s2.dtype)
        s2 = self.geom_att(s2, affine, affine_mask)
        s = s + s2

        # Apply FFW projector (without residual connection)
        s = self.linear2(self.dropout(F.gelu(self.linear1(s))))
        return s  # to be packed later
