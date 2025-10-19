"""Utils funtions and classes to combined ESM-IF1 model with PoET"""

import warnings

import esm
import torch
import torch.nn.functional as F
from esm.inverse_folding import util as if_util
from tqdm import tqdm

from poet.alphabets import Uniprot21

AAS = "ACDEFGHIKLMNPQRSTVWY"
aa2idx = {aa: i for i, aa in enumerate(AAS)}


class EsmIF1(object):
    def __init__(self, device: str = None):
        self.model, self.alphabet = self.load_model_and_alphabet(device)

    @staticmethod
    def load_model_and_alphabet(device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():
            # Ignore UserWarning about missing logistic regression weights for contacts prediction
            warnings.filterwarnings("ignore", category=UserWarning)
            model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        model = model.to(device)
        return model, alphabet

    @property
    def device(self):
        return next(self.model.parameters()).device

    def load_coords(self, pdbfile, chain="A"):
        return if_util.load_coords(str(pdbfile), chain)

    def get_encoder(self):
        return self.model.encoder

    def get_batch_converter(self):
        return if_util.CoordBatchConverter(self.alphabet)

    @torch.inference_mode()
    def embed(self, pdbfile, chain="A"):
        coords = self.load_coords(pdbfile, chain)[
            0
        ]  # returns tuple (coords, native_seq)
        # taken from if_util.get_encoder_output()
        batch_converter = if_util.CoordBatchConverter(self.alphabet)
        batch = [(coords, None, None)]
        coords, confidence, _, _, padding_mask = batch_converter(
            batch, device=self.device
        )
        encoder_out = self.model.encoder(
            coords, padding_mask, confidence, return_all_hiddens=False
        )
        # remove beginning and end (bos and eos tokens)
        return encoder_out["encoder_out"][0][1:-1, 0]

    @torch.inference_mode()
    def compute_logits(self, pdbfile, chain="A"):
        coords, native_seq = self.load_coords(pdbfile, chain)
        batch_converter = if_util.CoordBatchConverter(self.alphabet)
        batch = [(coords, None, native_seq)]
        coords, confidence, _, tokens, padding_mask = batch_converter(
            batch, device=self.device
        )
        prev_output_tokens = tokens[:, :-1].to(self.device)
        target = tokens.squeeze()[1:]
        target_padding_mask = target == self.alphabet.padding_idx
        logits, _ = self.model(coords, padding_mask, confidence, prev_output_tokens)
        return logits, target, target_padding_mask[0]

    @torch.inference_mode()
    def _compute_loss_single_sequence(self, coords, seq, batch_converter):
        batch = [(coords, None, seq)]
        coords, confidence, _, tokens, padding_mask = batch_converter(
            batch, device=self.device
        )
        prev_output_tokens = tokens[:, :-1].to(self.device)
        target = tokens[:, 1:]
        logits, _ = self.model(coords, padding_mask, confidence, prev_output_tokens)
        loss = F.cross_entropy(logits, target, reduction="mean").item()
        return loss

    @torch.inference_mode()
    def get_sequence_loss(self, coords, seq):
        return if_util.get_sequence_loss(self.model, self.alphabet, coords, seq)

    def compute_mutations_scores(self, pdbfile, chain="A"):
        coords, native_seq = self.load_coords(pdbfile, chain)
        scores = torch.zeros((len(self.aas), len(native_seq))).to(self.device)
        batch_converter = if_util.CoordBatchConverter(self.alphabet)
        wt_loss = self._compute_loss_single_sequence(
            coords, native_seq, batch_converter
        )
        mut_data = _get_all_mutations(native_seq)
        for i, mut_aa, mutated_seq in tqdm(mut_data):
            loss = self._compute_loss_single_sequence(
                coords, mutated_seq, batch_converter
            )
            scores[aa2idx[mut_aa], i] = wt_loss - loss
        return scores.cpu().numpy()

    def compute_proba(self, logits=None, pdbfile=None, chain="A"):
        if pdbfile is None and logits is None:
            raise ValueError(
                "At least one between `pdbfile` and `logits` must be provided"
            )
        if logits is None:
            logits = self.compute_logits(pdbfile, chain)[0]
        proba = F.softmax(logits, dim=-2).squeeze().cpu().numpy()
        return proba

    def sanitize_logits(self, logits, poet_alphabet: Uniprot21):
        def utf8_decode(token):
            return poet_alphabet.decode(token).decode("utf-8")

        poet_to_esm_tokens = {
            utf8_decode(poet_alphabet.mask_token): "<eos>",
            utf8_decode(poet_alphabet.start_token): "<mask>",
            utf8_decode(poet_alphabet.stop_token): "<eos>",
        }
        poet_vocab = poet_alphabet.chars[: poet_alphabet.size].tobytes().decode()
        esm_vocab = self.alphabet.all_toks
        reorder_idx = [
            esm_vocab.index(poet_to_esm_tokens.get(aa, aa)) for aa in poet_vocab
        ]
        return logits[..., reorder_idx], reorder_idx

    @torch.inference_mode()
    def sample(
        self,
        coords=None,
        pdb_file=None,
        partial_seq=None,
        num_seqs=1,
        temperature=1.0,
        confidence=None,
    ):
        """
        Samples sequences based on multinomial sampling (no beam search).
        Adapted from esm.inverse_folding.gvp_transformer.GVPTransformerModel.sample()

        Args:
            coords: Optional, L x 3 x 3 list representing one backbone. If not provided,
                `pdb_file` must be provided.
            pdb_file: Optional, path to PDB file to load coordinates from.
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        if coords is None:
            if pdb_file is None:
                raise ValueError("Either `coords` or `pdb_file` must be provided")
            coords = self.load_coords(pdb_file)[0]
        L = len(coords)
        # Convert to batch format, by taking the same pdb num_seqs times
        batch_converter = if_util.CoordBatchConverter(self.alphabet)
        batch_coords, confidence, _, _, padding_mask = batch_converter(
            [(coords, confidence, None) for _ in range(num_seqs)], device=self.device
        )

        # Run encoder only once
        encoder_out = self.model.encoder(batch_coords, padding_mask, confidence)

        # Start with prepend token
        mask_idx = self.alphabet.get_idx("<mask>")
        sampled_tokens = torch.full(
            size=(num_seqs, 1 + L), fill_value=mask_idx, dtype=int, device=self.device
        )
        sampled_tokens[:, 0] = self.alphabet.get_idx("<cath>")
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[:, i + 1] = self.alphabet.get_idx(c)

        # Save incremental states for faster sampling
        incremental_state = dict()

        # Decode one token at a time
        for i in range(1, L + 1):
            logits, _ = self.model.decoder(
                sampled_tokens[:, :i],
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(
                    probs, num_seqs, replacement=True
                ).squeeze(-1)
        sampled_seqs = sampled_tokens[:, 1:]

        # Convert back to string via lookup
        sequences = [
            "".join([self.alphabet.get_tok(a) for a in sampled_seq])
            for sampled_seq in sampled_seqs
        ]

        return sequences


# def _logits_to_logodds(logits, target):  # logits dim (..., aa, L)
#     logits = logits.squeeze()
#     target = target.squeeze()
#     scores = F.log_softmax(logits, dim=-2)
#     scores = (
#         scores - scores[target, range(target.shape[-1])]
#     )  # add negative log-likelihood
#     scores = scores.cpu().numpy()
#     return scores


def _get_all_mutations(seq, aas=AAS):
    mut_data = []
    for i, aa in enumerate(seq):
        for new_aa in aas:
            if new_aa != aa:
                mut_data.append((i, new_aa, seq[:i] + new_aa + seq[i + 1 :]))
    return mut_data
