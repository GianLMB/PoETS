"""Functions for scoring variants with PoET."""

import string
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from poet.alphabets import Uniprot21
from poet.fasta import parse_stream
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet import PoET

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()
T = TypeVar("T", np.ndarray, torch.Tensor)


def append_startstop(x: T, alphabet: Uniprot21, stop=True) -> T:
    # esm_if1 does not have an eos token, so we don't append eos token for variants to score
    # for the computation of the logits (but we append it for the conditioning sequences extracted
    # from the MSA, as they are used to compute the embeddings)
    x_ndim = x.ndim
    assert x_ndim in {1, 2}
    if x_ndim == 1:
        x = x[None, :]

    if isinstance(x, torch.Tensor):
        empty_func = torch.empty
    else:
        empty_func = np.empty
    x_ = empty_func((x.shape[0], x.shape[1] + 1 + int(stop)), dtype=x.dtype)
    x_[:, 0] = alphabet.start_token
    if stop:
        x_[:, -1] = alphabet.stop_token
        x_[:, 1:-1] = x
    else:
        x_[:, 1:] = x
    if x_ndim == 1:
        x_ = x_.flatten()
    return x_


def get_seqs_from_fastalike(
    filepath: Path, upper: bool = False, max_sequences: int = 40000
) -> list[bytes]:
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=upper)][
        :max_sequences
    ]


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> np.ndarray:
    return np.vstack(
        [
            alphabet.encode(s.translate(None, delete=ASCII_LOWERCASE_BYTES))
            for s in msa_sequences
        ]
    )


def sample_msa_sequences(
    get_sequence_fn: Callable[[int], bytes],
    sample_idxs: Sequence[int],
    max_tokens: int,
    alphabet: Uniprot21,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    truncate: bool = True,
) -> list[np.ndarray]:
    assert alphabet.start_token != -1
    assert alphabet.stop_token != -1
    if not shuffle:
        assert shuffle_seed is None

    seqs, total_tokens = [], 0
    for idx in sample_idxs:
        next_sequence = get_sequence_fn(idx)
        seqs.append(append_startstop(alphabet.encode(next_sequence), alphabet=alphabet))
        total_tokens += len(seqs[-1])
        if total_tokens > max_tokens:
            break

    # shuffle order and truncate to max tokens
    if shuffle:
        rng = (
            np.random.default_rng(shuffle_seed)
            if shuffle_seed is not None
            else np.random
        )
        final_permutation = rng.permutation(len(seqs))
    else:
        final_permutation = np.arange(len(seqs))
    final_seqs, total_tokens = [], 0
    for seq in [seqs[i] for i in final_permutation]:
        if truncate and (total_tokens + len(seq) > max_tokens):
            seq = seq[: max_tokens - total_tokens]
        total_tokens += len(seq)
        final_seqs.append(seq)
        if total_tokens >= max_tokens:
            break
    return final_seqs


def jit_warmup(embedding_model: PoET, alphabet: Uniprot21):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().cuda()
    segment_sizes = torch.tensor(segment_sizes).long().cuda()
    _ = embedding_model.embed(x.unsqueeze(0), segment_sizes.unsqueeze(0))


def trace_to_temperature(
    tjet_trace: torch.Tensor, low: float = 0.5, high: float = 2.0, mode="sigmoid"
) -> torch.Tensor:
    tjet_trace = (
        1 - tjet_trace
    )  # invert the trace, low values should be high temperatures
    if mode == "sigmoid":
        tjet_trace = tjet_trace * (high - low)
        return (
            F.sigmoid(tjet_trace - tjet_trace.median()) * 2
        )  # center the sigmoid around the mdeian value
    elif mode == "linear":
        return tjet_trace * (high - low) + low
    elif mode == "exp":
        return torch.exp(tjet_trace - 0.5) * (high - low) + low
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _get_logps_tiered_fast(
    memory: Optional[list[PackedTensorSequences]],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    if_logits: Optional[torch.Tensor] = None,
    tjet_trace: Optional[np.ndarray] = None,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    max_variant_length = max(len(v) for v in variants)
    memory = model.logits_allocate_memory(
        memory=memory,
        batch_size=batch_size,
        length=max_variant_length - 1,  # discount stop token
    )
    criteria = nn.CrossEntropyLoss(ignore_index=alphabet.mask_token, reduction="none")
    logps = []
    if tjet_trace is not None:
        tjet_trace = torch.from_numpy(tjet_trace).cuda()
        # update logits using tjet trace as temperature-like parameter
        # temperature = 2 - tjet_trace
        # trace_to_temperature(tjet_trace, low=0.0, high=2.0, mode="sigmoid")
        # temperature = 1 / (tjet_trace + 0.01)
        # temperature = 1 / (1  + tjet_trace - tjet_trace.mean())
        temperature = 1 - tjet_trace  # + tjet_trace.mean()
        temperature = temperature.unsqueeze(-1)
    if pbar_position is not None:
        pbar = trange(
            0,
            len(variants),
            batch_size,
            desc=f"[{pbar_position}] decoding",
            leave=False,
            position=pbar_position,
        )
    else:
        pbar = range(0, len(variants), batch_size)
    for start_idx in pbar:
        this_variants = variants[start_idx : start_idx + batch_size]
        this_variants = pad_sequence(
            [torch.from_numpy(v).long() for v in this_variants],
            batch_first=True,
            padding_value=alphabet.mask_token,
        )
        if this_variants.size(1) < max_variant_length:
            this_variants = F.pad(
                this_variants,
                (0, max_variant_length - this_variants.size(1)),
                value=alphabet.mask_token,
            )
        assert (this_variants == alphabet.gap_token).sum() == 0
        this_variants = this_variants.cuda()
        logits = model.logits(this_variants[:, :-1], memory, preallocated_memory=True)

        # expand if logits to batch size and average over poet logits
        if if_logits is not None:
            logits = (logits + if_logits.expand(this_variants.size(0), -1, -1)) / 2
        if tjet_trace is not None:
            logits = logits / temperature
        targets = this_variants[:, 1:]  # remove eos token
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        # epi_score = -criteria.forward(epi_logits.transpose(1, 2), targets).float().sum(dim=1)
        # if tjet_trace is not None:
        #     tjet_score = (score * F.softmax(torch.from_numpy(tjet_trace), dim=-1)).sum(axis=1)
        #     # score = tjet_score
        #     # option to blend with the original score
        #     alpha = 0.5
        #     score = alpha * score.mean(axis=1) + (1 - alpha) * tjet_score
        # else:
        #     score = score.mean(axis=1)
        logps.append(
            score.cpu().numpy()
        )  # at the end, perform average between the two scores
    return np.hstack(logps)


def get_logps_tiered_fast(
    msa_sequences: Sequence[np.ndarray],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    if_logits: Optional[torch.Tensor] = None,
    tjet_trace: Optional[np.ndarray] = None,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).cuda()
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).cuda()
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
            pbar_position=pbar_position,
        )
    else:
        memory = None

    return _get_logps_tiered_fast(
        memory=memory,
        variants=variants,
        model=model,
        if_logits=if_logits,
        tjet_trace=tjet_trace,
        batch_size=batch_size,
        alphabet=alphabet,
        pbar_position=pbar_position,
    )
