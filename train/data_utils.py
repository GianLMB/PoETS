"""Data utils for training Flamingo model"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from poet.alphabets import Alphabet, Uniprot21
from poet.msa.sampling import RandomSampler


class FlamingoDataset(Dataset):
    """Dataset for training Flamingo model."""

    def __init__(
        self,
        funfam_ids: Sequence[str],
        msa_dir: Union[str, Path],
        encoded_pdb_dir: Union[str, Path],
        alphabet: Optional[Alphabet] = None,
        context_length: int = 8000,
    ):
        self.funfam_ids = funfam_ids
        self.msa_dir = Path(msa_dir)
        self.encoded_pdb_dir = Path(encoded_pdb_dir)

        if alphabet is None:
            alphabet = Uniprot21(
                include_gap=True, include_startstop=True, distinct_startstop=True
            )
        self.alphabet = alphabet
        self.sampler = RandomSampler()
        self.context_length = context_length

        # load all msa sequences in memory to accelerate sampling
        self.msas = [
            self._parse_fasta(self.msa_dir / f"{funfam_id}.fasta")
            for funfam_id in tqdm(funfam_ids, desc="Loading MSAs")
        ]
        self.msa_lengths = [len(msa) for msa in self.msas]

    def __len__(self):
        return len(self.funfam_ids)

    def __getitem__(self, idx: int, seed: Optional[int] = None):
        funfam_id = self.funfam_ids[idx]
        msa = self.msas[idx]
        if seed is None:
            seed = np.random.randint(1_000_000)
        # shuffle indexes
        sample_idxs = self.sampler.get_sample_idxs(msa=msa, seed=seed)
        # Get samples until context_length is reached
        tokens = []
        segment_sizes = []
        total_tokens = 0
        for i in sample_idxs:
            next_sequence = msa[i]
            segment_size = len(next_sequence)
            total_tokens += segment_size
            segment_sizes.append(segment_size)
            tokens.append(next_sequence)
            if total_tokens > self.context_length:
                segment_sizes[-1] -= total_tokens - self.context_length
                break
        tokens = np.concatenate(tokens)[: self.context_length]
        tokens = torch.from_numpy(tokens).long().unsqueeze(0)
        segment_sizes = torch.tensor(segment_sizes).unsqueeze(0)
        # structure encoding loaded on the fly due to large memory footprint
        struct_data = torch.load(self.encoded_pdb_dir / f"{funfam_id}.pt")
        # sizes: tokens: (1, L_SoS), structure_encoding: (1, L, D)
        structure_encoding = struct_data["embed"].unsqueeze(0)
        coords = struct_data.get("coords")
        if coords is not None:
            coords = coords.unsqueeze(0)
        return tokens, segment_sizes, structure_encoding, coords

    def get_collator(self):
        return FlamingoCollator(self.alphabet)

    def _append_startstop(self, x: np.ndarray) -> np.ndarray:
        x_ = np.empty((len(x) + 2), dtype=x.dtype)
        x_[0] = self.alphabet.start_token
        x_[-1] = self.alphabet.stop_token
        x_[1:-1] = x
        return x_

    def _parse_fasta(self, file: Union[str, Path]) -> list[np.ndarray]:
        """Parse FASTA file and return list of byte-encoded sequences, without gaps."""
        with open(file, "rb") as f:
            return [
                self._append_startstop(
                    self.alphabet.encode(seq.translate(None, delete=b"-"))
                )
                for seq in parse_stream_fasta(f)
            ]


class FlamingoCollator(object):
    """Collator class for FlamingoDataset."""

    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet

    def __call__(self, batch):
        tokens, segment_sizes, structure_encodings, coords = zip(*batch)

        # Pad sequences with mask token to the same length
        max_length = max(seq.size(1) for seq in tokens)
        padded_tokens = torch.full(
            (len(tokens), max_length), self.alphabet.mask_token, dtype=torch.long
        )
        for i, seq in enumerate(tokens):
            padded_tokens[i, : seq.size(1)] = seq

        # Pad segment sizes with zeros to the same length
        max_length = max(size.size(1) for size in segment_sizes)
        padded_segment_sizes = torch.zeros(
            (len(segment_sizes), max_length), dtype=torch.long
        )
        for i, size in enumerate(segment_sizes):
            padded_segment_sizes[i, : size.size(1)] = size

        # Pad structure encodings with zeros to the same length
        max_length = max(enc.size(1) for enc in structure_encodings)
        padded_encodings = torch.zeros(
            (len(structure_encodings), max_length, structure_encodings[0].size(2)),
            dtype=structure_encodings[0].dtype,
        )
        for i, enc in enumerate(structure_encodings):
            padded_encodings[i, : enc.size(1), :] = enc

        # Pad coords with inf to the same length
        # coords mask is constructed with
        if coords[0] is not None:
            max_length = (
                max_length - 2
            )  # same as structure encoding, but without start and stop tokens
            padded_coords = torch.full(
                (len(coords), max_length, 3, 3),
                torch.finfo(torch.float).max,  # (batch, L, BB, N_dims)
            )
            for i, coord in enumerate(coords):
                padded_coords[i, : coord.size(1), :, :] = coord
        else:
            padded_coords = None

        # labels are the same as input tokens, shifted by one token for language modeling
        # and with -100 for padding tokens
        labels = padded_tokens.clone()
        labels[
            padded_tokens == self.alphabet.mask_token
        ] = -100  # ignore loss for padding tokens
        labels = labels[:, 1:]

        # return output as dictionary for HuggingFace Trainer compatibility
        return {
            "input_ids": padded_tokens,
            "segment_sizes": padded_segment_sizes,
            "structure_embeds": padded_encodings,
            "coords": padded_coords,
            "labels": labels,  # shift by one token for language modeling
        }


def parse_stream_fasta(f, comment=b"#"):
    sequence = []
    append = sequence.append
    join = b"".join

    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b">"):
            if sequence:
                yield join(sequence)
            sequence = []
            append = sequence.append
        else:
            append(line)

    if sequence:
        yield join(sequence)


def get_data_splits(
    funfam_ids_path: Union[str, Path],
    msa_dir: Union[str, Path],
    encoded_pdb_dir: Union[str, Path],
    alphabet: Optional[Alphabet] = None,
    context_length: int = 8000,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
):
    """Split data into train, validation, and test sets."""
    assert train_frac + val_frac + test_frac == 1.0

    with open(funfam_ids_path, "r") as f:
        funfam_ids = f.read().splitlines()

    np.random.seed(seed)
    np.random.shuffle(funfam_ids)

    num_train = int(train_frac * len(funfam_ids))
    num_val = int(val_frac * len(funfam_ids))

    train_ids = funfam_ids[:num_train]
    val_ids = funfam_ids[num_train : num_train + num_val]
    test_ids = funfam_ids[num_train + num_val :]

    train_dataset = FlamingoDataset(
        train_ids,
        msa_dir,
        encoded_pdb_dir,
        alphabet=alphabet,
        context_length=context_length,
    )
    val_dataset = FlamingoDataset(
        val_ids,
        msa_dir,
        encoded_pdb_dir,
        alphabet=alphabet,
        context_length=context_length,
    )
    test_dataset = FlamingoDataset(
        test_ids,
        msa_dir,
        encoded_pdb_dir,
        alphabet=alphabet,
        context_length=context_length,
    )

    return train_dataset, val_dataset, test_dataset
