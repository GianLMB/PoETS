"""
Utils for generation, with or without ESM-IF1 integration.
- If only ESM-IF1 is used for generation (full structure-based generation), the sampling follows
the ESM-IF1 sampling procedure, as implemented in `EsmIF1.sample()` method.
- If only PoET is used for generation (full homology-based generation), the sampling follows
the PoET sampling procedure, as implemented in `PoET.sample()` method.
- If the two are combined, we first sample sequences from ESM-IF1 (that have all the same length),
then we combined them with sequences retrieved from msa, and we sample with PoET using
the combined sequences for conditioning. This is and indirect way of conditioning PoET on
structure information, as the ESM-IF1 sequences are generated from structure, while allowing
generated sequences to have variable lengths.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from poet.alphabets import Uniprot21
from poet.inverse_folding.esm_if1_utils import EsmIF1
from poet.models.poet import PoET
from poet.msa.sampling import MSASampler, NeighborsSampler
from poet.variants import scoring_utils


def encode_esmif1_sequences(sequences: Sequence[str], alphabet: Uniprot21):
    sequences = [
        alphabet.encode(
            seq.replace("-", "")
            .replace(".", "")
            .encode()  # remove gaps and convert to bytes
        )
        for seq in sequences
    ]
    return sequences


def sample(
    poet_temperature: float = 0.6,
    poet_context_length: int = 61122,
    poet_max_similarity: float = 0.7,
    esmif1_temperature: float = 1.0,
    max_length: int = 1000,
    num_samples: int = 1,
    esm_if1_retrieving_ratio: float = 0.5,
    poet_model: Optional[PoET] = None,
    esmif1_model: Optional[EsmIF1] = None,
    msa_sequences: Optional[Sequence[str]] = None,
    pdb_file: Optional[Union[str, Path]] = None,
    seed: int = 188257,
):
    if poet_model is None and esmif1_model is None:
        raise ValueError("At least one of poet or esmif1 model(s) must be provided.")

    elif poet_model is None:
        # Only ESM-IF1 is used for generation
        if pdb_file is None:
            raise ValueError("pdb_file must be provided if poet is not provided.")

        return esmif1_model.sample(
            pdb_file=pdb_file,
            num_seqs=num_samples,
            temperature=esmif1_temperature,
        )

    else:
        # apply data preprocessing to msa sequences
        if msa_sequences is None:
            raise ValueError(
                "msa_sequences must be provided if PoET is used for generation."
            )

        alphabet = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        )
        sampler = MSASampler(
            method=NeighborsSampler(
                can_use_torch=False,
            ),
            max_similarity=poet_max_similarity,
        )
        sample_idxs = sampler.get_sample_idxs(
            msa=msa_sequences,
            gap_token=alphabet.gap_token,
            seed=seed,
        )  # code for sampler to be checked

        # create the sequence-of-sequences
        this_msa_sequences = scoring_utils.sample_msa_sequences(
            get_sequence_fn=lambda ii: msa_sequences[ii]
            .upper()
            .translate(None, delete=b"-"),
            sample_idxs=sample_idxs,
            max_tokens=poet_context_length,
            alphabet=alphabet,
            shuffle_seed=seed,
            truncate=False,
        )

        if esmif1_model is not None:
            esmif1_sequences = esmif1_model.sample(
                pdb_file=pdb_file,
                num_seqs=num_samples,
                temperature=esmif1_temperature,
            )
            esmif1_sequences = encode_esmif1_sequences(
                sequences=esmif1_sequences,
                alphabet=alphabet,
            )
            # combine the sequences
            combined_sequences = (
                this_msa_sequences,
                esmif1_sequences,
                esm_if1_retrieving_ratio,
                "TODO",
            )  # fucntion to be implemented
        else:
            combined_sequences = this_msa_sequences

        segment_sizes = torch.tensor([len(s) for s in combined_sequences]).cuda()
        combined_sequences = torch.cat(
            [torch.from_numpy(s).long() for s in combined_sequences]
        ).cuda()
        memory = poet_model.embed(
            combined_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
        )

        # sample from poet
        encoded_sequences = poet_model.sample(
            memory=memory,
            temperature=poet_temperature,
            maxlen=max_length,
            alphabet=alphabet,
            remove_invalid=True,
        )

        # decode the sequences
        decoded_sequences = [alphabet.decode(seq).decode() for seq in encoded_sequences]

        return decoded_sequences
