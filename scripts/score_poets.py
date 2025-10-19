"""Score the variants in the ProteinGym dataset using PoET / PoET+ESM-IF1 model."""

import argparse
import itertools
import os
from pathlib import Path
from typing import Optional, Sequence

import dotenv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

from poet.alphabets import Uniprot21
from poet.inverse_folding.esm_if1_utils import EsmIF1
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet_struct import PoETS
from poet.msa.sampling import MSASampler, NeighborsSampler
from poet.variants import scoring_utils

PBAR_POSITION = 1
_SEED = 188257


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int)
    parser.add_argument("--ckpt_path", type=str, default="data/poet.ckpt")
    parser.add_argument("--add_esmif1", action="store_true")
    parser.add_argument("--output_dir", type=str, default="PoETS")
    parser.add_argument("--relative_to_wt", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--seed", type=int, default=188257)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run only 1/15 params from the msa sampling and filtering ensemble",
    )
    args = parser.parse_args()
    args.index = int(args.index)
    args.output_dir = Path(os.getenv("BASE_OUTPUT_PATH")) / args.output_dir
    return args


@torch.inference_mode()
def embed_struct(if_model: EsmIF1, pdb_paths: Sequence[str]) -> torch.Tensor:
    embeds = []
    all_coords = []
    for pdb_path in pdb_paths:
        coords = if_model.load_coords(pdb_path, chain="A")[0]
        batch_converter = if_model.get_batch_converter()
        batch = [(coords, None, None)]
        coords, confidence, _, _, padding_mask = batch_converter(batch, device=if_model.device)
        encoder_out = if_model.model.encoder(
            coords, padding_mask, confidence, return_all_hiddens=False
        )["encoder_out"][0].transpose(
            0, 1
        )  # (L, B) -> (B, L)
        embeds.append(encoder_out)  # .half())
        all_coords.append(coords)  # .half())
    embeds = torch.cat(embeds, dim=1)  # concatenate along the sequence dimension
    coords = torch.cat(all_coords, dim=0)  # concatenate along the batch dimension
    return embeds, coords


def _get_logps_tiered_fast(
    memory: Optional[list[PackedTensorSequences]],
    variants: Sequence[np.ndarray],
    model: PoETS,
    batch_size: int,
    alphabet: Uniprot21,
    struct_embed: torch.Tensor,
    coords: torch.Tensor,
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
        logits = model.logits(
            this_variants[:, :-1], s=struct_embed, coords=coords, memory=memory, preallocated_memory=True
        )

        targets = this_variants[:, 1:]  # remove eos token
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        logps.append(score.cpu().numpy())  # at the end, perform average between the two scores
    return np.hstack(logps)


def get_logps_tiered_fast(
    msa_sequences: Sequence[np.ndarray],
    variants: Sequence[np.ndarray],
    model: PoETS,
    batch_size: int,
    alphabet: Uniprot21,
    struct_embed: torch.Tensor,
    coords: torch.Tensor,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).cuda()
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).cuda()
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            struct_embed,
            coords,
            segment_sizes.unsqueeze(0),
            pbar_position=pbar_position,
        )
    else:
        memory = None

    return _get_logps_tiered_fast(
        memory=memory,
        variants=variants,
        model=model,
        struct_embed=struct_embed,
        coords=coords,
        batch_size=batch_size,
        alphabet=alphabet,
        pbar_position=pbar_position,
    )


@torch.inference_mode()
def score_assay(
    model: PoETS,
    alphabet: Uniprot21,
    wt_sequence: str,
    variants: Sequence[str],
    struct_embed: torch.Tensor,
    coords: torch.Tensor,
    msa_path: str,
    args: argparse.Namespace,
) -> np.ndarray:

    wt = scoring_utils.append_startstop(
        alphabet.encode(wt_sequence.encode()), alphabet=alphabet, stop=True
    )
    if args.relative_to_wt:
        variants.append(wt)
    variants = [
        scoring_utils.append_startstop(alphabet.encode(v.encode()), alphabet=alphabet, stop=True)
        for v in variants
    ]

    # process msa
    msa_sequences = scoring_utils.get_seqs_from_fastalike(msa_path, upper=True)
    msa = scoring_utils.get_encoded_msa_from_a3m_seqs(
        msa_sequences=msa_sequences, alphabet=alphabet
    )

    # score the variants
    logps = []
    if not args.debug:
        params = list(
            itertools.product(
                [6144, 12288, 24576],
                [1.0, 0.95, 0.90, 0.70, 0.50],
            )
        )
    else:
        params = [(12288, 0.95)]
    for max_tokens, max_similarity in tqdm(params, desc="ensemble"):
        sampler = MSASampler(
            method=NeighborsSampler(
                can_use_torch=False,
            ),
            max_similarity=max_similarity,
        )
        sample_idxs = sampler.get_sample_idxs(
            msa=msa,
            gap_token=alphabet.gap_token,
            seed=_SEED,
        )
        # create the sequence-of-sequences
        this_msa_sequences = scoring_utils.sample_msa_sequences(
            get_sequence_fn=lambda ii: msa_sequences[ii].upper().translate(None, delete=b"-"),
            sample_idxs=sample_idxs,
            max_tokens=max_tokens,
            alphabet=alphabet,
            shuffle_seed=_SEED,
            truncate=False,
        )
        forward_logps = get_logps_tiered_fast(
            msa_sequences=this_msa_sequences,
            variants=variants,
            model=model,
            struct_embed=struct_embed,
            coords=coords,
            batch_size=args.batch_size,
            alphabet=alphabet,
            pbar_position=PBAR_POSITION,
        )
        this_logps = forward_logps  # only consider forward logps
        logps.append(this_logps)
    logps = np.vstack(logps).mean(axis=0)
    if args.relative_to_wt:
        logps = logps[:-1] - logps[-1]
    return logps


def main(args):
    model_spec = {
        "n_vocab": 24,
        "hidden_dim": 1024,
        "num_layers": 12,
        "nhead": 16,
        "use_multi_rotary": True,
        "norm": True,
        "encoder_dim": 512,
        "dropout": 0.0,
        "projector": "geom_att",
        "add_gaussian_noise": False,
    }
    ckpt = torch.load(Path(args.ckpt_path) / "pytorch_model.bin")
    model = PoETS(**model_spec)
    model.load_state_dict(ckpt)
    del ckpt
    model = model.cuda().eval()  # .half().eval()
    alphabet = Uniprot21(include_gap=True, include_startstop=True, distinct_startstop=True)
    # scoring_utils.jit_warmup(model, alphabet)

    # get variants to score
    ref_series = pd.read_csv(os.getenv("REF_FILE_PATH"))
    base_variants_path = Path(os.getenv("DMS_FILES_PATH"))
    base_msa_path = Path(os.getenv("MSA_FILES_PATH"))
    base_pdb_path = Path(os.getenv("PDB_FILES_PATH"))

    # create the output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load esm-if1 model and get sanitized logits for given structure
    if_model = EsmIF1(device="cuda")

    row = ref_series.iloc[args.index]
    assay_id = row["DMS_id"]
    wt_sequence = row["target_seq"]
    variants_filename = row["DMS_filename"]
    variants_df = pd.read_csv(base_variants_path / variants_filename)
    msa_path = base_msa_path / row["MSA_filename"]
    print(f"Processing {assay_id}")

    if "mutated_sequence" in variants_df.columns:
        variants = variants_df["mutated_sequence"].values
    elif "mutant" in variants_df.columns:
        variants = variants_df["mutant"].values
    else:
        raise ValueError("Sequence column not present in DMS file")

    # In the script for PoET model, the input sequences to score are cut according to
    # the MSA start and end positions. If we consider ESM-IF1, however, we have to cut
    # them according to the PDB start and end positions.
    pdb_range = row["pdb_range"].split("-")
    pdb_start, pdb_end = int(pdb_range[0]), int(pdb_range[-1])
    pdb_files = [base_pdb_path / file for file in row["pdb_file"].split("|")]
    struct_embed, coords = embed_struct(if_model, pdb_files)
    wt_sequence = wt_sequence[pdb_start - 1 : pdb_end]
    # cut the variant sequences as well
    variants = [v[pdb_start - 1 : pdb_end] for v in variants]

    scores = score_assay(
        model=model,
        alphabet=alphabet,
        msa_path=msa_path,
        wt_sequence=wt_sequence,
        variants=variants,
        struct_embed=struct_embed,
        coords=coords,
        args=args,
    )

    df = variants_df.copy()
    df["poets_score"] = scores
    df.to_csv(args.output_dir / variants_filename, index=False)


if __name__ == "__main__":
    # Load env vars
    dotenv.load_dotenv(override=True)
    if os.getenv("PROTEINGYM_PATH") is None:
        raise ValueError("PROTEINGYM_PATH environment variable is not set.")
    args = parse_args()
    main(args)
