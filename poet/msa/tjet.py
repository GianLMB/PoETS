"""TJET implementation for trace computation."""

import multiprocessing
import string
import warnings
from itertools import cycle
from typing import Optional, Sequence

import networkx as nx
import numpy as np
from biotite.application.clustalo import ClustalOmegaApp
from biotite.application.muscle import Muscle5App
from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_multiple, get_codes
from biotite.sequence.io.fasta import FastaFile
from biotite.sequence.phylo import neighbor_joining
from numba import jit

from .matrix import SubsMat


class TJET(object):
    """
    TJET interface wrapper for trace computation.
    """

    def __init__(self, query_sequence: str, samples: list[Sequence[str]]):
        """
        Initialize the TJET object with the query sequence and the samples to be used for trace
        computation.

        Parameters
        ----------
        query_sequence : str
            The query (wild type) sequence.
        samples : List[Sequence[str]]
            The list of different samples to be used for trace computation and pooling.
        method : str, optional
            The method to be used for multiple sequence alignment. Should be one of `default`,
            `clustalo`, or `muscle`.
            - `default`: Use biotite's `align_multiple` function.
            - `clustalo`: Use ClustalOmega for alignment.
            - `muscle`: Use MUSCLE v5 for alignment.
            If `clustalo` or `muscle` is selected, the binary for the corresponding tool should be
            available in the PATH.
        """
        self.query_sequence = query_sequence
        self._samples = samples

    def get_samples(self) -> np.ndarray:
        """
        Get the samples used for trace computation.
        """
        return self._samples

    def get_samples_as_strings(self) -> list[Sequence[str]]:
        """
        Get the samples used for trace computation as strings.
        """
        return [
            ["".join(seq).replace("-", "") for seq in sample]
            for sample in self.get_samples()
        ]

    def compute_sample_trace(
        self, sample_idx: int, method: str = "default"
    ) -> np.ndarray:
        """
        Compute the tree level trace for the given sample.

        Parameters
        ----------
        sample_idx : int
            The index of the sample to be used for trace computation.
        method : str, optional
            The method to be used for multiple sequence alignment. Should be one of `default`,
            `clustalo`, or `muscle`.
            - `default`: Use biotite's `align_multiple` function.
            - `clustalo`: Use ClustalOmega for alignment.
            - `muscle`: Use MUSCLE v5 for alignment.
            If `clustalo` or `muscle` is selected, the binary for the corresponding tool should be
            available in the PATH.

        Returns
        -------
        trace : np.ndarray
            The computed trace for the given sample.
        """
        sample = self.get_samples_as_strings()[sample_idx]
        return get_sample_tjet_trace(
            sample, self.query_sequence, method, sample_idx=sample_idx
        )

    def compute_trace(
        self, method: str = "default", num_workers: int = 8
    ) -> np.ndarray:
        """
        Compute the tree level trace for the given samples.

        Parameters
        ----------
        num_workers : int, optional
            The number of workers to be used for parallel computation. Default is 8.

        Returns
        -------
        trace : np.ndarray
            The computed trace for the given samples.
        """
        samples = self.get_samples_as_strings()
        self.trace_ = get_tjet_trace(samples, self.query_sequence, method, num_workers)
        return self.trace_

    @classmethod
    def from_alignment(
        cls,
        filepath: str,
        max_length: int = 100000,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize the TJET object from a FASTA-like file.

        Parameters
        ----------
        filepath : str
            The path to the FASTA-like file containing the sequences.
        max_length : int, optional
            The maximum number of sequences to be read from the file. Default is 40000.
        seed : int, optional
            The seed to be used for random sampling. Default is 42.

        Returns
        -------
        tjet : TJET
            The initialized TJET object.
        """
        msa = read_fastalike(filepath, max_length)
        query_sequence, samples = tjet_sample_msa(msa, num_samples, seed)
        return cls(query_sequence, samples)


def read_fastalike(filepath: str, max_length=40000) -> np.ndarray:
    """
    Read sequences from a FASTA-like file and return the query sequence and the samples.
    """
    msa = [entry[1].replace(".", "-") for entry in FastaFile.read_iter(filepath)]
    # check if all sequences have the same length
    if all(len(seq) == len(msa[0]) for seq in msa):
        msa = np.vstack([list(seq) for seq in msa])
    else:
        try:
            table = str.maketrans("", "", string.ascii_lowercase)
            msa = np.vstack([list(seq.translate(str.maketrans(table))) for seq in msa])
        except ValueError:
            raise ValueError(
                "The sequences in the alignment file have different lengths"
            )
    msa = msa[:max_length]
    print("Alignment shape:", msa.shape)
    return msa


def _get_max_samples_by_thresholds(
    num_sequences_by_bin: np.ndarray, thresholds: list[float]
) -> int:
    """
    Get the maximum number of sequences to be sampled from each bin such that the condition
    C(N_i, x) >= threshold is satisfied. If the condition is not satisfied for a given threshold,
    the number of samples for that bin is set to nan.
    """
    max_samples = np.zeros((len(num_sequences_by_bin), len(thresholds)), dtype=int)
    for i, bin_sequences in enumerate(num_sequences_by_bin):
        if bin_sequences == 0:
            continue
        stop = bin_sequences // 2 + 1
        x = bin_sequences - 1
        comb = bin_sequences
        for j, threshold in enumerate(thresholds):
            while comb <= threshold:
                x -= 1
                comb *= (x + 1) / (bin_sequences - x)
                if x <= stop:  # < to handle the case when initial x <= 3
                    x += 1  # avoid negative values, reinitialize x before while loop breaks
                    break
            max_samples[i, j] = x
    return max_samples.T


def _get_num_samples_by_bin(
    num_sequences_by_bin: np.ndarray[int], len_sample: int
) -> np.ndarray[int]:
    """
    Compute the number of samples to be taken from each bin based on the number of sequences in
    each bin, based on the procedure desscribed in TJET.
    """
    nonzero_mask = num_sequences_by_bin != 0

    combinations_thresholds = np.linspace(1, 2, 5) * len_sample
    max_samples = _get_max_samples_by_thresholds(
        num_sequences_by_bin, combinations_thresholds
    )

    # ATTENTION: BUG found in execution for SPG1_STRSG_Wu_2016 assay in proteingym (index 182)
    # accept_condition found with dimension (5, 3) instead of (5, 4)
    accept_condition = (
        max_samples[:, nonzero_mask]
        != num_sequences_by_bin[None, nonzero_mask] // 2 + 1
    )
    valid_max_samples = max_samples[accept_condition.all(axis=1)]
    if valid_max_samples.size == 0:
        warnings.warn(
            "The provided sequences don't show enough diversity for TJET sampling in at least one"
            " sequence identity interval. Sequences from that interval are exluded from the"
            " sampling.",
            category=UserWarning,
        )
        max_samples = np.where(accept_condition, max_samples, 0)[0]
    else:
        max_samples = valid_max_samples[-1]

    print("Max samples:", max_samples, len_sample)

    if sum(max_samples) < len_sample:
        raise ValueError(
            "The number of provided sequences or the sequence identity with the query sequence is"
            " too low for TJET sampling"
        )

    samples_by_bin = np.minimum(max_samples, len_sample // 4)
    remaining_samples = len_sample - samples_by_bin.sum()
    cycle_indices = np.where(nonzero_mask)[0]  # [::-1]
    cycler = cycle(cycle_indices)

    while remaining_samples > 0:
        i = next(cycler)
        if samples_by_bin[i] < max_samples[i]:
            samples_by_bin[i] += 1
            remaining_samples -= 1

    return samples_by_bin


def tjet_sample_msa(
    msa: np.ndarray,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> tuple[str, list[Sequence[str]]]:
    """
    Sample sequences from the given multiple sequence alignment as in TJET. For a MSA of N
    sequences, we extract sqrt(N) sequences randomly for sqrt(N) times.
    If the MSA contains less then 100 sequences, we extract 10 samples of 10 sequences each.
    The first sequence is considered as the query sequence and is kept separate, to be then
    included in all the samples.
    The MSA is first divided in four bins based on the sequence identity with the query sequence,
    and the sampling is performed randomly from each bin.
    """
    if isinstance(msa[0, 0], bytes):
        msa = np.vectorize(lambda x: x.decode())(msa)  # convert bytes to str

    query_seq, msa = msa[0], msa[1:]
    seq_id = (msa == query_seq).sum(axis=1) / msa.shape[1]
    identity_mask = (seq_id > 0.2) & (seq_id <= 0.98)
    seq_id = seq_id[identity_mask]
    msa = msa[identity_mask]

    if num_samples is None:
        num_samples = len_sample = int(max(msa.shape[0] ** 0.5, 10))
    else:
        len_sample = num_samples

    # Assign sequences to bins based on sequence identity with the query sequence
    bins = np.digitize(seq_id, [0.4, 0.6, 0.8])
    num_sequences_by_bin = np.bincount(bins, minlength=4)[:4]
    samples_by_bin = _get_num_samples_by_bin(num_sequences_by_bin, len_sample)
    print("Number of sequences in each bin:", num_sequences_by_bin)
    print("Number of samples in each bin:", samples_by_bin)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(num_samples):
        sample = []
        for bin_idx in range(4):
            if samples_by_bin[bin_idx] == 0:
                continue
            bin_samples = rng.choice(
                np.where(bins == bin_idx)[0], samples_by_bin[bin_idx], replace=False
            )
            sample.append(msa[bin_samples])
        sample = np.concatenate(sample)
        samples.append(sample)

    return query_seq, samples


# def _decode_alignment(alignment: np.ndarray, alphabet: Alphabet) -> np.ndarray:

#     def decode(alphabet: Alphabet, code: int) -> str:
#         if code == -1:  # gap
#             return "-"
#         return alphabet.decode(code)

#     alignment = np.vectorize(decode)(alphabet, alignment)
#     return alignment


@jit(forceobj=True)
def _compute_distance_pairwise(
    seq1: np.ndarray, seq2: np.ndarray, matrix: np.ndarray, gap_code: int
) -> tuple[float, int]:
    """
    Compute the distance between two encoded sequences using the substitution matrix.
    """
    mask = (seq1 != gap_code) & (seq2 != gap_code)
    score = np.sum(matrix[seq1[mask], seq2[mask]])
    length = np.sum(mask)
    return score, length


@jit(forceobj=True)
def _compute_score(
    alignment: np.ndarray, matrix: np.ndarray, gap_code: int
) -> tuple[np.ndarray]:
    num_sequences, _ = alignment.shape
    score = lengths = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(i, num_sequences):
            score[i, j], lengths[i, j] = _compute_distance_pairwise(
                alignment[i], alignment[j], matrix, gap_code
            )
            if i != j:
                score[j, i] = score[i, j]
                lengths[j, i] = lengths[i, j]
    return score, lengths


def _compute_scoredist_distance_matrix(
    alignment: np.ndarray, subs_mat: SubsMat
) -> np.ndarray:
    """
    Compute distance matrix for multiple sequence alignment using ScoreDist algorithm
    by Sonnhammer and Hollich (DOI: 10.1186/1471-2105-6-108).
    """

    # convert custom classes to numpy array and compute distance matrix with numba jit decorator
    # the computation (of complexity O(n^3)) is much faster with numba jit
    # alignment is decoded at the end
    matrix = subs_mat.matrix.score_matrix()
    score, lengths = _compute_score(alignment, matrix, gap_code=-1)
    min_score = subs_mat.expected * lengths  # len_sequences

    # for very distant sequences, score - min_score might be negative resulting in an
    # error in the computation of the distance matrix. So, we check that before computing the
    # distance matrix, and remove the sequences with negative scores
    # find indices in score matrix that have negative values
    negative_scores = np.argwhere(score <= min_score)
    if negative_scores.size > 0:
        indices_to_remove = np.unique(negative_scores)
        indices_to_remove = indices_to_remove[
            indices_to_remove != 0
        ]  # do not remove the query sequence
        alignment = np.delete(alignment, indices_to_remove, axis=0)
        warnings.warn(
            f"Removed sequences with negative scores. New alignment length: {alignment.shape[0]}",
            category=UserWarning,
        )
        if alignment.shape[0] < 10:
            return None, None

        # remove entries from score and lengths matrices
        score = np.delete(score, indices_to_remove, axis=0)
        score = np.delete(score, indices_to_remove, axis=1)
        min_score = np.delete(min_score, indices_to_remove, axis=0)
        min_score = np.delete(min_score, indices_to_remove, axis=1)

    diag = np.diag(score)
    max_score = (diag[:, None] + diag[None, :]) * 0.5

    distance_matrix = (
        -np.log((score - min_score) / (max_score - min_score)) * 100
    ) + 0.0  # bias set to 0 in TJET

    # alignment = _decode_alignment(alignment, subs_mat.matrix.get_alphabet1())
    return distance_matrix, alignment


def _midpoint_rooting(tree: nx.DiGraph) -> nx.DiGraph:
    undirected_tree = tree.to_undirected()
    # set negative distances to 0
    for u, v in undirected_tree.edges:
        if undirected_tree[u][v]["distance"] < 0:
            undirected_tree[u][v]["distance"] = 0

    # save data for direct and reverse edges, to be loaded in the rooted tree
    edge_data = {(u, v): d for u, v, d in undirected_tree.edges.data()}
    edge_data.update({(v, u): d for u, v, d in undirected_tree.edges.data()})

    # find the two furthest nodes in the tree and their distance
    distances = dict(
        nx.all_pairs_dijkstra_path_length(undirected_tree, weight="distance")
    )
    distances = {
        (u, v): dist
        for u, dists in distances.items()
        for v, dist in dists.items()
        if isinstance(u, int) and isinstance(v, int)  # exclude non-leaf nodes
    }
    (u, v), max_dist = max(distances.items(), key=lambda x: x[1])

    # Find the node in the path from u to v that is closer to dist/2 from u
    path = nx.shortest_path(undirected_tree, source=u, target=v)
    midpoint_distance = max_dist / 2
    current_distance = 0
    midpoint_node = path[0]
    # Traverse the path to find the midpoint
    for i in range(1, len(path)):
        next_node = path[i]
        edge_distance = undirected_tree[midpoint_node][next_node]["distance"]
        current_distance += edge_distance

        if current_distance >= midpoint_distance:
            # Determine which node is closer to the midpoint
            if abs(current_distance - midpoint_distance) < abs(
                (current_distance - edge_distance) - midpoint_distance
            ):
                midpoint_node = next_node
            break
        midpoint_node = next_node

    # Set root node to the midpoint and return the rooted tree
    # Being already a tree topology, BFS and DFS are equivalent
    root = midpoint_node
    tree = nx.bfs_tree(undirected_tree, root)
    nx.set_edge_attributes(tree, edge_data)  # only data for existing edges is updated
    return tree, root


def _assign_consensus_sequences(tree: nx.DiGraph, alignment: np.ndarray):
    """
    Assign consensus sequences to the nodes in the tree. The consensus sequence is the sequence
    that contains conserved residues in chidren nodes, and gaps in the rest of the positions.
    """
    # Iterate over the nodes in topological order, starting from leaves
    for node in reversed(list(nx.topological_sort(tree))):
        children = list(tree.successors(node))
        # For leaf nodes, the consensus is the sequence itself taken from the alignment
        if not children:
            tree.nodes[node]["consensus"] = alignment[node]
            continue
        # For internal nodes, is the intersection of the consensus sequences of the children
        # Unmatched positions are filled with -1 (gap)
        children_consensus = np.array(
            [tree.nodes[child]["consensus"] for child in children]
        )
        consensus_positions = np.all(
            children_consensus == children_consensus[0], axis=0
        )
        tree.nodes[node]["consensus"] = np.where(
            consensus_positions, children_consensus[0], -1
        )
    return tree


def _assign_backtrace_sequences(tree: nx.DiGraph):
    """
    Assign backtrace sequences to the nodes in the tree. The backtrace sequence
    is the sequence which records all residues in the consensus sequence associated
    to x that do not already belong to the back-trace of the father of x.
    """
    # Iterate over the nodes in topological order, starting from the root
    for node in nx.topological_sort(tree):
        parent = next(tree.predecessors(node), None)
        if parent is None:  # root node
            tree.nodes[node]["backtrace"] = tree.nodes[node]["consensus"]
            continue
        parent_consensus = tree.nodes[parent]["consensus"]
        tree.nodes[node]["backtrace"] = np.where(
            tree.nodes[node]["consensus"] == parent_consensus,
            -1,
            tree.nodes[node]["consensus"],
        )
    return tree


def _rank_nodes(tree: nx.DiGraph, root: tuple) -> nx.DiGraph:
    """
    Compute the distances from each node to the root and rank them starting from the closest.
    Leaves are not ranked.
    """
    # get the root node and compute the distances from the root
    distances = nx.shortest_path_length(tree, source=root, weight="distance")
    # Rank nodes by distance
    ordered_nodes = sorted(distances.items(), key=lambda x: x[1])
    rank = 0
    prev_distance = -1
    for node, distance in ordered_nodes:
        if tree.out_degree(node) == 0:
            continue
        if distance > prev_distance:
            rank += 1
            prev_distance = distance
        tree.nodes[node]["rank"] = rank  # rank starts from 1 (root)
    return tree


def _compute_tree_level_ranking_trace(tree: nx.DiGraph, root: tuple) -> np.ndarray:
    # rank internal nodes based on their distances from the root
    tree = _rank_nodes(tree, root)

    # apply tjet trace computation, strating from the root. gaps ("-") are encoded as -1
    # root = next(nx.topological_sort(tree))
    # path_to_query = nx.shortest_path(tree, source=root, target=0)
    # tree_level_trace = np.where(tree.nodes[root]["backtrace"] == -1, np.nan, 1)

    # for node in path_to_query[1:-1]:
    #     current_backtrace = tree.nodes[node]["backtrace"]
    #     current_rank = tree.nodes[node]["rank"]
    #     # find other subtrees (nodes) that have the rank equal or higher than the current node,
    #     # for which the parent node has a rank equal or lower than the current node,
    #     # and that have at least two children (i.e. are not leaves)
    #     # and extract the corresponding backtraces
    #     subtree_backtraces = np.vstack(
    #         [current_backtrace]
    #         + [
    #             tree.nodes[n]["backtrace"]
    #             for n in tree.nodes
    #             if "rank" in tree.nodes[n]  # exclude leaves
    #             and tree.nodes[n]["rank"] >= current_rank
    #             and tree.nodes[next(tree.predecessors(n))]["rank"] <= current_rank
    #         ]
    #     )
    #     traces = np.sum(subtree_backtraces != -1, axis=0) >= 2
    #     # assign rank to the corresponding positions in tree_level_trace, that were nan
    #     tree_level_trace = np.where(
    #         np.isnan(tree_level_trace) & traces, current_rank, tree_level_trace
    #    )

    # Correct implementation: iterate over nodes' ranks, not over the path to the query sequence
    tree_level_trace = np.where(tree.nodes[root]["backtrace"] == -1, np.nan, 1)
    max_rank = max(tree.nodes(data="rank", default=-1), key=lambda x: x[1])[1]

    for rank in range(2, max_rank + 1):
        subtree_backtraces = np.vstack(
            [
                tree.nodes[n]["backtrace"]
                for n in tree.nodes
                if tree.nodes[n].get("rank", -1) >= rank
                for _ in (
                    [1]
                    if tree.nodes[next(tree.predecessors(n))].get("rank", -1) < rank
                    else [1, 2]
                    if tree.nodes[next(tree.predecessors(n))].get("rank", -1) == rank
                    else []
                )
                # and tree.nodes[next(tree.predecessors(n))].get("rank", -1) <= rank
            ]
        )
        traces = np.sum(subtree_backtraces != -1, axis=0) >= 2
        # assign rank to the corresponding positions in tree_level_trace, that were nan
        tree_level_trace = np.where(
            np.isnan(tree_level_trace) & traces, rank, tree_level_trace
        )

    # normalization
    ordered_ranks = np.sort(tree_level_trace)
    max_rank = ordered_ranks[int(0.95 * len(ordered_ranks))]
    if np.isnan(max_rank):
        max_rank = np.nanmax(tree_level_trace)

    tree_level_trace = np.where(
        tree_level_trace <= max_rank, 1 - tree_level_trace / max_rank, np.nan
    )
    return tree_level_trace


def compute_tree_level_trace(tree: nx.DiGraph, alignment: np.ndarray):
    """
    Compute tree level trace as done in TJET. For each position in the query sequence, find the
    distance from the query sequence to the furthest node in the tree where the corresponding
    amino acid appears and is conserved throughout the path.
    """
    tree, root = _midpoint_rooting(tree)
    tree = _assign_consensus_sequences(tree, alignment)  # leaves -> root
    tree = _assign_backtrace_sequences(tree)  # root -> leaves
    tree_level_trace = _compute_tree_level_ranking_trace(tree, root)
    return tree_level_trace


def _align_sample(
    sequences: Sequence[ProteinSequence],
    subs_mat: SubsMat,
    method: str = "default",
) -> np.ndarray:
    """
    Align the given sequences and remove gaps from the query sequence.
    """
    # Align the sequences
    if method == "default":
        alignment = align_multiple(sequences, matrix=subs_mat.matrix)[0]
    else:
        if method == "clustalo":
            alignment = ClustalOmegaApp.align(sequences)
        elif method == "muscle":
            alignment = Muscle5App.align(sequences)

    # Convert to numpy array and remove gaps from the query sequence
    alignment = get_codes(alignment)
    query_mask = alignment[0] != -1  # gap code
    alignment = alignment[:, query_mask]
    return alignment


def align_sample(
    sample: Sequence[str],
    query_sequence: str,
    method: str = "default",
) -> np.ndarray:
    # Add the query sequence to the list of sequences
    sequences = [ProteinSequence(query_sequence)] + [
        ProteinSequence(seq) for seq in sample
    ]

    # Compute distance matrix using Score Dist algorithm and generate the tree
    # Start with BLOSUM62 matrix for computation, and use the next available matrix if the
    # distance matrix is not positive definite, even after removing sequences with negative scores
    # until the number of sequences is less than 10
    # If no positive definite matrix is found, return array of nan as trace
    subs_mat_iter = iter(SubsMat)
    while True:
        try:
            subs_mat = next(subs_mat_iter)
        except StopIteration:
            return np.full(len(query_sequence), np.nan)
        # compute distance matrix with the current substitution matrix and original alignment
        try:
            alignment = _align_sample(sequences, subs_mat, method)
        except ValueError:
            if method == "default":
                warnings.warn(
                    f"Failed to compute alignment with {subs_mat.name}. Trying next matrix...",
                    category=UserWarning,
                )
            else:
                raise ValueError(
                    f"Failed to compute alignment with {method} method. Please try with 'default'"
                    " method, that automatically tries different substitution matrices if one of"
                    " them fails."
                )
            continue
        distance_matrix, updated_alignment = _compute_scoredist_distance_matrix(
            alignment, subs_mat
        )
        if distance_matrix is not None:
            break
    return distance_matrix, updated_alignment


def get_sample_tjet_trace(
    sample: Sequence[str],
    query_sequence: str,
    method: str = "default",
    sample_idx: int = 0,
) -> np.ndarray:
    """
    Generate a phylogenetic tree from the given sequences and compute tjet trace.

    Parameters
    ----------
    sample : Sequence[str]
        The sequences sampled from the MSA. They should be a list or tuple of ungapped sequences.
    query_sequence : str
        The query (wild type) sequence.
    method : str, optional

    Returns
    -------
    tree_level_trace : np.ndarray
        The computed tree level trace as in TJET.
    """
    # Align the sample
    distance_matrix, updated_alignment = align_sample(sample, query_sequence, method)

    # Generate the tree and compute the tree level trace
    tree = neighbor_joining(distance_matrix).as_graph()
    tree_level_trace = compute_tree_level_trace(tree, updated_alignment)
    print(f"Trace computed for sample {sample_idx}")
    return tree_level_trace


def get_tjet_trace(
    samples: list[Sequence[str]],
    query_sequence: str,
    method: str = "default",
    num_workers: int = 8,
) -> np.ndarray:
    """
    Compute trees from different samples in parallel.

    Parameters
    ----------
    query_sequence : str
        The query (wild type) sequence.
    samples : List[Sequence[str]]
        The list of different samples to be used for trace computation and pooling.
    method : str, optional
        The method to be used for multiple sequence alignment. Should be one of `default`,
        `clustalo`, or `muscle`.
        - `default`: Use biotite's `align_multiple` function.
        - `clustalo`: Use ClustalOmega for alignment.
        - `muscle`: Use MUSCLE v5 for alignment.
        If `clustalo` or `muscle` is selected, the binary for the corresponding tool should be
        available in the PATH.
    num_workers : int, optional
        The number of workers to be used for parallel computation. Default is 8.

    Returns
    -------
    trace : np.ndarray
        The computed trace for the given samples.
    """
    allowed_methods = ["default", "clustalo", "muscle"]
    if method not in allowed_methods:
        raise ValueError(f"Invalid method. Should be one of {allowed_methods}")

    # Compute the tree level trace for each sample in parallel
    tasks = [
        (sample, query_sequence, method, sample_idx)
        for sample_idx, sample in enumerate(samples)
    ]
    with multiprocessing.Pool(num_workers) as pool:
        trace = pool.starmap(get_sample_tjet_trace, tasks)

    # Pool the traces by taking the mean of non-nan values. Assign 0 to resulting nan values
    trace = np.array(trace)
    frequency = np.sum(~np.isnan(trace), axis=0) / trace.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=RuntimeWarning
        )  # ignore mean of empty slice
        trace = np.nanmean(trace, axis=0)
    # print("Frequency:", frequency)
    # print("Initial trace:", trace)
    trace = np.nan_to_num(trace, nan=0.0) * frequency
    return trace
