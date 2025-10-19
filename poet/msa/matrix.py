"""
Overridden version of biotite.sequencealign.matrix module. It is limited
to the substitution matrices used in TJET and allows float values.
"""

from enum import Enum
from pathlib import Path

import numpy as np
from biotite.sequence import Alphabet, ProteinSequence
from biotite.sequence.align import SubstitutionMatrix

# locally stored substitution matrices used in TJET
_DB_DIR = Path(__file__).parent / "matrix_data"


class SubstitutionMatrix(SubstitutionMatrix):
    def __init__(self, alphabet1, alphabet2, score_matrix):
        self._alph1 = alphabet1
        self._alph2 = alphabet2
        if isinstance(score_matrix, dict):
            self._fill_with_matrix_dict(score_matrix)
        elif isinstance(score_matrix, np.ndarray):
            alph_shape = (len(alphabet1), len(alphabet2))
            if score_matrix.shape != alph_shape:
                raise ValueError(
                    f"Matrix has shape {score_matrix.shape}, "
                    f"but {alph_shape} is required"
                )
            if not np.issubdtype(score_matrix.dtype, np.integer):
                raise TypeError("Score matrix must be an integer ndarray")
            self._matrix = score_matrix.astype(np.int32)
            # If the score matrix was converted from a a float matrix,
            # inf values would be converted to 2**31,
            # which is probably undesired and gives overflow issues in the alignment
            # functions
            if (
                np.any(self._matrix == np.iinfo(np.int32).max) or
                np.any(self._matrix == np.iinfo(np.int32).min)
            ):  # fmt: skip
                raise ValueError(
                    "Score values are too large. "
                    "Maybe it was converted from a float matrix containing inf values?"
                )
        elif isinstance(score_matrix, str):
            matrix_dict = SubstitutionMatrix.dict_from_db(score_matrix)
            self._fill_with_matrix_dict(matrix_dict)
        else:
            raise TypeError(
                "Matrix must be either a dictionary, " "an 2-D ndarray or a string"
            )
        # This class is immutable and has a getter function for the
        # score matrix -> make the score matrix read-only
        self._matrix.setflags(write=False)

    @staticmethod
    def dict_from_str(string):
        """
        Create a matrix dictionary from a string in NCBI matrix format.

        Symbols of the first alphabet are taken from the left column,
        symbols of the second alphabet are taken from the top row.

        The keys of the dictionary consist of tuples containing the
        aligned symbols and the values are the corresponding scores.

        Returns
        -------
        matrix_dict : dict
            A dictionary representing the substitution matrix.
        """
        lines = [line.strip() for line in string.split("\n")]
        lines = [line for line in lines if len(line) != 0 and line[0] != "#"]
        symbols1 = [line.split()[0] for line in lines[1:]]
        symbols2 = [e for e in lines[0].split()]
        # Start of modified code
        scores = np.array([line.split()[1:] for line in lines[1:]])
        try:
            scores = scores.astype(int)
        except ValueError:
            scores = scores.astype(float)
        # End of modified code
        scores = np.transpose(scores)

        matrix_dict = {}
        for i in range(len(symbols1)):
            for j in range(len(symbols2)):
                matrix_dict[(symbols1[i], symbols2[j])] = scores[i, j]
        return matrix_dict

    @staticmethod
    def dict_from_db(matrix_name):
        """
        Create a matrix dictionary from a valid matrix name in the
        internal matrix database.

        The keys of the dictionary consist of tuples containing the
        aligned symbols and the values are the corresponding scores.

        Returns
        -------
        matrix_dict : dict
            A dictionary representing the substitution matrix.
        """
        filename = _DB_DIR / f"{matrix_name}.mat"
        with open(filename, "r") as f:
            return SubstitutionMatrix.dict_from_str(f.read())

    @staticmethod
    def list_db():
        """
        List all matrix names in the internal database.

        Returns
        -------
        db_list : list
            List of matrix names in the internal database.
        """
        return [path.stem for path in _DB_DIR.glob("*.mat")]

    def std_nucleotide_matrix():
        raise NotImplementedError(
            "This method is not available in this overridden class"
        )

    def std_3di_matrix():
        raise NotImplementedError(
            "This method is not available in this overridden class"
        )

    def std_protein_blocks_matrix():
        raise NotImplementedError(
            "This method is not available in this overridden class"
        )


class SubsMat(Enum):
    BLOSUM62 = (ProteinSequence.alphabet, -0.5209)
    GONNET = (Alphabet(list("CSTPAGNDEQHRKMILVFYWX*")), -0.6152)
    HSDM = (ProteinSequence.alphabet, -0.3665)

    def __init__(self, alphabet, expected):
        self.alphabet = alphabet
        self.expected = expected
        self.matrix = SubstitutionMatrix(alphabet, alphabet, self.name)
