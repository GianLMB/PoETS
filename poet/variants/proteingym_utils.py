"""Utils to load ProteinGym data for scoring"""

import os
from pathlib import Path
from typing import Callable, Optional

import dotenv
import pandas as pd

dotenv.load_dotenv()


def get_ref_df() -> pd.DataFrame:
    """Load reference data"""
    ref_path = Path(os.getenv("PROTEINGYM_REF_PATH"))
    ref_df = pd.read_csv(ref_path)
    return ref_df


def score_dms(index: int, ref_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Score DMS data"""
    if ref_df is None:
        ref_df = get_ref_df()
    assay = ref_df.iloc[index]
    dms_path = Path(os.getenv("DMS_FILES_PATH")) / assay["dms_file"]
    dms_data = pd.read_csv(dms_path)
    return
