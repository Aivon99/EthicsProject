from __future__ import annotations

import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import sys 

from src.utils import get_logger, load_config

logger = get_logger(__name__)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        