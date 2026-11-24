"""Single source of truth for paths, constants and environment variables.

Every other module imports from here instead of building its own paths.
Importing this module also loads the .env file (API keys).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# project root = two levels above this file (src/finguard/config.py)
ROOT = Path(os.environ.get("FINGUARD_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# output of src/finguard/ml/train.py, read by the agent tools
SCORED_STORE = PROCESSED_DIR / "transactions_scored.parquet"

# ASSUMPTION used to tune the decision threshold: dollars of analyst time per flagged transaction
REVIEW_COST = 10

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "finguard_agent" / ".env")
