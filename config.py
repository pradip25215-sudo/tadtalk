"""Project settings shared by every file: folder paths and the LLM model name."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent   # the tadtalk/ folder
DATA = ROOT / "data"                     # clean tables that the tools read
RAW = DATA / "raw"                       # original downloads and copies
PLOTS = ROOT / "plots"                   # PNG plots made by the tools
PLOTS.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")               # loads ANTHROPIC_API_KEY and MODEL from .env
MODEL = os.getenv("MODEL", "anthropic:claude-haiku-4-5-20251001")
