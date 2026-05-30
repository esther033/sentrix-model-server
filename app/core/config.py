from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR            = BASE_DIR / "models"
FEATURE_SCHEMA_PATH  = BASE_DIR / "app" / "features" / "feature_schema.yml"
FEATURE_SCHEMA_VERSION = "v1"
