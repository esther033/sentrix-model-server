import json
import joblib
import yaml
import numpy as np
from pathlib import Path
from app.core.config import MODEL_DIR, FEATURE_SCHEMA_PATH


class ModelLoader:
    def __init__(self):
        self.detector   = None
        self.classifier = None
        self.scaler     = None
        self.metadata   = None
        self.schema     = None
        self.loaded     = False

    def load(self):
        try:
            self.detector   = joblib.load(MODEL_DIR / "detector.pkl")
            self.classifier = joblib.load(MODEL_DIR / "classifier.pkl")
            self.scaler     = joblib.load(MODEL_DIR / "scaler.pkl")

            with open(MODEL_DIR / "metadata.json") as f:
                self.metadata = json.load(f)

            with open(FEATURE_SCHEMA_PATH) as f:
                self.schema = yaml.safe_load(f)

            self.loaded = True
            print(f"[ModelLoader] 모델 로드 완료")
            print(f"  feature 수: {self.metadata['n_features']}")
            print(f"  threshold:  {self.metadata['threshold']:.4f}")
            print(f"  fault_types: {self.metadata['fault_types']}")

        except Exception as e:
            self.loaded = False
            print(f"[ModelLoader] 로드 실패: {e}")
            raise

    @property
    def feature_cols(self) -> list:
        return self.metadata["feature_cols"]

    @property
    def threshold(self) -> float:
        return self.metadata["threshold"]

    @property
    def fault_types(self) -> list:
        return self.metadata["fault_types"]

    @property
    def schema_version(self) -> str:
        return self.metadata["schema_version"]

    @property
    def evidence_rules(self) -> dict:
        return self.schema.get("evidence_rules", {})


# 싱글톤
model_loader = ModelLoader()
