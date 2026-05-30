import numpy as np
from typing import Dict, List
from app.services.model_loader import model_loader
from app.services.evidence_service import get_evidence_features
from app.schemas.response import (
    DetectionResult, ClassificationResult,
    FaultTypeConfidence, DiagnoseResponse, EvidenceFeature
)
from datetime import datetime


def run_inference(
    timestamp: datetime,
    schema_version: str,
    feature_vector: List[float],
    features_raw: Dict[str, float]
) -> DiagnoseResponse:
    """
    feature vector를 받아 anomaly detection + fault classification을 수행한다.
    """
    X = np.array(feature_vector).reshape(1, -1)

    # 1. Scaling
    X_scaled = model_loader.scaler.transform(X)

    # 2. Anomaly Detection (Isolation Forest)
    raw_score    = model_loader.detector.score_samples(X_scaled)[0]
    # score_samples 값의 범위를 0~1로 정규화하기 위해 학습 시 min/max를 쓰는 게 이상적이나,
    # 여기선 단순하게 IF의 decision_function을 변환
    # decision_function: 양수 = 정상, 음수 = 이상
    anomaly_score = float(1 / (1 + np.exp(raw_score * 10)))  # sigmoid 변환 → 0~1
    anomaly_score = round(anomaly_score, 4)

    threshold    = model_loader.threshold
    is_anomaly   = anomaly_score > threshold
    det_status   = "ANOMALY" if is_anomaly else "NORMAL"

    detection = DetectionResult(
        status=det_status,
        anomalyScore=anomaly_score,
        threshold=threshold,
    )

    # 3. Fault Classification (XGBoost)
    proba      = model_loader.classifier.predict_proba(X_scaled)[0]
    fault_types = model_loader.fault_types  # LabelEncoder 순서

    top_idx    = int(np.argmax(proba))
    fault_type = fault_types[top_idx]
    confidence = round(float(proba[top_idx]), 4)

    # top3
    sorted_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        FaultTypeConfidence(
            faultType=fault_types[i],
            confidence=round(float(proba[i]), 4)
        )
        for i in sorted_idx
    ]

    classification = ClassificationResult(
        faultType=fault_type,
        confidence=confidence,
        top3=top3,
    )

    # 4. Evidence Features
    evidence = get_evidence_features(fault_type, features_raw)

    return DiagnoseResponse(
        timestamp=timestamp,
        featureSchemaVersion=schema_version,
        detection=detection,
        classification=classification,
        evidenceFeatures=evidence,
    )
