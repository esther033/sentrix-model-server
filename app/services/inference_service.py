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
    raw_score = float(model_loader.detector.score_samples(X_scaled)[0])

    # 학습 데이터 기준 min-max 정규화 → 0~1 (높을수록 이상)
    score_min = model_loader.metadata["score_min"]
    score_max = model_loader.metadata["score_max"]

    if score_max != score_min:
        normalized    = (raw_score - score_min) / (score_max - score_min)
        anomaly_score = float(1.0 - normalized)  # 낮을수록 이상 → 반전
    else:
        anomaly_score = 0.0

    anomaly_score = round(float(np.clip(anomaly_score, 0.0, 1.0)), 4)

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

    # detection이 NORMAL이면 classification 결과 무시하고 NORMAL로 오버라이드
    if det_status == "NORMAL":
        fault_type = "NORMAL"
        confidence = 1.0
        top3 = [FaultTypeConfidence(faultType="NORMAL", confidence=1.0)]

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