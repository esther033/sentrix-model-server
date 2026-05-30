from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class DetectionResult(BaseModel):
    status: str           # NORMAL or ANOMALY
    anomalyScore: float
    threshold: float


class FaultTypeConfidence(BaseModel):
    faultType: str
    confidence: float


class ClassificationResult(BaseModel):
    faultType: str
    confidence: float
    top3: List[FaultTypeConfidence]


class EvidenceFeature(BaseModel):
    featureName: str
    score: float


class DiagnoseResponse(BaseModel):
    timestamp: datetime
    featureSchemaVersion: str
    detection: DetectionResult
    classification: ClassificationResult
    evidenceFeatures: List[EvidenceFeature]


class HealthResponse(BaseModel):
    status: str
    modelLoaded: bool
    featureSchemaVersion: str
    models: dict


class ErrorResponse(BaseModel):
    errorCode: str
    message: str
