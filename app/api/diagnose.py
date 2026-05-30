from fastapi import APIRouter, HTTPException
from app.schemas.request import DiagnoseRequest
from app.schemas.response import DiagnoseResponse, HealthResponse, ErrorResponse
from app.services.model_loader import model_loader
from app.services.inference_service import run_inference
from app.features.feature_validator import validate_and_build_vector
from app.core.config import FEATURE_SCHEMA_VERSION

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="UP" if model_loader.loaded else "DOWN",
        modelLoaded=model_loader.loaded,
        featureSchemaVersion=FEATURE_SCHEMA_VERSION,
        models={
            "detector":   "IsolationForest",
            "classifier": "XGBoost",
        }
    )


@router.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):

    # 1. 모델 로드 확인
    if not model_loader.loaded:
        raise HTTPException(
            status_code=503,
            detail={"errorCode": "MODEL_NOT_LOADED", "message": "Model files are missing or failed to load."}
        )

    # 2. schema version 확인
    if req.featureSchemaVersion != FEATURE_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail={"errorCode": "INVALID_SCHEMA_VERSION", "message": f"Unsupported feature schema version: {req.featureSchemaVersion}"}
        )

    # 3. feature 검증 및 vector 변환
    try:
        vector, missing = validate_and_build_vector(req.features, req.featureSchemaVersion)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"errorCode": "INVALID_SCHEMA_VERSION", "message": str(e)}
        )

    if missing:
        raise HTTPException(
            status_code=400,
            detail={"errorCode": "MISSING_FEATURE", "message": f"Required features missing: {missing[:3]}..."}
        )

    # 4. inference
    try:
        result = run_inference(
            timestamp=req.timestamp,
            schema_version=req.featureSchemaVersion,
            feature_vector=vector,
            features_raw=req.features,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "INFERENCE_FAILED", "message": f"Unexpected error during inference: {str(e)}"}
        )

    return result
