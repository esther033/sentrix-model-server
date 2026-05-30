from typing import Dict, List, Tuple
from app.services.model_loader import model_loader


def validate_and_build_vector(
    features: Dict[str, float],
    schema_version: str
) -> Tuple[List[float], List[str]]:
    """
    요청 feature dict를 검증하고 모델 입력 vector로 변환한다.

    Returns:
        (feature_vector, missing_features)
        missing_features가 비어있으면 정상.
    """
    if schema_version != model_loader.schema_version:
        raise ValueError(f"Unsupported schema version: {schema_version}")

    feature_cols   = model_loader.feature_cols
    missing        = [f for f in feature_cols if f not in features]

    if missing:
        return [], missing

    vector = [float(features[f]) for f in feature_cols]
    return vector, []
