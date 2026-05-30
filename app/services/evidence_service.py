from typing import List, Dict
from app.schemas.response import EvidenceFeature
from app.services.model_loader import model_loader


def get_evidence_features(
    fault_type: str,
    features: Dict[str, float],
    top_n: int = 4
) -> List[EvidenceFeature]:
    """
    rule-based evidence: fault type별 주요 feature를 feature_schema.yml에서 읽어 반환한다.
    feature 값이 클수록 score가 높다고 가정 (정규화 후 상대 비교).
    """
    if fault_type == "NORMAL":
        return []

    rules        = model_loader.evidence_rules
    target_feats = rules.get(fault_type, [])

    if not target_feats:
        return []

    # 해당 feature들의 값을 추출
    evidence = []
    values   = [abs(features.get(f, 0.0)) for f in target_feats]
    max_val  = max(values) if values else 1.0
    if max_val == 0:
        max_val = 1.0

    for feat, val in zip(target_feats, values):
        score = round(val / max_val, 4)
        evidence.append(EvidenceFeature(featureName=feat, score=score))

    # score 내림차순 정렬
    evidence.sort(key=lambda x: x.score, reverse=True)
    return evidence[:top_n]
