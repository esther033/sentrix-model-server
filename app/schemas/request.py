from pydantic import BaseModel
from typing import Dict
from datetime import datetime


class DiagnoseRequest(BaseModel):
    timestamp: datetime
    featureSchemaVersion: str
    features: Dict[str, float]
