from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.diagnose import router
from app.services.model_loader import model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 모델 로드
    print("[SentriX Model Server] 모델 로딩 중...")
    model_loader.load()
    print("[SentriX Model Server] 준비 완료")
    yield
    # 서버 종료 시 (필요 시 정리)
    print("[SentriX Model Server] 종료")


app = FastAPI(
    title="SentriX Model Server",
    description="Anomaly Detection & Fault Classification for SentriX",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
