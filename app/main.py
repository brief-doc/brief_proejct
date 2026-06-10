import os
import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes.auth import router as auth_router
from app.db.database import engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redis 연결
redis_client = redis.from_url(os.environ["REDIS_URL"])

# include API routers
app.include_router(auth_router)


@app.get("/")
def root():
    return {"message": "FastAPI + Postgres + Redis 정상 작동 중!"}


@app.get("/health")
def health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    redis_client.ping()
    return {"db": "ok", "redis": "ok"}


@app.get("/counter")
def counter():
    count = redis_client.incr("visits")
    return {"visits": count}
