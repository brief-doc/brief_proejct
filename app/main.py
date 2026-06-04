import os
from fastapi import FastAPI
from sqlalchemy import create_engine, text
import redis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL 연결
engine = create_engine(os.environ["DATABASE_URL"])

# Redis 연결
redis_client = redis.from_url(os.environ["REDIS_URL"])


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