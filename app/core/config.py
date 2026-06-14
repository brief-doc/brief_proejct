from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 비밀값 → .env 필수 (기본값 X)
    SECRET_KEY: str
    DATABASE_URL: str

    # 기본값 있는 설정 (.env에 있으면 덮어씀)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    POPPLER_PATH: str = "/usr/bin"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "documents"
    app_name: str = "PDF RAG"
    app_version: str = "1.0.0"
    debug: bool = False


settings = Settings()
