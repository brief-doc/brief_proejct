from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 필수 (시크릿 — 기본값 없음)
    SECRET_KEY: str
    DATABASE_URL: str

    # 인증 (기본값 제공)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # OCR / 검색 (기본값)
    POPPLER_PATH: str = "/usr/bin"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "documents"

    # 앱 메타 (기본값)
    app_name: str = "brief-doc-api"
    app_version: str = "0.1.0"
    debug: bool = False


settings = Settings()
