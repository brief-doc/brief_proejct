from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Pydantic v2 스타일로 설정을 하나로 모읍니다.
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # 기존 class Config에 있던 설정을 여기로 이동
    )

    SECRET_KEY: str 
    ALGORITHM: str 
    ACCESS_TOKEN_EXPIRE_MINUTES: int    
    REFRESH_TOKEN_EXPIRE_DAYS: int
    POPPLER_PATH: str
    elasticsearch_url: str
    elasticsearch_index: str
    app_name: str
    app_version: str
    debug: bool

    # DB 설정
    DATABASE_URL: str

settings = Settings()