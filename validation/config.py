import os
from typing import Optional
try:
    # Pydantic v2
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # fallback for environments still on pydantic v1
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = None  # type: ignore

class ValidationSettings(BaseSettings):
    """Validation 서비스 설정"""
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 5004
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Validation 설정
    MAX_CONCURRENT_VALIDATIONS: int = 5
    
    # 콜백 설정
    CALLBACK_ENABLED: bool = True
    CALLBACK_TIMEOUT: int = 30
    CALLBACK_RETRY_COUNT: int = 3
    CALLBACK_URL: str = "http://api-server:5002/validation/yolo/result"  # API 서버 콜백 URL
    
    # Pydantic v2 style settings config
    if 'SettingsConfigDict' in globals() and SettingsConfigDict:
        model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    else:
        # Pydantic v1 compatibility
        class Config:  # type: ignore
            env_file = ".env"
            case_sensitive = False

# 전역 설정 인스턴스
settings = ValidationSettings()

