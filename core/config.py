from functools import lru_cache
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    news_api_key: str | None = None
    hf_token: str | None = None
    hf_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    cryptocompare_api_key: str | None = None
    deepseek_api_key: str | None = None
    database_url: str
    redis_url: str

    class Config:
        env_prefix = ''
        case_sensitive = False
        env_file = '.env'

    @validator('database_url', 'redis_url')
    def validate_urls(cls, v: str) -> str:
        if not v:
            raise ValueError('Environment variable must be set')
        return v

    @validator(
        'binance_api_key',
        'binance_api_secret',
        'news_api_key',
        'hf_token',
        'hf_model',
        'cryptocompare_api_key',
        'deepseek_api_key',
        pre=True,
    )
    def empty_strings_to_none(cls, v):
        return v or None


@lru_cache()
def get_settings() -> Settings:
    return Settings()
