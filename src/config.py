from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = (
        "postgresql+asyncpg://congress:congress@localhost:5432/congress_predictions"
    )
    database_url_sync: str = (
        "postgresql://congress:congress@localhost:5432/congress_predictions"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Neo4j (Phase 3)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "congress_neo4j"

    # External APIs
    fmp_api_key: str = ""
    fec_api_key: str = ""
    twitter_bearer_token: str = ""
    congress_gov_api_key: str = ""
    gnews_api_key: str = ""
    newsdata_api_key: str = ""
    govinfo_api_key: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Production (Phase 8)
    rate_limit_per_minute: int = 100
    rate_limit_authenticated: int = 1000
    api_keys: str = ""  # comma-separated API keys


settings = Settings()
