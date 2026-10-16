from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    broker_url: str = ''
    api_key: str = ''
    broker_topic: str = ''
    
    db_host: str = ''
    db_port: str = ''
    db_user: str = ''
    db_password: str = ''
    
    @property  # pyright: ignore[reportArgumentType]
    def db_url() -> str:
        ...
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf8", extra="ignore")

settings = Config()
