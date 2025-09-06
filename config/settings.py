from pydantic import BaseSettings


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash-exp"



    class Config:
        env_file = ".env"


settings = Settings()