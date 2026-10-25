from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 三项故意不给默认值,缺了启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    # DeepSeek 默认思考模式下不支持结构化输出的 tool_choice,ch01 用 disabled;
    # 换到 OpenAI 等其他家时留空即可,不发送 thinking 参数。
    chat_thinking: str = "disabled"
    token_budget: int = 2000


settings = Settings()
