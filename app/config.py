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
    # 本地 MySQL,默认值对齐 docker-compose;同步 ORM + PyMySQL。
    database_url: str = (
        "mysql+pymysql://mewhelp:mewhelp@127.0.0.1:3307/mewhelp?charset=utf8mb4"
    )
    tool_timeout: float = 10.0
    tool_max_retries: int = 2
    # ch03 知识库:嵌入走硅基流动 BGE-M3(OpenAI 兼容);key 验收前由用户补入 .env
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    milvus_uri: str = "http://localhost:19530"
    dense_top_k: int = 5
    dense_min_score: float = 0.40


settings = Settings()
