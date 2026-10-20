"""Environment settings shared by the active text examples."""

import os
from urllib.parse import urlsplit


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
MINIMAX_MODELS = ("MiniMax-M3.1-Flash", "MiniMax-M3")


def is_minimax_url(base_url):
    url = urlsplit(base_url)
    return (
        url.scheme == "https"
        and url.netloc in ("api.minimax.io", "api.minimaxi.com")
        and url.path.rstrip("/") == "/v1"
    )


def chat_settings(default_model, model_variable="MODEL_NAME"):
    base_url = os.getenv("BASE_URL", DEFAULT_BASE_URL)
    default = MINIMAX_MODELS[0] if is_minimax_url(base_url) else default_model
    model = os.getenv(model_variable, os.getenv("MODEL_NAME", default))
    return base_url, model
