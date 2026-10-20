# 项目环境配置

## 安装项目依赖包

1. 基础环境：Python3.12+

2. 安装UV
```shell
pip install uv
set UV_INDEX=https://mirrors.aliyun.com/pypi/simple
```

3. 安装Python依赖包
```shell
uv sync --python 3.12 --all-extras
```

4. 切换到本地环境(.venv)，请安装whl包
```shell
cd .venv/Scripts
activate
```

## 启动Jupyter编辑器

```shell
jupyter notebook
```

## 模型加载

1. 在项目根目录下新建`.env`文件，并添加以下内容
```text
API_KEY=your_api_key
BASE_URL=model_base_url
```

2. 加载模型  
如果需要调用模型，请直接在notebook中添加以下代码（注意需要加载项目根目录到环境的path中），不需要额外的配置。
```python
import os
from dotenv import load_dotenv, find_dotenv

loaded = load_dotenv(find_dotenv(), override=True)
# 从环境变量中获取 OpenAI API Key 或者直接赋值
API_KEY = os.getenv("API_KEY")

# Read the model settings from the environment.
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
```

## MiniMax configuration reference

Set `MODEL_NAME` and `BASE_URL` in `.env` when using the MiniMax API:

```text
API_KEY=your_api_key
MODEL_NAME=MiniMax-M3.1-Flash
BASE_URL=https://api.minimax.io/v1
```

The active text helpers (`utils/text2text_chat.py`, `utils/langchain_chat.py`,
and `utils/chat_robot.py`) read these settings. For a direct MiniMax endpoint,
an omitted model defaults to `MiniMax-M3.1-Flash`; `MiniMax-M3` is the other
current choice. Explicit `MODEL_NAME` values, including saved legacy or custom
IDs, pass through unchanged. `CHAT_MODEL_NAME` overrides `MODEL_NAME` in the
LangChain helpers, and `get_completions(..., model_endpoint=...)` overrides it
for an individual call. Other endpoints retain their existing model defaults.
The LangChain embedding model and endpoint remain on SiliconFlow; set
`EMBEDDING_API_KEY` to its key when `API_KEY` belongs to MiniMax. Image helpers
retain their existing configuration and credentials.

Choose the endpoint that matches the account region and client protocol:

| Region | OpenAI-compatible endpoint | Anthropic-compatible endpoint | Documentation |
| --- | --- | --- | --- |
| Global | `https://api.minimax.io/v1` | `https://api.minimax.io/anthropic` | `https://platform.minimax.io/docs` |
| China | `https://api.minimaxi.com/v1` | `https://api.minimaxi.com/anthropic` | `https://platform.minimaxi.com/docs` |

These helpers use Chat Completions; the Anthropic endpoints above are a
reference for clients using that protocol, not interchangeable helper URLs.
Use the endpoint and key for your account region. Gateway routes require their
own model IDs and metadata; no gateway model is selected automatically.

Current direct API model metadata:

| Model | Context window | Input modalities | Thinking modes |
| --- | ---: | --- | --- |
| `MiniMax-M3.1-Flash` | 1,000,000 | text, image, video | adaptive only |
| `MiniMax-M3` | 1,000,000 | text, image, video | adaptive, disabled |

Flash supports up to 524,288 output tokens, including thinking.

Standard pricing in USD per million tokens:

| Model / input length | Input | Output | Cache read |
| --- | ---: | ---: | ---: |
| `MiniMax-M3.1-Flash` | $0.15 | $0.60 | $0.015 |
| `MiniMax-M3`, at most 512k input tokens | $0.30 | $1.20 | $0.06 |
| `MiniMax-M3`, above 512k input tokens | $0.60 | $2.40 | $0.12 |

Cache-write pricing is not listed. M3 priority admission costs 1.5 times the
standard rate; Flash does not accept `service_tier`.

Flash always thinks: `thinking.type="disabled"`, `reasoning_effort="none"`, and
`reasoning_split=false` are unsupported. Its effort levels are `low`, `medium`,
`high`, `xhigh`, and `max` (the default). Direct Chat Completions uses
`reasoning_effort`; Anthropic clients use `output_config.effort`. M3 supports
adaptive or disabled thinking, defaults to adaptive with Chat Completions and
disabled with Anthropic, and does not tune thinking depth with effort.

The examples leave MiniMax thinking and effort at their API defaults. Their
4,096-token generation budget includes thinking; increase `max_tokens` for
longer answers (for example, `get_completions(prompt, max_tokens=16384)`).
The LangChain adapter uses full, non-streaming responses to retain both
`reasoning_content` and tool calls with the pinned SDK. Keep complete assistant
messages and tool results when extending a conversation.

Sources: [models](https://platform.minimax.io/docs/guides/models-intro),
[Chat Completions](https://platform.minimax.io/docs/api-reference/text-openai-api),
[Anthropic compatibility](https://platform.minimax.io/docs/api-reference/text-anthropic-api),
and [pricing](https://platform.minimax.io/docs/pricing/overview#models/text).
