#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: langchain_chat.py
@time: 2025/6/10 15:55
@project: llm-cookbook
@desc: 
"""
import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from utils.chat_settings import chat_settings
from utils.minimax_chat import create_chat_model

loaded = load_dotenv(find_dotenv(), override=True)
# 从环境变量中获取 OpenAI API Key 或者直接赋值
API_KEY = os.getenv("API_KEY")

# 如果您使用的是官方 API，就直接用 https://api.siliconflow.cn/v1 就行。
BASE_URL = "https://api.siliconflow.cn/v1"

# Text configuration is separate from the existing embedding endpoint.
CHAT_BASE_URL, CHAT_MODEL_NAME = chat_settings("deepseek-ai/DeepSeek-V3", "CHAT_MODEL_NAME")
chat_model = create_chat_model(temperature=0, model_name=CHAT_MODEL_NAME, max_tokens=4096,
                        openai_api_key=API_KEY, openai_api_base=CHAT_BASE_URL, max_retries=3,
                        seed=42, presence_penalty=0.1, frequency_penalty=0.1,
                        extra_body={
                            "enable_thinking": False
                        }
                        )

embedding_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("EMBEDDING_API_KEY", API_KEY),
    openai_api_base=BASE_URL,
    model="BAAI/bge-m3"
)