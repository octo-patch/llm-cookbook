"""Keep MiniMax reasoning history with the cookbook's pinned chat SDK."""

from langchain_openai import ChatOpenAI

from utils.chat_settings import is_minimax_url


class MiniMaxChatOpenAI(ChatOpenAI):
    """Preserve reasoning in the examples' non-streaming tool conversations."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        body = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, body["choices"]):
            reasoning = choice["message"].get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        for message, serialized in zip(messages, payload["messages"]):
            if message.type == "ai" and "reasoning_content" in message.additional_kwargs:
                serialized["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        return payload


def create_chat_model(**kwargs):
    if is_minimax_url(kwargs["openai_api_base"]):
        # Leave adaptive thinking and effort at the API defaults. The pinned
        # SDK's streaming parser drops reasoning_content, so use full responses.
        kwargs["extra_body"] = {}
        kwargs["disable_streaming"] = True
        return MiniMaxChatOpenAI(**kwargs)
    return ChatOpenAI(**kwargs)
