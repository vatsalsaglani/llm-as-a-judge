from openai import AsyncOpenAI
from configs import OPENAI_API_KEY
from typing import List, Dict, Union


class OpenAILLM:

    def __init__(self, **kwargs):
        api_key = OPENAI_API_KEY
        self.client = AsyncOpenAI(api_key=api_key)

    async def __complete__(self, messages: List[Dict], model: str, **kwargs):
        managed_messages = messages
        output = await self.client.chat.completions.create(
            messages=managed_messages, model=model, **kwargs)
        # print(output)
        usage = output.usage.__dict__
        output_content = output.choices[0].message
        return output_content, usage

    async def __stream__(self, messages: List[Dict], model: str, **kwargs):
        managed_messages = messages
        stream = await self.client.chat.completions.create(
            model=model, messages=managed_messages, stream=True, **kwargs)
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""
