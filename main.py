import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

chat_template = ChatPromptTemplate.from_messages([
    (
        "human",
        "I've recently adopted a {pet} which is a {breed}. "
        "Could you suggest training tips?"
    )
])


def call_llm(prompt_value):
    # Convert LangChain messages -> OpenAI/LiteLLM format
    messages = [
        {
            "role": "user" if msg.type == "human" else msg.type,
            "content": msg.content
        }
        for msg in prompt_value.to_messages()
    ]

    response = completion(
        model="openai/google/gemma-3-4b",
        api_base="http://192.168.244.67:1234/v1",
        api_key="lm-studio",
        messages=messages,
        max_tokens=256,
        timeout=60,
    )

    return response.choices[0].message.content


llm = RunnableLambda(call_llm)

chain = chat_template | llm


inputs = [
    {"pet": "dog", "breed": "Labrador"},
    {"pet": "dog", "breed": "German Shepherd"},
    {"pet": "cat", "breed": "Siamese"},
    {"pet": "dog", "breed": "Golden Retriever"},
]

responses = chain.batch(inputs)

for input_data, response in zip(inputs, responses):
    print(f"\n{input_data['pet']} - {input_data['breed']}")
    print(response)