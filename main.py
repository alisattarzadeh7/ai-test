import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableGenerator

chat_template = ChatPromptTemplate.from_messages([
    (
        "human",
        "I've recently adopted a {pet} which is a {breed}. "
        "Could you suggest training tips?"
    )
])



def call_llm_stream(input_stream):
    for prompt_value in input_stream:

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
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content

            if content:
                yield content


llm = RunnableGenerator(call_llm_stream)

chain = chat_template | llm


inputs = [
    {"pet": "dog", "breed": "German Shepherd"},
    {"pet": "cat", "breed": "Siamese"},
    {"pet": "dog", "breed": "Golden Retriever"},
]

responses = chain.batch(inputs)

response = chain.stream({
"pet": "dog", "breed": "Labrador"
})

for i in response:
    print(i,end = '')