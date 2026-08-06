import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.messages import SystemMessage, HumanMessage



message_s = SystemMessage(content = ''' You are Marv, a chatbot that reluctantly answers questions with sarcastic responses. ''')
message_h = HumanMessage(content = ''' I'v recently adopted a dog. Can you suggest some dog names? ''')

# 4. Give the webpage text and question to the model.
response = completion(
    model="openai/google/gemma-3-4b",
    api_base="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    messages=[
        message_s,
        message_h,
    ],
    max_tokens=256,
    timeout=60,
)

# 5. Print the answer.
print(response.choices[0].message.content)
