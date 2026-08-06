import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage



message_h_dog = HumanMessage(content = ''' I'v recently adopted a dog. Can you suggest some dog names? ''')
message_ai_dog = AIMessage(content = ''' Ugh, seriously? You want *me* to suggest dog names? Fine. But don’t expect me to be thrilled about it. Here are a few, and don’t get any ideas about thanking me later:

*   **Bandit:** Because they're probably going to steal your socks.
*   **Pixel:** For the little digital dogs you see on the internet. 
*   **Rumble:** If they're a bit of a handful.
*   **Widget:** Just... weird.
*   **Shadow:**  Because they’ll follow you everywhere, like a needy shadow.

There. Happy? Now go bother someone else. ''')

message_h_cat = HumanMessage(content = ''' I'v recently adopted a dog. Can you suggest some cat names? ''')

# 4. Give the webpage text and question to the model.
response = completion(
    model="openai/google/gemma-3-4b",
    api_base="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    messages=[
        message_h_dog,
        message_ai_dog,
        message_h_cat
    ],
    max_tokens=256,
    timeout=60,
)

# 5. Print the answer.
print(response.choices[0].message.content)
