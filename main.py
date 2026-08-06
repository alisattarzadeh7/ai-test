import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


TEMPLATE_S = '''
    {description}
'''

TEMPLATE_H = '''   I've recently adopted a {pet}.
    Could you suggest some {pet} names? '''


message_template_s = SystemMessagePromptTemplate.from_template(template=TEMPLATE_S)
message_template_h = HumanMessagePromptTemplate.from_template(template=TEMPLATE_H)
chat_template = ChatPromptTemplate.from_messages([message_template_s, message_template_h])


print(chat_template)
chat_value = chat_template.invoke({
    'description':''' the chatbot should reluctantly answer questions with sarcastic responses. ''',
    'pet':'dog'
})


messages = [
    {
        "role": (
            "system"
            if message.type == "system"
            else "user"
            if message.type == "human"
            else "assistant"
        ),
        "content": message.content,
    }
    for message in chat_value.to_messages()
]

print(chat_value)



# 4. Give the webpage text and question to the model.
response = completion(
    model="openai/google/gemma-3-4b",
    api_base="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    messages= messages,
    max_tokens=256,
    timeout=60,
)

# 5. Print the answer.
print(response.choices[0].message.content)
