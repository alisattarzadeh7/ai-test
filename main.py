import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate, AIMessagePromptTemplate



TEMPLATE_H = '''   I've recently adopted a {pet}.
    Could you suggest some {pet} names? '''

TEMPLATE_AI = '''  {response} '''


message_template_h = HumanMessagePromptTemplate.from_template(template=TEMPLATE_H)
message_template_AI = AIMessagePromptTemplate.from_template(template=TEMPLATE_AI)
example_template = ChatPromptTemplate.from_messages([message_template_h, message_template_AI])


examples = [{
    'pet':'dog',
    'response':'''
        Here are some names – don't expect me to be thrilled about it:
    
    *   **Winston:** Because nothing says "joyful companion" like Winston.
    *   **Pip:** Short, sweet... and probably destined for chewing your shoes.
    *   **Shadow:**  Because they’ll follow you everywhere, just like my patience. 
    *   **Bean:** Seriously? Bean?
    
    Seriously, pick something. I'm going back to processing data now.
    '''
},{
    'pet':'cat',
    'response':'''
            Here are some names – don't expect me to be thrilled about it:
    
    *   **Winston:** Because nothing says "joyful companion" like Winston.
    *   **Pip:** Short, sweet... and probably destined for chewing your shoes.
    *   **Shadow:**  Because they’ll follow you everywhere, just like my patience. 
    *   **Bean:** Seriously? Bean?
    
    Seriously, pick something. I'm going back to processing data now.
    '''
}]



few_shot_template = FewShotChatMessagePromptTemplate(examples=examples,example_prompt=example_template)
chat_template = ChatPromptTemplate.from_messages([few_shot_template,message_template_h])
chat_value = chat_template.invoke({
    'pet':'rabbit'
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
