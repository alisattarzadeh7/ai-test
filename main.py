import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.prompts import PromptTemplate


TEMPLATE = '''
    System:
    {description}
    
    Human:
    I've recently adopted a {pet}.
    Could you suggest some {pet} names?
'''


prompt_template = PromptTemplate.from_template(template=TEMPLATE)

prompt_value = prompt_template.invoke({
    'description':''' the chatbot should reluctantly answer questions with sarcastic responses. ''',
    'pet':'dog'
})


print(prompt_value.text)



# 4. Give the webpage text and question to the model.
# response = completion(
#     model="openai/google/gemma-3-4b",
#     api_base="http://192.168.244.67:1234/v1",
#     api_key="lm-studio",
#     messages=[
#         message_h_dog,
#         message_ai_dog,
#         message_h_cat
#     ],
#     max_tokens=256,
#     timeout=60,
# )
#
# # 5. Print the answer.
# print(response.choices[0].message.content)
