import os
import sys

from langchain_classic.output_parsers import DatetimeOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from litellm import completion
from langchain_core.messages import HumanMessage


message_h = HumanMessage(content=f'''  when was chatgpt launched?
    {DatetimeOutputParser().get_format_instructions()}

''')

print(message_h.content)

# # 4. Give the webpage text and question to the model.
response = completion(
    model="openai/google/gemma-3-4b",
    api_base="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    messages= [message_h],
    max_tokens=256,
    timeout=60,
)

date_output_parser  = DatetimeOutputParser()
response_parsed = date_output_parser.invoke(response.choices[0].message.content)
print(response_parsed)
