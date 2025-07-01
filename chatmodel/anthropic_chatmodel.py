from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chat = ChatAnthropic(model = 'claude-3-5-sonnet-20241022')

result = chat.invoke("What is the capital of Switzerland")

print(result.content)