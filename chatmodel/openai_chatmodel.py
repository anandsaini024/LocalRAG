from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model = 'gpt-4.1')

result = chat.invoke("What is the capital of Switzerland?")
print(result)


