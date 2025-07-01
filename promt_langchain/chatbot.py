from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model = 'claude-3-5-sonnet-20241022')

chat_history = [
    SystemMessage(content = 'You are a helpul Ai Assistant')
]
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ", result.content)

print(chat_history)