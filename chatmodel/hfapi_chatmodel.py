from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "conversational",
    provider = "fireworks-ai"
)


result = llm.invoke("what is the capital of Switzerland?")

print(result.content)
