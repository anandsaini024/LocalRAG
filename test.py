import voyageai

from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client()
result = vo.embed(["hello world"], model="voyage-3.5")
print(result.embeddings)