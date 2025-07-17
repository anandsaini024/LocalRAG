from langchain_anthropic import ChatAnthropic
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import voyageai
from dotenv import load_dotenv

load_dotenv()

chat = ChatAnthropic(model = 'claude-3-5-sonnet-20241022')

video_id = "shOLHtyUaFg" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript_list[0])

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

class VoyageAIEmbeddings(Embeddings):
    def __init__(self, model="voyage-3.5-lite", **kwargs):
        self.client = voyageai.Client()
        self.model = model
        self.embed_kwargs = kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(texts, model=self.model, **self.embed_kwargs)
        return response.embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed([text], model=self.model, **self.embed_kwargs)
        return response.embeddings[0]

embeddings = VoyageAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
print(vector_store.index_to_docstore_id)
print(vector_store.get_by_ids([vector_store.index_to_docstore_id[0]]))

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
print(retriever)

print(retriever.invoke('What are meta giants saying?'))

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)




#final_prompt = prompt.invoke({"context": context_text, "question": question})


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | chat | parser

print(main_chain.invoke('did it talk about nuclear ?'))