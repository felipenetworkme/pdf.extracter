from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str
    metadata: dict

@app.get("/")
def read_root():
    return {"message": "LangChain Embedder API is running", "status": "ok"}

@app.post("/embed")
def embed_text(request: EmbedRequest):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n", ".", " "]
    )

    chunks = splitter.split_text(request.text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    vector_store = Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="central_kb",
        metadata=[request.metadata] * len(chunks),
        client=client
    )

    return {"status": "ok", "chunks": len(chunks)}
