from fastapi import FastAPI
from pydantic import BaseModel
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

class Chunk(BaseModel):
    content: str
    tokens: int

class CreateChunkReuqest(BaseModel):
    content: str
    max_chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None

app = FastAPI()


@app.post("/chunks")
async def create_chunks(request: CreateChunkReuqest):
    chunk_size = request.max_chunk_size_tokens or 800
    chunk_overlap = request.chunk_overlap_tokens or 400

    delta = chunk_size - chunk_overlap
    chunks = []

    tokens = enc.encode(request.content)
    for i in range(0, len(tokens) - delta, delta):
        real_tokens = tokens[i:(i + chunk_size)]
        chunks.append(Chunk(content=enc.decode(real_tokens), tokens=len(real_tokens)))
    
    return chunks
