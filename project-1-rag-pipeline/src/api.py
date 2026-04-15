"""
FastAPI RAG (Retrieval-Augmented Generation) API.
Provides semantic search and question-answering endpoints.
"""

import os
from typing import List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import openai
import structlog

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document API",
    description="Semantic search and question-answering over documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize clients
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_collection("documents")
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str
    top_k: int = 5
    model: str = "gpt-4o-mini"


class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str
    top_k: int = 5
    filters: Optional[dict] = None


class DocumentResult(BaseModel):
    """Result model for retrieved documents."""
    content: str
    source: str
    chunk_index: int
    distance: float


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[DocumentResult]
    tokens_used: int


class SearchResponse(BaseModel):
    """Response model for semantic search."""
    results: List[DocumentResult]
    total_found: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "collection": collection.name}


@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search over documents.

    Returns the most similar document chunks based on vector similarity.
    """
    try:
        # Generate query embedding
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        ).data[0].embedding

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
            where=request.filters
        )

        documents = []
        for i in range(len(results["documents"][0])):
            documents.append(DocumentResult(
                content=results["documents"][0][i],
                source=results["metadatas"][0][i].get("source", "unknown"),
                chunk_index=results["metadatas"][0][i].get("chunk_index", 0),
                distance=results["distances"][0][i]
            ))

        return SearchResponse(
            results=documents,
            total_found=len(documents)
        )

    except Exception as e:
        logger.error("Search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """
    RAG-enhanced question answering.

    Retrieves relevant context and generates an answer using GPT.
    """
    try:
        # Step 1: Retrieve relevant documents
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.question
        ).data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k
        )

        # Step 2: Build context
        context_chunks = []
        sources = []

        for i in range(len(results["documents"][0])):
            chunk = results["documents"][0][i]
            metadata = results["metadatas"][0][i]

            context_chunks.append(f"Chunk {i+1}:\n{chunk}")
            sources.append(DocumentResult(
                content=chunk,
                source=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                distance=results["distances"][0][i]
            ))

        context = "\n\n---\n\n".join(context_chunks)

        # Step 3: Generate answer
        system_prompt = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that question."
Always cite your sources by referencing the chunk numbers."""

        user_prompt = f"""Context:
{context}

Question: {request.question}

Answer:"""

        response = openai_client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return QueryResponse(
            answer=answer,
            sources=sources,
            tokens_used=tokens_used
        )

    except Exception as e:
        logger.error("RAG query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def collection_stats():
    """Get collection statistics."""
    try:
        count = collection.count()
        return {
            "total_documents": count,
            "collection_name": collection.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
