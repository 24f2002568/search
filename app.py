from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"],  # Exposes all headers
)
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define request model
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Define response model
class SimilarityResponse(BaseModel):
    matches: List[str]

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using OpenAI's text-embedding-3-small model
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings
    """
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

@app.post("/similarity", response_model=SimilarityResponse)
async def find_similar_documents(request: SimilarityRequest):
    """
    Endpoint that accepts documents and a query, then returns the three most similar documents
    based on semantic similarity using OpenAI embeddings.
    """
    # Validate input
    if not request.docs:
        raise HTTPException(status_code=400, detail="Documents array cannot be empty")
    if not request.query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
    
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(request.query)
        
        # Generate embeddings for all documents and calculate similarities
        similarities = []
        
        for doc in request.docs:
            # Generate embedding for document
            doc_embedding = get_embedding(doc)
            
            # Calculate similarity
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Get indices of top 3 documents with highest similarity scores
        # If fewer than 3 documents, return all
        num_matches = min(3, len(request.docs))
        top_indices = np.argsort(similarities)[-num_matches:][::-1]
        
        # Get the actual documents for these indices
        matches = [request.docs[i] for i in top_indices]
        
        return SimilarityResponse(matches=matches)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

# For running the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
