import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

app = FastAPI()

# --- 1. Load ML Models (Pipeline) ---
print("Loading Summarization Model...")
# Using DistilBART because it is faster and lighter for student projects
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

print("Loading Embedding Model...")
# Used for Semantic Search/Vector Storage
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. Connect to Vector DB (ChromaDB) ---
print("Connecting to ChromaDB...")
# We connect to the 'chromadb' container defined in docker-compose
chroma_client = chromadb.HttpClient(host='chromadb', port=8000)

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="summary_history")

# --- 3. Data Models ---
class TextRequest(BaseModel):
    text: str

# --- 4. API Endpoints ---
@app.get("/")
def health_check():
    return {"status": "Backend is running", "models_loaded": True}

@app.post("/summarize")
def summarize_text(request: TextRequest):
    input_text = request.text.strip()
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # A. Perform Summarization
        # Limit input to 1024 tokens to prevent errors, generate summary
        summary_result = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        summary_text = summary_result[0]['summary_text']

        # B. Generate Embeddings (for Vector DB)
        embedding = embedder.encode(input_text).tolist()

        # C. Store in ChromaDB
        doc_id = str(uuid.uuid4())
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[input_text],
            metadatas=[{"summary": summary_text, "type": "generated"}]
        )

        return {
            "original_length": len(input_text),
            "summary": summary_text,
            "saved_to_db": True,
            "id": doc_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    # Retrieve last 10 items
    count = collection.count()
    if count == 0:
        return {"history": []}
    
    # Get the raw data
    results = collection.peek(limit=10)
    
    if 'embeddings' in results:
        del results['embeddings']
        
    return {"history": results}