import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb

app = FastAPI()

# --- Load Models Globally (Startup) ---
print("Loading Summarization Model...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

print("Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Database Connection ---
print("Connecting to ChromaDB...")
chroma_client = chromadb.HttpClient(host='chromadb', port=8000)
collection = chroma_client.get_or_create_collection(name="summary_history")

# --- Request Model ---
class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "Backend is running", "models_loaded": True}

@app.post("/summarize")
def summarize_text(request: TextRequest):
    input_text = request.text.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # 1. Summarize
        summary_result = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        summary_text = summary_result[0]['summary_text']

        # 2. Embed
        embedding = embedder.encode(input_text).tolist()

        # 3. Save to DB
        doc_id = str(uuid.uuid4())
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[input_text],
            metadatas=[{"summary": summary_text}]
        )

        return {
            "summary": summary_text,
            "id": doc_id,
            "message": "Saved to Vector DB"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    # Fetch last 10 items
    count = collection.count()
    if count == 0:
        return {"history": []}
    
    results = collection.peek(limit=10)

    # CRITICAL FIX: Remove the 'embeddings' field.
    # The vectors are too big (384 numbers) and crash the JSON response.
    if 'embeddings' in results:
        del results['embeddings']
        
    return {"history": results}