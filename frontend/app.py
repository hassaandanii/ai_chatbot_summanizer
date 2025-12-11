import gradio as gr
import requests
import json

# The hostname 'backend' matches the service name in docker-compose
BACKEND_URL = "http://backend:8000"

def summarize_logic(text):
    if not text:
        return "Please enter text.", "N/A"
    
    try:
        payload = {"text": text}
        response = requests.post(f"{BACKEND_URL}/summarize", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data["summary"], f"Saved to ChromaDB (ID: {data['id']})"
        else:
            return f"Error: {response.text}", "Failed"
    except Exception as e:
        return f"Connection Error: {str(e)}", "Failed"

def get_history_logic():
    try:
        response = requests.get(f"{BACKEND_URL}/history")
        if response.status_code == 200:
            data = response.json()
            return json.dumps(data, indent=2)
        return "Could not fetch history"
    except:
        return "Connection Error"

# Build Gradio UI
with gr.Blocks(title="AI Summarizer & Knowledge Base") as demo:
    gr.Markdown("# 🤖 AI Summarizer with Vector Memory")
    gr.Markdown("This tool uses a Deep Learning Transformer model to summarize text and stores embeddings in ChromaDB.")
    
    with gr.Tab("Summarizer"):
        with gr.Row():
            input_text = gr.Textbox(lines=10, label="Input Text", placeholder="Paste article here...")
            output_text = gr.Textbox(label="Summary")
        
        status_msg = gr.Label(label="System Status")
        submit_btn = gr.Button("Summarize", variant="primary")
        
        submit_btn.click(fn=summarize_logic, inputs=input_text, outputs=[output_text, status_msg])

    with gr.Tab("Database History"):
        gr.Markdown("View raw data stored in ChromaDB vector store.")
        refresh_btn = gr.Button("Refresh History")
        history_box = gr.JSON(label="Vector DB Contents")
        
        refresh_btn.click(fn=get_history_logic, inputs=None, outputs=history_box)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)