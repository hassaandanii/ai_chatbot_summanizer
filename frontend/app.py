import gradio as gr
import requests
import json

# CHANGED: Pointing to the new internal backend port 9001
BACKEND_URL = "http://backend:9001"

def summarize_logic(text):
    if not text:
        return "Please enter text.", "N/A"
    
    try:
        payload = {"text": text}
        response = requests.post(f"{BACKEND_URL}/summarize", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data["summary"], f"Success! ID: {data['id']}"
        else:
            return f"Error: {response.text}", "Failed"
    except Exception as e:
        return f"Connection Error: {str(e)}", "Failed"

def get_history_logic():
    try:
        response = requests.get(f"{BACKEND_URL}/history")
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        return "Could not fetch history"
    except:
        return "Connection Error"

with gr.Blocks(title="AI Summarizer") as demo:
    gr.Markdown("# 🤖 AI Summarizer & Knowledge Base")
    
    with gr.Tab("Summarizer"):
        with gr.Row():
            input_text = gr.Textbox(lines=5, label="Input Text")
            output_text = gr.Textbox(label="Summary")
        
        status_msg = gr.Label(label="Status")
        submit_btn = gr.Button("Summarize", variant="primary")
        submit_btn.click(fn=summarize_logic, inputs=input_text, outputs=[output_text, status_msg])

    with gr.Tab("Database History"):
        gr.Markdown("View data persisted in ChromaDB vector store.")
        refresh_btn = gr.Button("Refresh History")
        history_box = gr.JSON(label="Stored Data")
        refresh_btn.click(fn=get_history_logic, inputs=None, outputs=history_box)

if __name__ == "__main__":
    # CHANGED: Launching on port 9000
    demo.launch(server_name="0.0.0.0", server_port=9000)