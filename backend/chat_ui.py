import gradio as gr
from rerank_pipeline import graph_rerank, memory
import uuid

history = []
current_thread_id = str(uuid.uuid4())

def chat_fn(message, history):
    config = {"configurable": {"thread_id": current_thread_id}}
    result = graph_rerank.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    answer = result["messages"][-1].content
    return answer

def new_chat():
    global current_thread_id
    current_thread_id = str(uuid.uuid4())
    return []

with gr.Blocks(theme="soft") as demo:
    chatbot = gr.Chatbot(type="messages", label="RAG Chatbot")
    msg = gr.Textbox()
    clear_btn = gr.Button("Yeni Sohbet")
    
    def respond(message, history):
        answer = chat_fn(message, history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, ""
    
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.click(new_chat, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(debug=True)