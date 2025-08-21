from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
from retriever_setup import ensemble_retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


# 1) Reranker
rerank_model = "rerank-english-v3.0"
compressor = CohereRerank(
    model=rerank_model,
    cohere_api_key=os.getenv("COHERE_API_KEY")
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever,
)

@tool(response_format="content_and_artifact")
def retrieve_compressed(query: str):
    """Hybrid (BM25+FAISS) -> Cohere Rerank ile sıkıştırılmış retrival."""
    retrieved_docs = compression_retriever.invoke(query)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# 2) Graph
def query_or_respond_rerank(state: MessagesState):
    """Generate tool call for compressed retrieval or respond."""
    # Tool kullanımı zorunlu değilse bu policy'yi çıkarabilirsin
    policy = SystemMessage(
        "Call exactly ONE tool: retrieve_compressed."
    )
    messages = [policy] + state["messages"]
    llm_with_tools = llm.bind_tools([retrieve_compressed])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tools_rerank = ToolNode([retrieve_compressed])

def generate_rerank(state: MessagesState):
    """Generate answer (compressed pipeline)."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an advanced assistant specialized in question-answering tasks. "
        "Utilize the following pieces of retrieved context to formulate a comprehensive answer to the question. "
        "Ensure to include the source documents in your response. If the answer is not known, respond with 'I don't know'. "
        "İf the question is not clear, ask for clarification.\n\n"
        "Please keep your answer concise, and always provide your source as url.\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder_rerank = StateGraph(MessagesState)
graph_builder_rerank.add_node("query_or_respond_rerank", query_or_respond_rerank)
graph_builder_rerank.add_node("tools_rerank", tools_rerank)
graph_builder_rerank.add_node("generate_rerank", generate_rerank)

graph_builder_rerank.set_entry_point("query_or_respond_rerank")
graph_builder_rerank.add_conditional_edges("query_or_respond_rerank", tools_condition, {END: END, "tools": "tools_rerank"})
graph_builder_rerank.add_edge("tools_rerank", "generate_rerank")
graph_builder_rerank.add_edge("generate_rerank", END)

memory = MemorySaver()
graph_rerank = graph_builder_rerank.compile(checkpointer=memory)

# 3) API’den çağırılacak fonksiyon
def ask_query(query: str):
    config = {"configurable": {"thread_id": "api-thread1"}}
    result = graph_rerank.invoke({"messages":[{"role":"user","content":query}]}, config=config)
    return result["messages"][-1].content