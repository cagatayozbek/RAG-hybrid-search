# embeddings.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# .env dosyasını yükle
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    
)