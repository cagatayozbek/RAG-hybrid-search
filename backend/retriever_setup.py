import faiss, json, re
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from embeddings import embeddings  

# 1) FAISS index yükle
index = faiss.read_index("data/faiss_index.bin")

# 2) chunks.json yükle
with open("data/chunks.json", "r") as f:
    chunked_data = json.load(f)

N = len(chunked_data)
assert index.ntotal == N, f"FAISS vektör sayısı ({index.ntotal}) != chunks.json sayısı ({N})"

# 3) Docstore
docstore_dict = {}
for i, c in enumerate(chunked_data):
    meta = {"title": c.get("title"), "source": c.get("source"), "chunk_id": i}
    docstore_dict[str(i)] = Document(page_content=c.get("text",""), metadata=meta)

docstore = InMemoryDocstore(docstore_dict)
index_to_docstore_id = {i: str(i) for i in range(N)}

# 4) Vector store (FAISS)
# embeddings nesnesini global import et

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 5) BM25 retriever
def simple_tokenize(s: str):
    return re.findall(r"\w+", s.lower())

docs = [
    Document(
        page_content=f"{c.get('title','')} {c.get('text','')}",
        metadata={"title": c.get("title",""), "source": c.get("source",""), "chunk_id": i}
    )
    for i, c in enumerate(chunked_data)
]

bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=simple_tokenize)
bm25_retriever.k = 5

# 6) Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weights=[0.3, 0.7],
)