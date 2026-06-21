from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import pickle
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def buildIndex(chunks: list[str]) -> np.ndarray:
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    faiss.write_index(index, "./vectorstore/index.faiss")
    
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    
    bm25 = BM25Okapi(tokenized_chunks)
    
    with open('./vectorstore/bm25.pkl', "wb") as f: pickle.dump(bm25,f)
    with open('./vectorstore/chunks.json', "w") as f: json.dump(chunks, f, indent=2)
    
    return "PASS"



def getEmbeddings(chunks: list[str]) -> np.ndarray:
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings.astype("float32")

    