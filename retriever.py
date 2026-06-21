import os
import numpy as np
import faiss
import json
import pickle
from collections import defaultdict
import logging
import requests
import aiohttp



logging.info(os.getcwd())

async def retrieveChunks(userQuery: str):
    if not os.path.exists("vectorstore/index.faiss"):
        return {"message":"Try again"}
    
    data = {}
    
    with open("./vectorstore/index.faiss", "rb") as f:
        index = f.read()
        data['index'] = faiss.deserialize_index(np.frombuffer(index, dtype=np.uint8))
    with open("./vectorstore/chunks.json", "r") as f:
        all_chunks = json.load(f)
        data['chunks'] = all_chunks
    with open('./vectorstore/bm25.pkl','rb') as f:
        data['bm25'] = pickle.load(f)   
        
    
    query_tokens = userQuery.lower().split()

    scores = data["bm25"].get_scores(query_tokens)
    
    k = 20

    bm25_ids = np.argsort(scores)[::-1][:k]
    bm25_scores = scores[bm25_ids]
    
    input = [userQuery]
    
    async with aiohttp.ClientSession() as session:

        async with session.get(
            "http://localhost:8000/embeddings",
            params={"queryList": input}
        ) as response:

            embeddingsData = await response.json()

    queryEmbeddings = np.array(
        embeddingsData["embeddings"],
        dtype="float32"
    )
    
    distances, indices = data['index'].search(
    queryEmbeddings,
    k=20
)
    faissIds = indices[0]
    faissScores = distances[0]
    
    
    rrf_scores = defaultdict(float)

    K = 60

    for rank, doc_id in enumerate(bm25_ids, start=1):
        rrf_scores[doc_id] += 1 / (K + rank)

    for rank, doc_id in enumerate(faissIds, start=1):
        rrf_scores[doc_id] += 1 / (K + rank)

    final_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )
    
    resultantChunks = []
    
    for i in final_ids:
        resultantChunks.append(data['chunks'][i])
        
    return resultantChunks
    