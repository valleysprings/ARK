import asyncio
import os
from dataclasses import dataclass
import numpy as np
from .storage import NanoVectorDB

@dataclass
class NanoVectorDBStorage():
    embedding_func: callable
    namespace: str
    working_dir: str
    
    cosine_better_than_threshold: float = 0.2
    _max_batch_size: int = 128

    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.working_dir, f"vdb_{self.namespace}.json"
        )
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )

    async def upsert(self, data: dict[str, dict]):
        print(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            print("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in {"entity_name"}},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        await self.index_done_callback()
        
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()
        
    def get_all_data(self):
        return self._client.get_all_data()

    def get_embedding_matrix(self):
        return self._client.get_embedding_matrix()
    
    def get_similarity_matrix(self):
        embeddings = self.get_embedding_matrix()
        print(f"embeddings shape: {embeddings.shape}")
        similarity_matrix = embeddings @ embeddings.T
        # diagonal is 1
        np.fill_diagonal(similarity_matrix, 1)
        return similarity_matrix
    
    def cut_off_similarity_matrix(self, similarity_matrix, threshold=0.8):
        # threshold the similarity matrix
        cut_off_matrix = similarity_matrix.copy()
        cut_off_matrix[cut_off_matrix < threshold] = 0
        return cut_off_matrix

