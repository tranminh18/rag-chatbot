from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Python là ngôn ngữ lập trình bậc cao, dễ đọc và linh hoạt.",
    "RAG giúp LLM trả lời dựa trên dữ liệu thực tế của bạn.",
    "Gồm 3 bước: retrieve, augment, generate.",
    "Vector database lưu trữ embeddings để tìm kiếm ngữ nghĩa.",
]

vectors = model.encode(chunks).astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

query = "RAG là gì?"
q_vec = model.encode([query]).astype("float32")
distances, indices = index.search(q_vec, k=2)

print("Câu hỏi:", query)
for i in indices[0]:
    print("->", chunks[i])