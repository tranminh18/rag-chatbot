from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import numpy as np

load_dotenv()
client = Groq()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = open("sample.txt").read().split("\n\n")
chunks = [c.strip() for c in chunks if c.strip()]
vectors = embed_model.encode(chunks).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

def rag_answer(question):
    q_vec = embed_model.encode([question]).astype("float32")
    _, idx = index.search(q_vec, k=2)
    context = "\n".join([chunks[i] for i in idx[0]])

    prompt = f"""Dựa vào thông tin sau:
{context}

Trả lời câu hỏi: {question}
Chỉ dùng thông tin trên. Nếu không có, nói "Tôi không có thông tin này"."""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

print(rag_answer("RAG gồm mấy bước?"))
print("---")
print(rag_answer("FAISS là gì?"))