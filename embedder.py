from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python là ngôn ngữ lập trình",
    "Con mèo ngồi trên bàn",
    "Lập trình với Python rất dễ"
]

vectors = model.encode(sentences)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Độ giống nhau câu 1 và 3 (cùng Python):", round(cosine(vectors[0], vectors[2]), 3))
print("Độ giống nhau câu 1 và 2 (khác chủ đề):", round(cosine(vectors[0], vectors[1]), 3))