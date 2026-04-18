# RAG Chatbot

Chatbot hỏi đáp thông minh sử dụng RAG (Retrieval-Augmented Generation).

## Tech stack
- Groq API (LLM)
- Sentence Transformers (embedding)
- FAISS / Qdrant (vector search)
- FastAPI (REST API)
- Streamlit (giao diện web)

## Các file chính
- `chat.py` — chatbot cơ bản với lịch sử hội thoại
- `rag.py` — RAG pipeline hoàn chỉnh
- `main.py` — FastAPI endpoint
- `embedder.py` — tạo và so sánh embeddings
- `vector_store.py` — tìm kiếm với FAISS
- `chunker.py` — cắt văn bản thành chunks

## Cách chạy
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Học được gì
- Cách gọi LLM API
- Chunking, embedding, vector search
- Xây RAG pipeline từ đầu
- Đánh giá chất lượng RAG
