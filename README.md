# RAG Chatbot

Chatbot hỏi đáp thông minh sử dụng RAG (Retrieval-Augmented Generation).

## Tech stack
- **Groq API** — LLM inference (llama-3.1-8b-instant)
- **Sentence Transformers** — semantic embeddings (all-MiniLM-L6-v2)
- **TF-IDF + FAISS** — hybrid vector search
- **FastAPI** — REST API với streaming (SSE)
- **SQLite** — lưu users, sessions, chats, documents

## Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| Upload tài liệu | Upload PDF/TXT, tự động chunk và index |
| Streaming response | Trả lời từng token qua Server-Sent Events |
| Multi-session chat | Nhiều cuộc hội thoại độc lập với context riêng |
| Hybrid RAG pipeline | Kết hợp TF-IDF (40%) + semantic search (60%) |
| JWT Auth | Đăng ký/đăng nhập, phân quyền user/admin |

## API Endpoints

### Auth
- `POST /register` — Đăng ký
- `POST /login` — Đăng nhập → JWT token

### Chat
- `POST /chat` — Hỏi đáp (có thể gắn session)
- `POST /chat/stream` — Hỏi đáp streaming (SSE)
- `GET /history?session_id=<id>` — Lịch sử hội thoại

### Sessions
- `POST /sessions` — Tạo session mới
- `GET /sessions` — Danh sách sessions
- `DELETE /sessions/{id}` — Xóa session

### Documents
- `POST /documents/upload` — Upload file PDF/TXT
- `GET /documents` — Danh sách tài liệu đã upload
- `DELETE /documents/{id}` — Xóa tài liệu

### Admin
- `GET /users` — Danh sách users (chỉ admin)
- `GET /health` — Trạng thái server

## Cách chạy

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn app_full:app --reload
```

Sau đó mở [http://localhost:8000/docs](http://localhost:8000/docs) để xem Swagger UI.

## Ví dụ streaming (JavaScript)

```javascript
const es = new EventSource('/chat/stream', {
  method: 'POST',
  headers: { Authorization: 'Bearer <token>' },
  body: JSON.stringify({ question: 'RAG là gì?', session_id: 1 })
});
es.onmessage = ({ data }) => {
  if (data === '[DONE]') return es.close();
  process.stdout.write(JSON.parse(data).token);
};
```

## Kiến trúc Hybrid RAG

```
Câu hỏi
   ├── TF-IDF vectorize  →  lexical score  (40%)
   └── SentenceTransformer →  semantic score (60%)
              ↓
       Hybrid ranking
              ↓
     Top-k context chunks
              ↓
     Session history (5 turns)
              ↓
     Groq LLM → Answer
```

## Học được gì
- Hybrid retrieval: kết hợp lexical + semantic search
- Streaming API với FastAPI và Server-Sent Events
- Multi-turn conversation với session isolation
- Per-user document management và vector index
- JWT authentication và role-based access control
