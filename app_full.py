from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3, hashlib, hmac, base64, json, io, os, time, threading
from typing import Optional, List

app = FastAPI(title="RAG Chatbot API - Enhanced")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Embedding Model (optional — falls back to TF-IDF if unavailable) ─
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embed_model = None


# ─── Hybrid Knowledge Base ─────────────────────────────────────────
class KnowledgeBase:
    """Hybrid TF-IDF + semantic vector store with per-user document isolation."""

    def __init__(self):
        self.chunks: List[str] = []
        self.meta: List[dict] = []   # {"doc_id": int, "username": str | None}
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = None
        self.embeddings: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def add_chunks(self, new_chunks: List[str], doc_id: int, username: Optional[str]):
        with self._lock:
            self.chunks.extend(new_chunks)
            self.meta.extend([{"doc_id": doc_id, "username": username}] * len(new_chunks))
            self._rebuild()

    def remove_chunks_by_doc(self, doc_id: int):
        with self._lock:
            keep = [i for i, m in enumerate(self.meta) if m["doc_id"] != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            self.meta = [self.meta[i] for i in keep]
            self._rebuild()

    def _rebuild(self):
        if not self.chunks:
            self.tfidf_matrix = None
            self.embeddings = None
            return
        self.tfidf_matrix = self.tfidf.fit_transform(self.chunks)
        if embed_model is not None:
            self.embeddings = embed_model.encode(self.chunks, show_progress_bar=False).astype("float32")
        else:
            self.embeddings = None

    def hybrid_search(self, question: str, username: str, k: int = 4) -> List[str]:
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Indices accessible to this user (global docs + their own uploads)
        available = [
            i for i, m in enumerate(self.meta)
            if m["username"] is None or m["username"] == username
        ]
        if not available:
            return []

        # TF-IDF cosine scores
        q_tfidf = self.tfidf.transform([question])
        tfidf_scores = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]

        if embed_model is not None and self.embeddings is not None:
            # Semantic cosine scores
            q_emb = embed_model.encode([question]).astype("float32")
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
            emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            emb_normalized = self.embeddings / (emb_norms + 1e-10)
            sem_scores = (q_norm @ emb_normalized.T)[0]
            # Weighted hybrid score: 40% lexical + 60% semantic
            hybrid = 0.4 * tfidf_scores + 0.6 * sem_scores
        else:
            # Fallback to TF-IDF only
            hybrid = tfidf_scores

        ranked = sorted(available, key=lambda i: hybrid[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]


kb = KnowledgeBase()

# Load default sample.txt as global knowledge (doc_id=0, username=None)
if os.path.exists("sample.txt"):
    _default_chunks = [c.strip() for c in open("sample.txt").read().split("\n\n") if c.strip()]
    kb.chunks = _default_chunks
    kb.meta = [{"doc_id": 0, "username": None}] * len(_default_chunks)
    kb._rebuild()


# ─── JWT (stdlib HMAC-SHA256, no cryptography dependency) ──────────
def _b64url_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def make_token(username: str, role: str) -> str:
    header = _b64url_enc(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_enc(json.dumps({"sub": username, "role": role, "exp": int(time.time()) + 86400}).encode())
    sig = hmac.new(SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_enc(sig)}"

def decode_token(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed")
    header, payload, sig = parts
    expected = hmac.new(SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_dec(sig), expected):
        raise ValueError("bad signature")
    data = json.loads(_b64url_dec(payload))
    if data.get("exp", 0) < time.time():
        raise ValueError("expired")
    return data

# ─── Auth helpers ──────────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def verify_pw(plain: str, hashed: str) -> bool:
    return hash_pw(plain) == hashed

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        return decode_token(creds.credentials)
    except (ValueError, KeyError, json.JSONDecodeError):
        raise HTTPException(401, "Token không hợp lệ")


# ─── Database ──────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    return sqlite3.connect("api_users.db", check_same_thread=False, timeout=10)

def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT UNIQUE NOT NULL,
        password    TEXT NOT NULL,
        role        TEXT DEFAULT 'user',
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT NOT NULL,
        title       TEXT DEFAULT 'Cuộc hội thoại mới',
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT NOT NULL,
        session_id  INTEGER,
        question    TEXT NOT NULL,
        answer      TEXT NOT NULL,
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS documents (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        username     TEXT NOT NULL,
        filename     TEXT NOT NULL,
        chunks_count INTEGER DEFAULT 0,
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

init_db()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "placeholder"))


# ─── Pydantic models ────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[int] = None

class SessionCreateRequest(BaseModel):
    title: str = "Cuộc hội thoại mới"


# ─── Auth endpoints ────────────────────────────────────────────────
@app.post("/register", tags=["Auth"])
def register(req: RegisterRequest):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?,?,?)",
            (req.username, hash_pw(req.password), req.role),
        )
        conn.commit()
        conn.close()
        return {"message": "Đăng ký thành công!"}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username đã tồn tại")

@app.post("/login", tags=["Auth"])
def login(req: LoginRequest):
    conn = get_db()
    user = conn.execute(
        "SELECT username, password, role FROM users WHERE username=?", (req.username,)
    ).fetchone()
    conn.close()
    if not user or not verify_pw(req.password, user[1]):
        raise HTTPException(401, "Sai username hoặc password")
    return {"access_token": make_token(user[0], user[2]), "token_type": "bearer"}


# ─── Session endpoints ─────────────────────────────────────────────
@app.post("/sessions", tags=["Sessions"])
def create_session(req: SessionCreateRequest, user=Depends(get_current_user)):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO sessions (username, title) VALUES (?,?)", (user["sub"], req.title)
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return {"id": session_id, "title": req.title}

@app.get("/sessions", tags=["Sessions"])
def list_sessions(user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at FROM sessions WHERE username=? ORDER BY created_at DESC",
        (user["sub"],),
    ).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: int, user=Depends(get_current_user)):
    conn = get_db()
    conn.execute(
        "DELETE FROM chats WHERE session_id=? AND username=?", (session_id, user["sub"])
    )
    conn.execute(
        "DELETE FROM sessions WHERE id=? AND username=?", (session_id, user["sub"])
    )
    conn.commit()
    conn.close()
    return {"message": "Đã xóa session"}


# ─── RAG helpers ───────────────────────────────────────────────────
def build_messages(
    question: str, context: str, session_id: Optional[int], username: str
) -> List[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý AI thông minh. Trả lời dựa trên context được cung cấp. "
                "Nếu context không đủ thông tin, hãy trả lời dựa trên kiến thức của bạn "
                "và nói rõ điều đó. Trả lời bằng ngôn ngữ của câu hỏi."
            ),
        }
    ]
    # Inject last 5 turns of session history for multi-turn context
    if session_id:
        conn = get_db()
        history = conn.execute(
            "SELECT question, answer FROM chats WHERE session_id=? ORDER BY created_at DESC LIMIT 5",
            (session_id,),
        ).fetchall()
        conn.close()
        for q, a in reversed(history):
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nCâu hỏi: {question}"})
    return messages


# ─── Chat endpoints ────────────────────────────────────────────────
@app.post("/chat", tags=["Chat"])
def chat(req: ChatRequest, user=Depends(get_current_user)):
    context_chunks = kb.hybrid_search(req.question, user["sub"])
    context = "\n\n".join(context_chunks) if context_chunks else "Không có context."
    messages = build_messages(req.question, context, req.session_id, user["sub"])

    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=messages
    )
    answer = res.choices[0].message.content

    conn = get_db()
    conn.execute(
        "INSERT INTO chats (username, session_id, question, answer) VALUES (?,?,?,?)",
        (user["sub"], req.session_id, req.question, answer),
    )
    conn.commit()
    conn.close()
    return {"answer": answer, "user": user["sub"]}


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(req: ChatRequest, user=Depends(get_current_user)):
    """Server-Sent Events streaming: each token arrives as `data: {"token": "..."}\\n\\n`."""
    context_chunks = kb.hybrid_search(req.question, user["sub"])
    context = "\n\n".join(context_chunks) if context_chunks else "Không có context."
    messages = build_messages(req.question, context, req.session_id, user["sub"])
    username = user["sub"]
    session_id = req.session_id
    question = req.question

    def generate():
        accumulated: List[str] = []
        stream = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=messages, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                accumulated.append(delta)
                yield f"data: {json.dumps({'token': delta})}\n\n"

        answer = "".join(accumulated)
        conn = get_db()
        conn.execute(
            "INSERT INTO chats (username, session_id, question, answer) VALUES (?,?,?,?)",
            (username, session_id, question, answer),
        )
        conn.commit()
        conn.close()
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/history", tags=["Chat"])
def history(
    session_id: Optional[int] = Query(None),
    user=Depends(get_current_user),
):
    conn = get_db()
    if session_id is not None:
        rows = conn.execute(
            "SELECT question, answer, created_at FROM chats "
            "WHERE username=? AND session_id=? ORDER BY created_at ASC LIMIT 50",
            (user["sub"], session_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT question, answer, created_at FROM chats "
            "WHERE username=? ORDER BY created_at DESC LIMIT 10",
            (user["sub"],),
        ).fetchall()
    conn.close()
    return [{"question": r[0], "answer": r[1], "time": r[2]} for r in rows]


# ─── Document upload ───────────────────────────────────────────────
def extract_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    if filename.lower().endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise HTTPException(500, "pdfplumber chưa được cài. Chạy: pip install pdfplumber")
    raise HTTPException(400, "Chỉ hỗ trợ file .txt và .pdf")


def chunk_text(text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Slide overlap: keep tail of previous chunk
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + "\n\n" + para).strip() if tail else para
    if current:
        chunks.append(current)
    return chunks


@app.post("/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...), user=Depends(get_current_user)):
    content = await file.read()
    text = extract_text(file.filename, content)
    new_chunks = chunk_text(text)
    if not new_chunks:
        raise HTTPException(400, "Không trích xuất được nội dung từ file")

    conn = get_db()
    cur = conn.execute(
        "INSERT INTO documents (username, filename, chunks_count) VALUES (?,?,?)",
        (user["sub"], file.filename, len(new_chunks)),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()

    kb.add_chunks(new_chunks, doc_id, user["sub"])

    return {
        "message": "Upload thành công",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_count": len(new_chunks),
    }


@app.get("/documents", tags=["Documents"])
def list_documents(user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, chunks_count, created_at FROM documents WHERE username=? ORDER BY created_at DESC",
        (user["sub"],),
    ).fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "chunks_count": r[2], "created_at": r[3]} for r in rows]


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: int, user=Depends(get_current_user)):
    conn = get_db()
    doc = conn.execute("SELECT username FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc or doc[0] != user["sub"]:
        raise HTTPException(404, "Tài liệu không tồn tại hoặc không có quyền xóa")
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()
    kb.remove_chunks_by_doc(doc_id)
    return {"message": "Đã xóa tài liệu"}


# ─── Admin ─────────────────────────────────────────────────────────
@app.get("/users", tags=["Admin"])
def list_users(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Chỉ admin mới xem được")
    conn = get_db()
    rows = conn.execute("SELECT id, username, role, created_at FROM users").fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "role": r[2], "created_at": r[3]} for r in rows]


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "chunks_loaded": len(kb.chunks)}
