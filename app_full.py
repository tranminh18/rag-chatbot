from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from groq import Groq
import sqlite3, hashlib, os
from datetime import datetime, timedelta
from jose import jwt, JWTError

app = FastAPI(title="RAG API with Auth")
security = HTTPBearer()

SECRET_KEY = "change-me-in-production"
ALGORITHM = "HS256"

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def verify_pw(plain, hashed): return hash_pw(plain) == hashed
def make_token(username, role):
    exp = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode({"sub": username, "role": role, "exp": exp}, SECRET_KEY, ALGORITHM)
def decode_token(token):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return decode_token(creds.credentials)
    except JWTError:
        raise HTTPException(401, "Token khong hop le")

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str

def get_db():
    return sqlite3.connect("api_users.db")

def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

init_db()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/register")
def register(req: RegisterRequest):
    try:
        conn = get_db()
        conn.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                     (req.username, hash_pw(req.password), req.role))
        conn.commit()
        conn.close()
        return {"message": "Dang ky thanh cong!"}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username da ton tai")

@app.post("/login")
def login(req: LoginRequest):
    conn = get_db()
    user = conn.execute("SELECT username,password,role FROM users WHERE username=?",
                        (req.username,)).fetchone()
    conn.close()
    if not user or not verify_pw(req.password, user[1]):
        raise HTTPException(401, "Sai username hoac password")
    return {"access_token": make_token(user[0], user[2]), "token_type": "bearer"}

@app.post("/chat")
def chat(req: ChatRequest, user=Depends(get_current_user)):
    q_vec = embed_model.encode([req.question]).tolist()[0]
    results = qdrant.query_points("docs", query=q_vec, limit=2).points
    context = "\n".join([r.payload["text"] for r in results])
    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":f"Context:\n{context}\n\nQ: {req.question}"}]
    )
    answer = res.choices[0].message.content
    conn = get_db()
    conn.execute("INSERT INTO chats (username,question,answer) VALUES (?,?,?)",
                 (user["sub"], req.question, answer))
    conn.commit()
    conn.close()
    return {"answer": answer, "user": user["sub"]}

@app.get("/history")
def history(user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("SELECT question,answer,created_at FROM chats WHERE username=? ORDER BY created_at DESC LIMIT 10",
                        (user["sub"],)).fetchall()
    conn.close()
    return [{"question": r[0], "answer": r[1], "time": r[2]} for r in rows]

@app.get("/users")
def list_users(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Chi admin moi xem duoc")
    conn = get_db()
    rows = conn.execute("SELECT id,username,role FROM users").fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "role": r[2]} for r in rows]
