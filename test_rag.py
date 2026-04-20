import unittest
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain, hashed):
    return hashlib.sha256(plain.encode()).hexdigest() == hashed

def make_cache_key(question):
    return "rag:" + hashlib.md5(question.encode()).hexdigest()

def chunk_text(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

class TestAuth(unittest.TestCase):
    def test_hash_consistent(self):
        self.assertEqual(hash_password("abc"), hash_password("abc"))
    def test_hash_different(self):
        self.assertNotEqual(hash_password("abc"), hash_password("xyz"))
    def test_verify_correct(self):
        self.assertTrue(verify_password("pw", hash_password("pw")))
    def test_verify_wrong(self):
        self.assertFalse(verify_password("wrong", hash_password("pw")))

class TestCache(unittest.TestCase):
    def test_key_format(self):
        self.assertTrue(make_cache_key("Q").startswith("rag:"))
    def test_key_consistent(self):
        self.assertEqual(make_cache_key("Q"), make_cache_key("Q"))

class TestChunking(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(len(chunk_text("A\n\nB\n\nC")), 3)
    def test_empty_skip(self):
        self.assertEqual(len(chunk_text("A\n\n\n\nB")), 2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
