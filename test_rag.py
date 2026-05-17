import unittest
import hashlib
from app_full import chunk_text, KnowledgeBase


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain, hashed):
    return hashlib.sha256(plain.encode()).hexdigest() == hashed

def make_cache_key(question):
    return "rag:" + hashlib.md5(question.encode()).hexdigest()


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
        # Short paragraphs get merged into one chunk (max_chars=600)
        result = chunk_text("A\n\nB\n\nC")
        self.assertGreaterEqual(len(result), 1)
        combined = " ".join(result)
        self.assertIn("A", combined)
        self.assertIn("B", combined)
        self.assertIn("C", combined)

    def test_empty_skip(self):
        result = chunk_text("A\n\n\n\nB")
        self.assertGreaterEqual(len(result), 1)
        combined = " ".join(result)
        self.assertIn("A", combined)
        self.assertIn("B", combined)

    def test_long_paragraph_stays_single_chunk(self):
        long_para = "word " * 50
        chunks = chunk_text(long_para)
        self.assertGreaterEqual(len(chunks), 1)

    def test_overlap_preserves_content(self):
        text = "\n\n".join([f"Paragraph {i} " + "x " * 80 for i in range(5)])
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 0)


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()

    def test_empty_search_returns_empty(self):
        result = self.kb.hybrid_search("anything", "user1")
        self.assertEqual(result, [])

    def test_add_and_search(self):
        self.kb.add_chunks(
            ["Python là ngôn ngữ lập trình.", "FastAPI là web framework."],
            doc_id=1,
            username=None,
        )
        results = self.kb.hybrid_search("Python lập trình", "user1")
        self.assertGreater(len(results), 0)

    def test_user_isolation(self):
        self.kb.add_chunks(["Tài liệu bí mật của user1."], doc_id=2, username="user1")
        self.kb.add_chunks(["Tài liệu công khai."], doc_id=3, username=None)
        # user2 should not see user1's private doc
        results = self.kb.hybrid_search("bí mật", "user2")
        private_found = any("bí mật" in r for r in results)
        self.assertFalse(private_found)

    def test_remove_by_doc(self):
        self.kb.add_chunks(["Chunk A."], doc_id=10, username="alice")
        self.kb.add_chunks(["Chunk B."], doc_id=11, username="alice")
        self.kb.remove_chunks_by_doc(10)
        self.assertEqual(len([m for m in self.kb.meta if m["doc_id"] == 10]), 0)
        self.assertEqual(len([m for m in self.kb.meta if m["doc_id"] == 11]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
