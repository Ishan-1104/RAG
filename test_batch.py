from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Test 1
print(model.encode("Hello world"))

# Test 2
print(model.encode(["Hello world", "How are you?"]))