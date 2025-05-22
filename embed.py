# embed.py
from sentence_transformers import SentenceTransformer
import sys
import json

# Load the embedding model
model = SentenceTransformer("nateraw/llama2-embed")  # This model is compatible with LLaMA-style embeddings

# Read input from Replicate (stdin)
input_data = json.load(sys.stdin)
text = input_data["text"]

# Generate the embedding
embedding = model.encode(text).tolist()

# Output the embedding as JSON to stdout
print(json.dumps(embedding))
