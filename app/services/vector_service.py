# app/services/vector_service.py
from chromadb import PersistentClient
from openai import AsyncOpenAI
import tiktoken, os
from dotenv import load_dotenv

load_dotenv() 

EMBED_MODEL  = "text-embedding-3-small"   # cheap & solid
CHROMA_PATH  = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION   = "doc_type_index"

client       = PersistentClient(path=CHROMA_PATH)
collection   = client.get_or_create_collection(COLLECTION)
oai          = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))              # uses OPENAI_API_KEY env var

async def embed(text: str) -> list[float]:
    enc = tiktoken.get_encoding("cl100k_base")
    text = text[:8192]                    # hard cap for speed/cost
    resp = await oai.embeddings.create(
        model=EMBED_MODEL,
        input=text.replace("\n", " ")
    )
    return resp.data[0].embedding

async def index_examples(examples: list[dict]):
    """
    examples = [
        {"id":"inv-001", "text": "...full text...", "label":"invoice"},
        ...
    ]
    """
    embeddings = [await embed(ex["text"]) for ex in examples]
    collection.add(
        ids   =[ex["id"]    for ex in examples],
        embeddings=embeddings,
        metadatas =[{"label": ex["label"]} for ex in examples]
    )

async def classify(text: str, k: int = 3) -> tuple[str, float]:
    """
    Returns (predicted_label, confidence 0-1)
    """
    vec   = await embed(text)
    res   = collection.query(query_embeddings=[vec], n_results=k)
    hits  = res['metadatas'][0]           # list[{'label':...}, ...]
    sims  = res['distances'][0]           # cosine distances (0=identical)

    # majority vote weighted by (1-distance)
    weights = {}
    for hit, dist in zip(hits, sims):
        w = 1 - dist
        lbl = hit["label"]
        weights[lbl] = weights.get(lbl, 0) + w
    best = max(weights, key=weights.get)
    conf = weights[best] / sum(weights.values())
    return best, conf
