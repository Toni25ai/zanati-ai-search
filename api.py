import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI
import boto3

# ========== FASTAPI APP ==========
app = FastAPI()

# ========== SUPABASE CONNECT ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OPENAI CONNECT ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== PARAMETRA ==========
GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.50

# ========== FUNKSIONE ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def safe_list(v):
    if isinstance(v, list): return v
    if v is None: return []
    return [v]

def to_arr(x):
    if x is None: return None
    if isinstance(x, list):
        arr = np.array(x, dtype=np.float32)
        return arr if arr.size else None
    if isinstance(x, str):
        try:
            arr = np.array(json.loads(x), dtype=np.float32)
            return arr if arr.size else None
        except:
            nums = [float(n) for n in re.split(r"[,\s]+", x.strip("[] ")) if n]
            arr = np.array(nums, dtype=np.float32)
            return arr if arr.size else None
    return None

# ==========  LOAD JSON 1x NGA R2 + MBAJE NE RAM  ==========
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    endpoint_url="https://3a4c4d0d75c22ad3e96653008476f710.r2.cloudflarestorage.com"
)

print("⬇️ Loading JSON 1x from bucket into RAM...")
try:
    obj = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = obj["Body"].read().decode("utf-8")
    SERVICES = json.loads(raw)
    print(f"✅ Loaded {len(SERVICES)} services into RAM")
except Exception as e:
    print("❌ JSON failed to load:", str(e))
    SERVICES = []

# ========== ENDPOINTS ==========
@app.get("/health")
def health():
    return {"status": "ok", "time_sec": 0.0}

@app.get("/columns")
def list_columns():
    sample = supabase.from_("detailedtable").select("*").limit(1).execute().data
    if not sample:
        return []
    return list(sample[0].keys())

@app.get("/search")
def search_service(q: str):
    t0 = time.time()

    # 1) clean input
    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", q.lower()).strip()
    
    # 2) refine = identik
    refined = cleaned

    # 3) embed query
    try:
        r = client.embeddings.create(model="text-embedding-3-large", input=refined)
        qemb = np.array(r.data[0].embedding, dtype=np.float32)
    except:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    # 4) krahaso me 661 services në RAM (identike si në lokal)
    scored = []
    for s in SERVICES:
        e = s.get("embedding_large") or s.get("embedding_clean")
        emb = to_arr(e)
        if emb is None:
            continue

        sim_raw = cosine(qemb, emb)
        sim01 = scale01(sim_raw)

        if sim01 < RED_TH:
            continue

        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    for sim01, sim_raw, s in scored[:4]:
        final.append({
            "id": s["id"],
            "name": s["name"],
            "score": round(sim01, 3),
            "uniqueid": s.get("uniqueid", "")
        })

    # GPT check fiks si versioni yt
    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    if 1 <= len(final) < 3 and yellows:
        third = yellows[0]
        if gpt_check(third[2]["name"], refined):
            final.append({
                "id": third[2]["id"],
                "name": third[2]["name"],
                "score": round(third[0], 3),
                "uniqueid": third[2].get("uniqueid", "")
            })

    return {"results": final, "time_sec": round(time.time() - t0, 2)}
