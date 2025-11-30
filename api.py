import os, time, re, json, boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, Query
from supabase import create_client, Client
from openai import OpenAI

# ========== FASTAPI APP ==========
app = FastAPI()

# ========== SUPABASE CONNECT ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OPENAI CONNECT ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== PRAGJET (identike me lokal) ==========
GREEN_TH = 0.70
YELLOW_TH = 0.60
RED_TH = 0.60  # cutoff strict

# ========== CACHE NË SERVER (Render cloud RAM) ==========
refine_cache = {}
embed_cache = {}

# ========== FUNKSIONE ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def safe_list(v):
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]

def to_arr(x):
    if x is None:
        return None
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

def gpt_check(service_name, query):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Kthe vetëm: po / jo.'
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3,
            seed=1234
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# ========== LOAD SERVICES 1x NGA CLOUDFLARE R2 ==========
print("⬇️ Po ngarkoj shërbimet nga R2 në server RAM të Render...\n")

R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET,
    endpoint_url=R2_BUCKET_URL,
    region_name="auto"
)

try:
    obj = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = obj["Body"].read().decode("utf-8")
    ALL_SERVICES = json.loads(raw)
    print(f"✅ U ngarkuan {len(ALL_SERVICES)} shërbime nga R2 në RAM\n")
except Exception as e:
    print("❌ Dështoi ngarkimi nga R2:", str(e))
    ALL_SERVICES = []

# I kthejmë në format uniform si në lokal
SERVICES = []
for s in ALL_SERVICES:
    emb = to_arr(s.get("embedding_clean")) or to_arr(s.get("embedding_large"))
    if emb is None:
        continue
    SERVICES.append({
        "id": s.get("id"),
        "name": s.get("name"),
        "category": s.get("category"),
        "keywords": [k.lower() for k in safe_list(s.get("keywords", []))],
        "embedding": emb,
        "uniqueid": s.get("uniqueid","")
    })

print(f"🚀 Në backend u indeksuan {len(SERVICES)} embedding vectors\n")

# ========== ENDPOINTS ==========

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/columns")
def list_columns():
    sample = supabase.from_("detailedtable").select("*").limit(1).execute().data
    if not sample:
        return []
    return list(sample[0].keys())

@app.get("/search")
def search_service(q: str = Query("", alias="q")):
    """
    Endpoint 100% i qëndrueshëm:
    clean -> refine query 1x -> embed me cache -> cosine -> threshold strict -> sort -> GPT check stable
    """
    t0 = time.time()

    # 1) PASRTOJ query
    cleaned, refined = refine_query(q)

    # 2) MARRIM embedding cached cloud
    key = refined.lower()
    if key in embed_cache:
        qemb = embed_cache[key]
    else:
        qemb = None
        for _ in range(3):
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=refined)
                qemb = np.array(r.data[0].embedding, dtype=np.float32)
                embed_cache[key] = qemb
                break
            except:
                time.sleep(0.2)

    if qemb is None:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    # 3) Krahaso me services
    scored = []
    for s in SERVICES:
        sim_raw = cosine(qemb, s["embedding"])
        sim01 = scale01(sim_raw)
        if sim01 < RED_TH:
            continue
        scored.append((sim01, sim_raw, s))

    # 4) Sort deterministic
    scored.sort(key=lambda x: x[0], reverse=True)

    # 5) Marrim 4 më të mirat
    top4 = scored[:4]

    # 6) Filtrat GREEN/YELLOW + GPT check stable
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    final = []
    if greens:
        for sc01, sc, s in greens[:4]:
            final.append(s)
        if 1 <= len(final) < 3 and yellows:
            third = yellows[0]
            if gpt_check(third[2]["name"], refined):
                final.append(third[2])
    else:
        if yellows:
            chosen = yellows[:2]
            if len(yellows) >= 3:
                cand = yellows[2]
                if gpt_check(cand[2]["name"], refined):
                    chosen.append(cand)
            for sc01, sc, s in chosen:
                final.append(s)

    # 7) Build JSON për Bubble
    results = []
    for sim01, sim_raw, s in top4:
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "score": round(sim01, 3),
            "uniqueid": s["uniqueid"]
        })

    return {
        "query": q,
        "cleaned": cleaned,
        "refined": refined,
        "results": results,
        "time_sec": round(time.time() - t0, 2)
    }
