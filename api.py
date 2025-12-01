import os, time, re, json, boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI
from supabase import create_client, Client
from openai import OpenAI

# ========== APP ==========
app = FastAPI()

# ========== Supabase ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OpenAI ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== PARAMETRA ==========
GREEN_TH = 0.70
YELLOW_TH = 0.60

# ========== CLOUD CACHES ==========
refine_cache = {}
embed_cache = {}

# ========== UTILS ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0)/2.0))

def safe_list(v):
    return v if isinstance(v,list) else [] if v is None else [v]

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
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Vetëm po/jo.'
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=3,
            seed=1234
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# ==== Load services 至 Render RAM (identik si lokal, nga R2) ====
R2_ENDPOINT = os.getenv("R2_BUCKET_URL")
R2_AK = os.getenv("R2_ACCESS_KEY_ID")
R2_SK = os.getenv("R2_SECRET_ACCESS_KEY")

# Cloudflare R2 S3 client
s3 = boto3.client("s3",
    aws_access_key_id=R2_AK,
    aws_secret_access_key=R2_SK,
    endpoint_url=R2_ENDPOINT,
    region_name="auto"
)

print("⬇️ Loading services from R2 → Render RAM...")
try:
    obj = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw_body = obj["Body"].read().decode("utf-8")
    services_cloud = json.loads(raw_body)
    print("✅ Loaded:", len(services_cloud))
except Exception as e:
    print("❌ Failed to load:", e)
    services_cloud = []

# Populate RAM list identike si lokal
RAM_SERVICES = []
for s in services_cloud:
    emb_watch1 = to_arr(s.get("embedding_clean"))
    emb_watch2 = to_arr(s.get("embedding_large"))

    if isinstance(emb_watch1, np.ndarray):
        vector = emb_watch1
    elif isinstance(emb_watch2, np.ndarray):
        vector = emb_watch2
    else:
        continue

    RAM_SERVICES.append({
        "id": s.get("id"),
        "name": s.get("name"),
        "category": s.get("category"),
        "keywords":[k for k in safe_list(s.get("keywords",[]))],
        "embedding": vector,
        "uniqueid": s.get("uniqueid","")
    })

print(f"🚀 Services in RAM:", len(RAM_SERVICES))

# ========== ENDPOINT /search (POST) ==========
@app.post("/search")
async def search_service(body: dict):

    user_query = body.get("q","")

    # 1) REFINE QUERY CACHED (fiks si lokal)
    key = user_query.strip().lower()
    if key in refine_cache:
        cleaned, refined = refine_cache[key]
    else:
        # IDENTIKE SI LOKAL (pa regex fallback)
        cleaned = user_query.strip()
        refined = cleaned
        refine_cache[key] = (cleaned, refined)

    # 2) EMBEDDING CACHED (fiks si lokal)
    ekey = refined.lower()
    if ekey in embed_cache:
        qemb = embed_cache[ekey]
    else:
        for _ in range(3):
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=refined)
                vector = np.array(r.data[0].embedding, dtype=np.float32)
                embed_cache[ekey] = vector
                qemb = vector
                break
            except:
                time.sleep(0.2)

    if 'qemb' not in locals():
        return {"results":[],"time_sec":round(time.time()-t0,2)}

    # 3) SIMILARITY (fiks si lokal)
    scored = []
    for s in RAM_SERVICES:
        sim_raw = cosine(qemb, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 4) GREEN / YELLOW + GPT CHECK (identik si lokal)
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]
    final = []

    if greens:
        for g in greens[:4]:
            final.append(g)
        if 1 <= len(final) < 3 and yellows:
            third = yellows[0]
            if gpt_check(third[2]["name"], refined):
                final.append(third)
    else:
        chosen = yellows[:2]
        if len(yellows) >= 3:
            cand = yellows[2]
            if gpt_check(cand[2]["name"], refined):
                chosen.append(cand)
        final = chosen

    final = [x for x in final if x[0] >= 0.60]

    # 5) BUILD RESULT JSON
    results=[]
    for sim01,cos,s in final:
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "score": round(sim01,3),
            "uniqueid": s["uniqueid"],
            "keywords": s.get("keywords",[])
        })

    return {
        "results": results[:TOP_N] if len(results)>=3 else results[:3],
        "time_sec": round(time.time()-t0,2),
        "cleaned": cleaned,
        "refined": refined
    }

@app.get("/columns")
def list_columns():
    s = supabase.table("detailedtable").select("*").limit(1).execute().data
    return [] if not s else list(s[0].keys())
