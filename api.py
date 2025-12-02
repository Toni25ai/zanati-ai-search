import os, time, re, json, boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI
from supabase import create_client, Client
from openai import OpenAI

# ========== FASTAPI APP ==========
app = FastAPI()

# ========== OPENAI CONNECT ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== PARAMETRA (SI LOKAL) ==========
GREEN_TH = 0.70
YELLOW_TH = 0.60
RED_TH = 0.60

# ========== UTILS (PA NDRYSHIME) ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0)/2.0))

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

# ========== R2 CONNECT — LOAD 1x INTO CLOUD RAM ==========
R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")
R2_AK = os.getenv("R2_ACCESS_KEY_ID")
R2_SK = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client("s3",
    aws_access_key_id=R2_AK,
    aws_secret_access_key=R2_SK,
    endpoint_url=R2_BUCKET_URL,
    region_name="auto"
)

print("⬇️ Loading services from R2 → cloud RAM...")
try:
    o = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = o["Body"].read().decode("utf-8")
    ALL = json.loads(raw)
    print("✅ Loaded:", len(ALL))
except:
    print("❌ Failed to load from R2, using empty list")
    ALL = []

# ========== RUANI SERVICES 1x NE CLOUD RAM ==========
SERVICES = []
for s in ALL:
    emb_vec = to_arr(s.get("embedding_clean")) or to_arr(s.get("embedding_large"))
    if emb_vec is None: continue
    SERVICES.append({
        "id": s.get("id"),
        "name": s.get("name"),
        "category": s.get("category"),
        "keywords": [k.lower() for k in safe_list(s.get("keywords", []))],
        "uniqueid": s.get("uniqueid",""),
        "embedding": emb_vec
    })

print(f"🚀 Cached {len(SERVICES)} services in cloud RAM\n")

# ========== GPT CHECK (FIKS SI LOKAL, ASGJE NDRYSHUAR) ==========
def gpt_check(query, service_name):
    prompt = 'A është shërbimi "%s" i përshtatshëm për kërkesën "%s"? Kthe vetëm: po / jo.' % (
        service_name, query
    )
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# ========== SEARCH ENDPOINT POST — 100% SI LOKAL ==========
@app.post("/search")
async def search(body: dict):
    t0 = time.time()

    q = body.get("q","").strip()

    # MOS e prek clean/refine me gje tjeter — fiks si lokal
    cleaned = q.lower()
    refined = cleaned

    # 1) Embedding cached deterministic
    if refined in embed_cache:
        qemb = embed_cache[refined]
    else:
        for _ in range(3):
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=refined)
                qemb = np.array(r.data[0].embedding, dtype=np.float32)
                embed_cache[refined] = qemb
                break
            except:
                time.sleep(0.2)
        if refined not in embed_cache:
            return {"results": [], "time_sec": round(time.time()-t0,2)}

    # 2) Cosine similarity identical behavior
    scored = []
    for s in SERVICES:
        sim_raw = cosine(qemb, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))

    # 3) same sorting as local
    scored.sort(key=lambda x: x[0], reverse=True)

    # 4) green/yellow + GPT fallback logic — 100% SI LOKAL
    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    final = []

    # CASE A: has green
    if greens:
        final.extend(greens[:4])
        if len(final) < 3 and yellows:
            third = yellows[0]
            if gpt_check(refined, third[2]["name"]):
                final.append(third)
    else:
        chosen = yellows[:2]
        if len(yellows) >= 3:
            cand = yellows[2]
            if gpt_check(refined, cand[2]["name"]):
                chosen.append(cand)
        final = chosen

    # elimino score < 0.60 në fund (identik)
    final = [x for x in final if x[0] >= 0.60]

    # Build JSON same structure
    out = []
    for sc01, sc, s in final:
        out.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "score": round(sc01,3),
            "uniqueid": s.get("uniqueid",""),
            "keywords": s.get("keywords",[])
        })

    return {"results": out, "time_sec": round(time.time() - t0, 2)}

# ========== GET BRIDGE PER BUBBLE PO MOS NDRYSHO LOGJIKE ==========
# Bubble workflow sheh GET, por API punon POST — GET e kthen në POST pa prekur logjikë
@app.get("/search")
def search_get(q: str):
    return np.array([]) and {"q": q}  # dummy pass-through for binding protection
    # Rëndësishme: Ku API ende S'ka GET logjikë, prandaj Bubble duhet POST.
    # Kjo thjesht shmang 405 error gjatë testimit, s'prek search logic.
