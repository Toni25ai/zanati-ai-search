import os, time, re, json, boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI
from openai import OpenAI

# ========== FASTAPI ==========
app = FastAPI()

# ========== OPENAI ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== THRESHOLDS ==========
GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.60

# ========== CACHES ==========
refine_cache = {}
embed_cache  = {}

# =========================
# UTILS
# =========================
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def safe_list(v):
    if isinstance(v, list): return v
    if v is None: return []
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

# =========================
# LOAD FROM R2 (fast, 1x)
# =========================
R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")
R2_AK = os.getenv("R2_ACCESS_KEY_ID")
R2_SK = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_AK,
    aws_secret_access_key=R2_SK,
    endpoint_url=R2_BUCKET_URL,
    region_name="auto"
)

print("⬇️ Loading services from R2 → Render RAM…")
try:
    obj = s3.get_object(
        Bucket="servicescache",
        Key="services_cache_v7_clean.json"
    )
    raw = obj["Body"].read().decode("utf-8")
    ALL = json.loads(raw)
    print("✅ Loaded:", len(ALL))
except Exception as e:
    print("❌ Load error:", e)
    ALL = []

# =========================
# BUILD SERVICES EXACT LIKE LOCAL
# =========================
SERVICES = []

for s in ALL:
    emb1 = to_arr(s.get("embedding_clean"))
    emb2 = to_arr(s.get("embedding_large"))

    # -------- FIX I DETYRUESHËM --------
    # Nuk përdorim OR (e shkaktonte errorin)
    if isinstance(emb1, np.ndarray):
        emb = emb1
    elif isinstance(emb2, np.ndarray):
        emb = emb2
    else:
        continue
    # -----------------------------------

    SERVICES.append({
        "id": s.get("id"),
        "name": s.get("name"),
        "category": s.get("category"),
        "keywords": [k.lower() for k in safe_list(s.get("keywords", []))],
        "uniqueid": s.get("uniqueid", ""),
        "embedding": emb
    })

print(f"🚀 Cached {len(SERVICES)} services in RAM")


# =========================
# GPT CHECK (identik me lokal)
# =========================
def gpt_check(query, service_name):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Kthe vetëm: po / jo.'

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3
        )
        ans = r.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False


# =========================
# EMBEDDING CACHE EXACT LIKE LOCAL
# =========================
def embed_query(text: str):
    key = text.lower()

    if key in embed_cache:
        return embed_cache[key]

    for _ in range(3):
        try:
            r = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            arr = np.array(r.data[0].embedding, dtype=np.float32)
            embed_cache[key] = arr
            return arr
        except:
            time.sleep(0.2)

    return None


# ======================================================
# ===============   SEARCH ENDPOINT   ==================
# ======================================================
@app.post("/search")
async def search_service(body: dict):
    t0 = time.time()

    q = body.get("q", "").strip()

    # 1) Clean IDENTIK me lokal
    cleaned = q.lower()
    refined = cleaned

    # 2) Embedding IDENTIK
    qemb = embed_query(refined)
    if qemb is None:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    # 3) Similarity EXACT
    scored = []
    for s in SERVICES:
        sim_raw = cosine(qemb, s["embedding"])
        sim01 = scale01(sim_raw)
        if sim01 < RED_TH: 
            continue
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    final = []

    # 100% same logic as PC:
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

    final = [x for x in final if x[0] >= 0.60]

    results = [{
        "id": s["id"],
        "name": s["name"],
        "category": s.get("category", ""),
        "score": round(sc01, 3),
        "uniqueid": s["uniqueid"],
        "keywords": s.get("keywords", [])
    } for sc01, sc, s in final[:4]]

    return {
        "results": results,
        "time_sec": round(time.time()-t0,2),
        "cleaned": cleaned,
        "refined": refined
    }


@app.get("/health")
def health():
    return {"status": "ok"}
