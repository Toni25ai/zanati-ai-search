import os, time, re, json, boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI
from supabase import create_client, Client
from openai import OpenAI

# ========== APP ==========
app = FastAPI()

# ========== SUPABASE CONNECT ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OPENAI CONNECT ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== PRAGJET (IDENTIKE ME PC) ==========
GREEN_TH = 0.70
YELLOW_TH = 0.60
RED_TH = 0.60  # 👈 FIKS si versione lokale

# ========== CACHING NE SERVER CLOUD (Render RAM) ==========
refine_cache = {}
embed_cache = {}

# ========== FUNKSIONET (IDENTIKE) ==========

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

# ========== REFINE QUERY DETERMINISTIK ==========
def refine_query(user_input: str):
    key = user_input.strip().lower()
    if key in refine_cache:
        return refine_cache[key]

    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", user_input.lower()).strip()
    refined = cleaned

    refine_cache[key] = (cleaned, refined)
    return cleaned, refined

# ========== EMBED QUERY ME CACHE NE CLOUD ==========
def embed_query(text: str):
    key = text.lower()
    if key in embed_cache:
        return embed_cache[key]

    for _ in range(3):
        try:
            r = client.embeddings.create(model="text-embedding-3-large", input=text)
            arr = np.array(r.data[0].embedding, dtype=np.float32)
            embed_cache[key] = arr
            return arr
        except:
            time.sleep(0.3)
    return None

# ========== GPT CHECK DETERMINISTIK 👇 FIX I RËNDËSISHËM ==========
def gpt_check(query, service_name):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Kthe vetëm: po / jo.'
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=3,
            seed=1234  # 👈 AI s’tolerohet nga restart i Render, por outputi LLM stabil
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# ========== LOAD SERVICES 1x NGA R2 NE RAM CLOUD ==========
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET,
    endpoint_url=os.getenv("R2_ENDPOINT_URL")
)

print("⬇️ Loading services from Cloudflare R2 into Render RAM...")
try:
    obj = s3.get_object(Bucket="servicescache", Key="detailed.json")
    raw = obj["Body"].read().decode("utf-8")
    SERVICES = json.loads(raw)
    print(f"✅ Loaded {len(SERVICES)} services from Cloudflare into Render RAM")
except Exception as e:
    print("❌ Failed to load services from R2:", str(e))
    SERVICES = []

# ========== SEARCH ENDPOINT (FIKS SI PC LOCAL) ==========
@app.get("/search")
def search_service(q: str = ""):
    t0 = time.time()

    cleaned, refined = refine_query(q)
    qemb = embed_query(refined)

    if qemb is None:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    scored = []
    for s in SERVICES:
        emb = to_arr(s.get("embedding_large"))
        if emb is None:
            continue

        sim_raw = cosine(qemb, emb)
        sim01 = scale01(sim_raw)
        if sim01 < 0.60:
            continue
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    if greens:
        for sim01, sim_raw, s in greens[:4]:
            final.append(s)
        if 1 <= len(final) < 3 and yellows:
            third = yellows[0]
            if gpt_check(refined, third[2]["name"]):
                final.append(third[2])
    else:
        chosen = yellows[:2]
        if len(yellows) >= 3:
            c = yellows[2]
            if gpt_check(refined, c[2]["name"]):
                chosen.append(c)
        for score, sim, s in chosen:
            final.append(s)

    final = [x for x in final if x[0] >= 0.60]

    results_json = []
    for sim01, sim_raw, s in scored[:4]:
        results_json.append({"id": s["id"], "name": s["name"], "score": round(sim01,3), "uniqueid": s.get("uniqueid","")})

    return {"query": q, "cleaned": cleaned, "refined": refined, "results": results_json, "time_sec": round(time.time() - t0, 2)}
