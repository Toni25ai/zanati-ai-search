import os, time, re, json
import boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, Query
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

# ========== Thresholds fixed (si në local) ==========
GREEN_TH = 0.70
YELLOW_TH = 0.60
RED_TH = 0.60

# ========== Cache server-side (te cloud, jo PC) ==========
refine_cache = {}
embed_cache = {}

# ========== Utils ==========
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

# ========== Refine (identik, deterministic, cached në cloud server) ==========
def refine_query(user_input: str):
    key = user_input.strip().lower()
    if key in refine_cache:
        return refine_cache[key]

    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", user_input.lower()).strip()
    refined = cleaned
    refine_cache[key] = (cleaned, refined)
    return cleaned, refined

# ========== Embeddings deterministic (cache server-side) ==========
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
            time.sleep(0.2)
    return None

# ========== GPT Check stable, deterministic me seed fixed ==========
def gpt_check(query, service_name):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Kthe vetëm: po / jo.'
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

# ========== Load services 1x nga Cloudflare R2 → qëndron vetëm në cloud RAM, siç ti do ==========
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

print("⬇️ Po ngarkoj shërbimet nga R2 në cloud RAM...")
try:
    obj = s3.get_object(Bucket="servicescache", Key="servicescache/services_cache_v7_clean.json")
    raw = obj["Body"].read().decode("utf-8")
    SERVICES = json.loads(raw)
    print(f"✅ U ngarkuan {len(SERVICES)} shërbime në RAM nga R2")
except Exception as e:
    print("❌ Dështoi load:", str(e))
    SERVICES = []

# ========== Endpoint /search stable ==========
@app.get("/search")
def search_service(q: str = Query("", alias="q")):
    t0 = time.time()

    # 1) pastrojmë + refine 1x
    cleaned, refined = refine_query(q)

    # 2) embedding nga cache server-side
    qemb = embed_query(refined)
    if qemb is None:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    # 3) krahasojmë me datasetin `SERVICES` në cloud RAM
    scored = []
    for s in SERVICES:
        vec = to_arr(s.get("embedding_clean") or s.get("embedding_large"))
        if vec is None:
            continue
        sim_raw = cosine(qemb, vec)
        sim01 = scale01(sim_raw)
        if sim01 < RED_TH:
            continue
        scored.append((sim01, sim_raw, s))

    # 4) sort deterministic
    scored.sort(key=lambda x: x[0], reverse=True)

    # 5) krijojmë listën finale pa random
    final = []
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    if greens:
        # marrim 4 të parat nga greens
        for sc01, sim, s in greens[:4]:
            final.append(s)
        # plotësojmë me yellows nëse duhen 3 rezultate
        if 1 <= len(final) < 3 and yellows:
            third = yellows[0][2]
            if gpt_check(refined, third["name"]):
                final.append(third)
    elif yellows:
        # marrim 2-3 yellows deterministic me GPT check
        chosen = yellows[:2]
        if len(yellows) >= 3:
            cand = yellows[2][2]
            if gpt_check(refined, cand["name"]):
                chosen.append((yellows[2][0], yellows[2][1], yellows[2][2]))
        for sc01, sim, s in chosen:
            final.append(s)

    # 6) Heqim çdo rezultat nën 0.60 (*identik me versionin lokal*)
    final_scored = []
    for sc01, sim, s in scored[:4]:
        if sc01 >= 0.60:
            final_scored.append((sc01, s))

    final_json = []
    for sc01, s in final_scored:
        final_json.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "score": round(sc01, 3),
            "uniqueid": s.get("uniqueid", ""),
            "category": s.get("category"),
            "keywords": s.get("keywords",[])
        })

    return {"query": q, "cleaned": cleaned, "refined": refined, "results": final_json, "time_sec": round(time.time() - t0, 2)}

@app.get("/columns")
def list_columns():
    sample = supabase.from_("detailedtable").select("*").limit(1).execute().data
    if not sample:
        return []
    return list(sample[0].keys())
