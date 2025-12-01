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
RED_TH = 0.60

# ========== CACHES (cloud RAM) ==========
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

def safe_list(v):
    return v if isinstance(v,list) else [] if v is None else [v]

def gpt_check(service_name, query):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Vetëm po/jo.'
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=3,
            seed=1234
        )
        ans = r.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# ========== LOAD SERVICES 1x NGA CLOUDFARE R2 ==========
R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")
R2_AK = os.getenv("R2_ACCESS_KEY_ID")
R2_SK = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client("s3",
    aws_access_key_id=R2_AK,
    aws_secret_access_key=R2_SK,
    endpoint_url=R2_BUCKET_URL,
    region_name="auto"
)

print("⬇️ Loading services from R2 → Render RAM…")
try:
    o = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = o["Body"].read().decode("utf-8")
    ALL = json.loads(raw)
    print("✅ Loaded:", len(ALL))
except Exception as e:
    print("❌ Load error:", e)
    ALL = []

SERVICES=[]
for s in ALL:
    e1 = to_arr(s.get("embedding_clean"))
    e2 = to_arr(s.get("embedding_large"))

    if isinstance(e1, np.ndarray):
        emb = e1
    elif isinstance(e2, np.ndarray):
        emb = e2
    else:
        continue

    SERVICES.append({
        "id": s.get("id"),
        "name": s.get("name"),
        "category": s.get("category"),
        "keywords":[k.lower() for k in safe_list(s.get("keywords",[]))],
        "embedding": emb,
        "uniqueid": s.get("uniqueid","")
    })

print(f"🚀 Indexed {len(SERVICES)} embeddings në cloud")

# ========== ENDPOINT /search — Bubble dynamic compatible ==========

@app.post("/search")
async def search_service(body: dict):
    t0 = time.time()

    # Merr query-n nga JSON body
    q = body.get("q", "")

    # 1) Clean + refine deterministic cached key
    key = q.strip().lower()
    if key in refine_cache:
        cleaned, refined = refine_cache[key]
    else:
        cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", q.lower()).strip()
        refined = cleaned
        refine_cache[key] = (cleaned, refined)

    # 2) Embedding cached cloud
    ekey = refined.lower()
    if ekey in embed_cache:
        qemb = embed_cache[ekey]
    else:
        qemb=None
        for _ in range(3):
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=refined)
                qemb=np.array(r.data[0].embedding,dtype=np.float32)
                embed_cache[ekey]=qemb
                break
            except:
                time.sleep(0.2)

    if qemb is None:
        return {"results":[],"time_sec":round(time.time()-t0,2)}

    # 3) Similarity + filter strict
    scored=[]
    for s in SERVICES:
        vec = s["embedding"]
        sim_raw=cosine(qemb,vec)
        sim01=scale01(sim_raw)
        if sim01<RED_TH: continue
        scored.append((sim01,sim_raw,s))
    scored.sort(key=lambda x:x[0],reverse=True)

    # 4) top 4
    top4=scored[:4]

    # 5) Green/Yellow logic deterministic + GPT check stable
    greens=[x for x in scored if x[0]>=GREEN_TH]
    yellows=[x for x in scored if YELLOW_TH<=x[0]<GREEN_TH]
    final=[]

    if greens:
        for sc01, sc, s in greens[:4]:
            final.append((sc01, sc, s))
        if 1<=len(final)<3 and yellows:
            third=yellows[0][2]
            if gpt_check(third["name"],refined):
                final.append((yellows[0][0], yellows[0][1], third))
    else:
        if yellows:
            chosen=yellows[:2]
            if len(yellows)>=3:
                cand=yellows[2][2]
                if gpt_check(cand["name"],refined):
                    chosen.append(cand)
            for sc01, sc, s in chosen:
                final.append((sc01,sc,s))

    final=[x for x in final if x[0]>=YELLOW_TH]

    # 6) Build JSON
    results=[]
    for sc01, sc, s in top4:
        if sc01<YELLOW_TH: continue
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "score": round(sc01,3),
            "uniqueid": s["uniqueid"],
            "keywords": s.get("keywords",[])
        })

    return {
        "results": results,
        "time_sec": round(time.time() - t0, 2),
        "cleaned": cleaned,
        "refined": refined
    }

@app.get("/columns")
def list_columns():
    s = supabase.table("detailedtable").select("*").limit(1).execute().data
    return [] if not s else list(s[0].keys())
