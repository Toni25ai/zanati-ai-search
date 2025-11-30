import os, time, re, json
import boto3
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

# ========== PRAGJET (SI NË PC LOCAL) ==========
GREEN_TH = 0.70
YELLOW_TH = 0.60
RED_TH = 0.60  # cutoff identical me lokal

# ========== CACHE NE RAM (Cloud Server-side, Render) ==========
refine_cache = {}
embed_cache = {}

# ========== UTILS ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b)/(na*nb))

def scale01(x):
    return max(0.0, min(1.0,(x+1.0)/2.0))

def to_arr(x):
    if x is None: return None
    if isinstance(x, list):
        a = np.array(x, dtype=np.float32)
        return a if a.size else None
    if isinstance(x, str):
        try:
            a = np.array(json.loads(x), dtype=np.float32)
            return a if a.size else None
        except:
            nums=[float(n) for n in re.split(r"[,\s]+", x.strip("[] ")) if n]
            a=np.array(nums, dtype=np.float32)
            return a if a.size else None
    return None

# ========== LOAD SERVICES 1x NGA R2 into Render RAM ==========
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT = R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY  # ensure exist

# 🚀 Endpoint BASE URL pa path:
R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")  # 👈 këtu duhet të jetë vetëm root endpoint

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    endpoint_url=R2_BUCKET_URL,   # ✅ tani e merr saktë
    region_name="auto"
)

print("⬇️ Loading shërbimet nga R2 në cloud RAM...")

try:
    obj = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = obj["Body"].read().decode("utf-8")
    SERVICES = json.loads(raw)
    print(f"✅ Loaded {len(SERVICES)} services në cloud RAM")
except Exception as e:
    print("❌ Failed to load services:", str(e))
    SERVICES = []

# ========== SEARCH ENDPOINT IDENTIK DETERMINISTIK ==========
@app.get("/search")
def search_service(
    q: str = Query("", alias="q")
):
    t0 = time.time()

    # 1) clean & refine (identik)
    key = q.strip().lower()
    if key in refine_cache:
        cleaned, refined = refine_cache[key]
    else:
        cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+","",q.lower()).strip()
        refined = cleaned
        refine_cache[key]=(cleaned, refined)

    # 2) embed me cache
    qemb = None
    ekey = refined.lower()
    if ekey in embed_cache:
        qemb = embed_cache[ekey]
    else:
        for _ in range(3):
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=refined)
                qemb = np.array(r.data[0].embedding, dtype=np.float32)
                embed_cache[ekey]=qemb
                break
            except:
                time.sleep(0.2)

    if qemb is None:
        return {"results": [], "time_sec": round(time.time()-t0,2)}

    # 3) similarity identical + filtering
    scored=[]
    for s in SERVICES:
        vec = to_arr(s.get("embedding_large") or s.get("embedding_clean"))
        if vec is None: continue
        sim_raw=cosine(qemb,vec)
        sim01=scale01(sim_raw)
        if sim01 < RED_TH: continue
        scored.append((sim01,sim_raw,s))

    scored.sort(key=lambda x:x[0], reverse=True)

    # 4) top4 deterministic pa random
    final=[]
    for sc01, sc, s in scored[:4]:
        final.append(
            {
              "id": s.get("id"),
              "name": s.get("name"),
              "score": round(sc01,3),
              "uniqueid": s.get("uniqueid",""),
              "category": s.get("category"),
              "keywords": s.get("keywords",[])
            }
        )

    # 5) GPT check deterministic me seed fixed
    CHECK_MODEL="gpt-4o-mini"
    for sc01, sc, s in yellows[:1]:
        try:
            prompt=f'A është shërbimi "{s["name"]}" i përshtatshëm për kërkesën "{refined}"? Vetëm po/jo'
            rsp = client.chat.completions.create(
                model=CHECK_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=3,
                seed=1234
            )
            ans=rsp.choices[0].message.content.strip().lower()
            if ans.startswith("p"):
                final.append({"id":s["id"],"name":s["name"], "score":round(sc01,3),"uniqueid":s.get("uniqueid","")})
        except:
            pass

    return {"query":q,"cleaned":cleaned,"refined":refined,"results":final,"time_sec":round(time.time()-t0,2)}
