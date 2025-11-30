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

# ========== PRAGJET (IDENTIKE ME LOKALIN) ==========
GREEN_TH = 0.70
YELLOW_TH = 0.60
RED_TH = 0.60  # Eliminim fiks si në PC lokal

# ========== SERVER-SIDE CACHE NE CLOUD ==========
refine_cache = {}
embed_cache = {}

# ========== FUNKSIONE UTILE ==========

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

# ========== REFINE QUERY (GPT layer removed – identical logic, no random) ==========
def refine_query(user_input: str):
    key = user_input.strip().lower()
    if key in refine_cache:
        return refine_cache[key]

    # Pastrojmë inputin fiks si në PC
    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", user_input.lower()).strip()
    refined = cleaned  # Nuk ndryshojmë logjikë, bëjmë identik

    refine_cache[key] = (cleaned, refined)
    return cleaned, refined

# ========== EMBEDDING QUERY me CACHE (server-side cloud) ==========
def embed_query(text: str):
    key = text.lower()
    if key in embed_cache:
        return embed_cache[key]

    # Provojmë 3 herë embedding, por ruajmë në cache serveri
    for _ in range(3):
        try:
            r = client.embeddings.create(model="text-embedding-3-large", input=text)
            arr = np.array(r.data[0].embedding, dtype=np.float32)
            embed_cache[key] = arr
            return arr
        except:
            time.sleep(0.2)
    return None

# ========== GPT CHECK deterministik (përdorim argumentet me rend FIXED) ==========
def gpt_check(service_name, query):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Kthe vetëm: po / jo.'
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=3,
            seed=1234  # 👈 Fiksojmë seedin për stabilitet
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# ========== LOAD SERVICES 1x NGA CLOUDFARE R2 INTO RENDER CLOUD RAM ==========
print("⬇️ Po ngarkoj shërbimet nga R2 në RAM të serverit cloud...")

R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    endpoint_url=R2_BUCKET_URL,
    region_name="auto"
)

try:
    obj = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = obj["Body"].read().decode("utf-8")
    SERVICES = json.loads(raw)
    print(f"✅ U ngarkuan {len(SERVICES)} services nga R2 në cloud RAM")
except Exception as e:
    print("❌ Dështoi load nga R2:", str(e))
    SERVICES = []

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
def search_service(
    q: str = Query("", alias="q")
):
    """
    Search i stabilizuar: refine → embed me cache → cosine → filter >0.60 → sort → GPT check
    """
    t0 = time.time()

    # 1) Pastrojmë dhe refine query 1x cloud only
    cleaned, refined = refine_query(q)

    # 2) Marrim embedding me cache server-side në cloud
    qemb = embed_query(refined)
    if qemb is None:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    # 3) Cosine similarity + filtering identik
    scored = []
    for s in SERVICES:
        emb = to_arr(s.get("embedding_large"))
        if emb is None:
            continue
        sim_raw = cosine(qemb, emb)
        sim01 = scale01(sim_raw)
        if sim01 < RED_TH:
            continue
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 4) Seleksionojmë 4 më të mirat
    final = []
    for sim01, sim_raw, s in scored[:4]:
        final.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "score": round(sim01, 3),
            "uniqueid": s.get("uniqueid", ""),
            "category": s.get("category"),
            "keywords": s.get("keywords", [])
        })

    # 5) Pragjet Green/Yellow + GPT check deterministic
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    if greens:
        # Nëse ka green, i përdorim
        if len(final) < 3 and yellows:
            third = yellows[0]
            if gpt_check(third[2]["name"], refined):
                final.append({
                    "id": third[2]["id"],
                    "name": third[2]["name"],
                    "score": round(third[0], 3),
                    "uniqueid": third[2].get("uniqueid", ""),
                    "category": third[2].get("category"),
                    "keywords": third[2].get("keywords", [])
                })

    return {"query": q, "cleaned": cleaned, "refined": refined, "results": final, "time_sec": round(time.time() - t0, 2)}
