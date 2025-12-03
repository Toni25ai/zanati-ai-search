import os, time, re, json, boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client, Client

# =========================
# KONFIGURIME
# =========================

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_QUERY_MODEL = "text-embedding-3-large"
ONECALL_MODEL    = "gpt-4o-mini"
CHECK_MODEL      = "gpt-4o-mini"

GREEN_TH  = 0.70
YELLOW_TH = 0.60

client = OpenAI(api_key=OPENAI_API_KEY)

# Supabase (vetëm për /columns, s’prek search-in)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# R2 – ku ke JSON-in me shërbimet
R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")
R2_AK = os.getenv("R2_ACCESS_KEY_ID")
R2_SK = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = "servicescache"
R2_KEY = "services_cache_v7_clean.json"

# FastAPI app
app = FastAPI()

# =========================
# UTILS – IDENTIKE ME LOKAL
# =========================

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

# =========================
# 1) REFINE INTELIGJENT (me CACHE) – IDENTIK
# =========================

refine_cache: dict[str, tuple[str,str]] = {}

def refine_query(user_input: str):
    key = user_input.strip().lower()
    if key in refine_cache:
        return refine_cache[key]

    prompt = """
Kthe vetëm JSON:
{
 "cleaned": "<korrigjim i shkurtër>",
 "refined": "<etiketë 2-6 fjalë: veprim objekt, kategori>"
}

RREGULLA:
- Pa pika. Pa fjali të gjata.
- Nëse kërkesa është profesion: lejo "marangoz, druri", "kurs anglisht, arsim".
- Nëse ka problem: "riparim bojleri, hidraulik".
- Të jetë shumë inteligjent me dialekte.
- MOS përdor fjalë si: dua, duhet, kam nevojë, problemi është, ndihmë.

Shembuj:
"bojleri nuk ngroh" -> "riparim bojleri, hidraulik"
"sdi qysh bajne dy plus 2" -> "mësim matematike, arsim"
"me duhet marangoz" -> "marangoz, druri"

Kërkesa: "%s"
""" % user_input

    for _ in range(3):
        try:
            rsp = client.chat.completions.create(
                model=ONECALL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )

            txt = rsp.choices[0].message.content.strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```[a-zA-Z]*", "", txt).strip("` \n")

            data = json.loads(txt)

            cleaned = data.get("cleaned", user_input).strip()
            refined = data.get("refined", cleaned).strip()
            refined = refined.replace(".", "")
            refined = re.sub(r"\s+", " ", refined)

            refine_cache[key] = (cleaned, refined)
            return cleaned, refined
        except Exception:
            time.sleep(0.2)

    refine_cache[key] = (user_input, user_input)
    return user_input, user_input

# =========================
# 2) EMBEDDING (me CACHE) – IDENTIK
# =========================

embed_cache: dict[str, np.ndarray] = {}

def embed_query(text: str):
    key = text.lower()
    if key in embed_cache:
        return embed_cache[key]
    for _ in range(3):
        try:
            r = client.embeddings.create(model=EMBED_QUERY_MODEL, input=text)
            arr = np.array(r.data[0].embedding, dtype=np.float32)
            embed_cache[key] = arr
            return arr
        except Exception:
            time.sleep(0.3)
    return None

# =========================
# 3) GPT CHECK – IDENTIK
# =========================

def gpt_check(query, service_name):
    prompt = 'A është shërbimi "%s" i përshtatshëm për kërkesën "%s"? Kthe vetëm: po / jo.' % (
        service_name, query
    )

    try:
        rsp = client.chat.completions.create(
            model=CHECK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3,
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except Exception:
        return False

# =========================
# 4) LOAD SERVICES NGA R2 – E NJEJTA LOGJIKË SI load_services()
# =========================

def load_services_from_r2():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=R2_AK,
        aws_secret_access_key=R2_SK,
        endpoint_url=R2_BUCKET_URL,
        region_name="auto",
    )

    print("⬇️ Loading services from R2 → Render RAM…")
    try:
        o = s3.get_object(Bucket=R2_BUCKET_NAME, Key=R2_KEY)
        raw = o["Body"].read().decode("utf-8")
        data = json.loads(raw)
        print("✅ Loaded:", len(data))
    except Exception as e:
        print("❌ Load error from R2:", e)
        data = []

    out = []
    for s in data:
        emb_clean = to_arr(s.get("embedding_clean"))
        if emb_clean is not None:
            emb = emb_clean
        else:
            emb_large = to_arr(s.get("embedding_large"))
            if emb_large is not None:
                emb = emb_large
            else:
                continue  # skip

        out.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "category": s.get("category"),
            "keywords": [k.lower() for k in safe_list(s.get("keywords", []))],
            "embedding": emb,
            "uniqueid": s.get("uniqueid", ""),
        })
    print(f"🚀 Cached {len(out)} services in RAM")
    return out

SERVICES = load_services_from_r2()

# =========================
# 5) SEARCH LOGJIKA – IDENTIKE SMART_SEARCH
# =========================

def smart_search(user_query, services):
    times = {}
    t0 = time.time()

    t = time.time()
    cleaned, refined = refine_query(user_query)
    times["refine"] = time.time() - t

    t = time.time()
    q_emb = embed_query(refined)
    times["embed"] = time.time() - t
    if q_emb is None:
        return [], times, cleaned, refined

    t = time.time()
    scored = []
    for s in services:
        sim_raw = cosine(q_emb, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    times["sim"] = time.time() - t

    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    final = []

    # CASE A: ka green
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

    # elimino poshtë 0.60 në fund (pa prekur renditjen)
    final = [x for x in final if x[0] >= 0.60]

    times["total"] = time.time() - t0
    return final, times, cleaned, refined

# =========================
# 6) FASTAPI MODELS & ENDPOINTS
# =========================

class SearchBody(BaseModel):
    q: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search_service(body: SearchBody):
    t0 = time.time()
    q = body.q or ""
    results, times, cleaned, refined = smart_search(q, SERVICES)

    api_results = []
    for sim01, sim_raw, s in results:
        api_results.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "category": s.get("category"),
            "score": round(float(sim01), 3),
            "uniqueid": s.get("uniqueid", ""),
            "keywords": s.get("keywords", []),
        })

    return {
        "results": api_results,
        "time_sec": round(time.time() - t0, 2),
        "cleaned": cleaned,
        "refined": refined,
    }

@app.get("/columns")
def list_columns():
    if supabase is None:
        return []
    s = supabase.table("detailedtable").select("*").limit(1).execute().data
    return [] if not s else list(s[0].keys())
