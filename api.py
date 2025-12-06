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

# ========== CACHES ==========
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

# ========== GPT CHECK ==========
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

# ========== REFINE (IDENTIK ME PC v62 INTEL OPT) ==========
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
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80
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
        except:
            time.sleep(0.2)

    refine_cache[key] = (user_input, user_input)
    return user_input, user_input

# ========== LOAD SERVICES NGA R2 ==========
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
    obj = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = obj["Body"].read().decode("utf-8")
    ALL = json.loads(raw)
    print("✅ Loaded:", len(ALL))
except Exception as e:
    print("❌ Load error:", e)
    ALL = []

SERVICES = []
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

print(f"🚀 Cached {len(SERVICES)} services in RAM")

# ========== ENDPOINT SEARCH (IDENTIK ME PC) ==========
@app.post("/search")
async def search_service(body: dict):
    t0 = time.time()
    q = body.get("q", "")

    # REFINE — identik me lokal
    cleaned, refined = refine_query(q)

    # EMBED
    ekey = refined.lower()
    if ekey in embed_cache:
        qemb = embed_cache[ekey]
    else:
        qemb = None
        for _ in range(3):
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=refined)
                qemb = np.array(r.data[0].embedding, dtype=np.float32)
                embed_cache[ekey] = qemb
                break
            except:
                time.sleep(0.2)

    if qemb is None:
        return {"results": [], "uniqueids": [], "time_sec": round(time.time()-t0,2)}

    # SIMILARITY
    scored = []
    for s in SERVICES:
        sim_raw = cosine(qemb, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # GREEN/YELLOW RULE (identik)
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    final = []

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

    final = [x for x in final if x[0] >= YELLOW_TH]

    # BUILD RESULTS (identik me PC)
    results = []
    top4 = scored[:4]

    for sc01, sc, s in top4:
        if sc01 < YELLOW_TH: 
            continue
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "score": round(sc01, 3),
            "uniqueid": s["uniqueid"],
            "keywords": s.get("keywords",[])
        })

    # LISTA E UNIQUEID sipas renditjes
    uniqueids = [s["uniqueid"] for (_, _, s) in top4 if s["uniqueid"]]

    return {
        "results": results,
        "uniqueids": uniqueids,
        "time_sec": round(time.time() - t0, 2),
        "cleaned": cleaned,
        "refined": refined
    }


@app.get("/columns")
def list_columns():
    s = supabase.table("detailedtable").select("*").limit(1).execute().data
    return [] if not s else list(s[0].keys())

