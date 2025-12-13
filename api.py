import os, json, time, re
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI

# =========================
# KONFIGURIME
# =========================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

GREEN_TH  = 0.70
YELLOW_TH = 0.60

# =========================
# RENDER DATA PATH
# =========================
DATA_FOLDER = "/opt/render/project/data"
os.makedirs(DATA_FOLDER, exist_ok=True)
DATA_PATH = f"{DATA_FOLDER}/services_cache.json"

# =========================
# FASTAPI
# =========================
app = FastAPI()

# =========================
# UTILS (IDENTIK)
# =========================
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
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

# =========================
# REFINE (IDENTIK v62)
# =========================
refine_cache = {}

def refine_query(user_input: str):
    key = user_input.strip().lower()
    if key in refine_cache:
        return refine_cache[key]

    prompt = f"""
Kthe vetëm JSON:
{{
 "cleaned": "<korrigjim i shkurtër>",
 "refined": "<etiketë 2-6 fjalë: veprim objekt, kategori>"
}}

RREGULLA:
- Pa pika. Pa fjali të gjata.
- Nëse profesion: "marangoz, druri"
- Nëse problem: "riparim bojleri, hidraulik"
- MOS përdor: dua, duhet, kam nevojë

Kërkesa: "{user_input}"
"""

    for _ in range(3):
        try:
            rsp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
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

    return user_input, user_input

# =========================
# EMBEDDING (CACHE)
# =========================
embed_cache = {}

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
            time.sleep(0.3)
    return None

# =========================
# GPT CHECK (IDENTIK)
# =========================
def gpt_check(query, service_name):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Kthe vetëm: po / jo.'
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=3
        )
        return rsp.choices[0].message.content.lower().startswith("p")
    except:
        return False

# =========================
# LOAD SERVICES
# =========================
SERVICES = []

def load_services():
    global SERVICES
    if not os.path.exists(DATA_PATH):
        SERVICES = []
        return

    data = json.load(open(DATA_PATH,"r",encoding="utf-8"))
    out = []

    for s in data:
        emb = to_arr(s.get("embedding_clean")) or to_arr(s.get("embedding_large"))
        if emb is None: continue

        out.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "category": s.get("category"),
            "keywords": [k.lower() for k in safe_list(s.get("keywords",[]))],
            "embedding": emb,
            "uniqueid": s.get("uniqueid","")
        })

    SERVICES = out
    print(f"✅ Loaded {len(SERVICES)} services")

load_services()

# =========================
# UPLOAD SERVICES
# =========================
@app.post("/upload_services")
async def upload_services(file: UploadFile = File(...)):
    raw = await file.read()
    with open(DATA_PATH,"wb") as f:
        f.write(raw)
    load_services()
    return {"status":"ok","count":len(SERVICES)}

# =========================
# SEARCH (FIXED)
# =========================
@app.post("/search")
async def search(body: dict):
    t0 = time.time()
    q = body.get("q","")

    cleaned, refined = refine_query(q)
    qemb = embed_query(refined)
    if qemb is None:
        return {"results":[],"uniqueids":[]}

    scored = []
    for s in SERVICES:
        sim_raw = cosine(qemb, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    final = []

    if greens:
        final.extend(greens[:4])
        if len(final) < 3 and yellows:
            if gpt_check(refined, yellows[0][2]["name"]):
                final.append(yellows[0])
    else:
        final = yellows[:2]
        if len(yellows) >= 3 and gpt_check(refined, yellows[2][2]["name"]):
            final.append(yellows[2])

    final = [x for x in final if x[0] >= 0.60]

    results = []
    uniqueids = []

    for sc01, sc, s in final:
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "score": round(sc01,3),
            "uniqueid": s["uniqueid"],
            "keywords": s["keywords"]
        })
        uniqueids.append(s["uniqueid"])

    return {
        "results": results,
        "uniqueids": uniqueids,
        "cleaned": cleaned,
        "refined": refined,
        "time_sec": round(time.time()-t0,2)
    }
