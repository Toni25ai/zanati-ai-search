import os, json, time, re
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI

# =========================
# CONFIG
# =========================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

GREEN_TH = 0.70
YELLOW_TH = 0.60

BASE_PATH = "/opt/render/project/data"
os.makedirs(BASE_PATH, exist_ok=True)
DATA_PATH = f"{BASE_PATH}/services_cache.json"

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
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def safe_list(v):
    return v if isinstance(v, list) else [] if v is None else [v]

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
            return None
    return None

# =========================
# GPT REFINE (IDENTIK ME LOKAL)
# =========================
refine_cache = {}

def refine_query(q):
    key = q.lower().strip()
    if key in refine_cache:
        return refine_cache[key]

    prompt = f"""
Kthe vetëm JSON:
{{ "cleaned": "...", "refined": "..." }}

RREGULLA:
- Pa fjali.
- Pa fjalë si: dua, duhet.
- Formë: "riparim bojleri, hidraulik"
- Inteligjent me dialekte.

Kërkesa: "{q}"
"""

    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80
            )
            txt = r.choices[0].message.content.strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```[a-zA-Z]*", "", txt).strip("` \n")
            data = json.loads(txt)

            cleaned = data.get("cleaned", q).strip()
            refined = data.get("refined", cleaned).strip()
            refined = refined.replace(".", "")
            refined = re.sub(r"\s+", " ", refined)

            refine_cache[key] = (cleaned, refined)
            return cleaned, refined
        except:
            time.sleep(0.2)

    return q, q

# =========================
# EMBEDDING CACHE
# =========================
embed_cache = {}

def embed_query(text):
    key = text.lower()
    if key in embed_cache:
        return embed_cache[key]

    r = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    arr = np.array(r.data[0].embedding, dtype=np.float32)
    embed_cache[key] = arr
    return arr

# =========================
# LOAD SERVICES
# =========================
SERVICES = []

def load_services():
    global SERVICES
    if not os.path.exists(DATA_PATH):
        print("❌ services_cache.json NUK ekziston")
        SERVICES = []
        return

    data = json.load(open(DATA_PATH, "r", encoding="utf-8"))
    out = []

    for s in data:
        emb = to_arr(s.get("embedding_clean")) or to_arr(s.get("embedding_large"))
        if emb is None:
            continue

        out.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "category": s.get("category"),
            "keywords": [k.lower() for k in safe_list(s.get("keywords"))],
            "embedding": emb,
            "uniqueid": s.get("uniqueid")
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
    with open(DATA_PATH, "wb") as f:
        f.write(raw)
    load_services()
    return {"status": "ok", "count": len(SERVICES)}

# =========================
# SEARCH (FIX RENDITJA)
# =========================
@app.post("/search")
async def search(body: dict):
    q = body.get("q", "").strip()
    t0 = time.time()

    cleaned, refined = refine_query(q)
    qemb = embed_query(refined)

    scored = []
    for s in SERVICES:
        sim_raw = cosine(qemb, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    top4 = [x for x in scored if x[0] >= YELLOW_TH][:4]

    results = []
    uniqueids = []

    for sc01, sc_raw, s in top4:
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "score": round(sc01, 3),
            "uniqueid": s["uniqueid"],
            "keywords": s["keywords"]
        })
        uniqueids.append(s["uniqueid"])

    return {
        "results": results,
        "uniqueids": uniqueids,
        "cleaned": cleaned,
        "refined": refined,
        "time_sec": round(time.time() - t0, 2)
    }
