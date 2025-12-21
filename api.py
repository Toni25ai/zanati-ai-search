import os, json, time, re
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI

# =========================
# KONFIGURIME (IDENTIKE)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

SERVICES_CACHE = "/opt/render/project/data/services_cache.json"

EMBED_QUERY_MODEL = "text-embedding-3-large"
ONECALL_MODEL    = "gpt-4o-mini"
CHECK_MODEL      = "gpt-4o-mini"

GREEN_TH  = 0.70
YELLOW_TH = 0.60

os.makedirs("/opt/render/project/data", exist_ok=True)

app = FastAPI()

# =========================
# PROFESSION LIST (NEW - minimal)
# =========================
# Mbaje të shkurtër. Shto gradualisht sa të duash.
PROFESSIONS = {
    "marangoz",
    "hidraulik",
    "elektricist",
    "bojaxhi",
    "pllakaxhi",
    "murator",
    "saldator",
    "kondicionerist",
    "gipsxhi",
    "kopshtar",
    "pastrues",
}

# =========================
# UTILS (IDENTIKE)
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
# 1) REFINE (IDENTIK + shembuj) + FAST-PATH profesion
# =========================
refine_cache = {}

def refine_query(user_input: str):
    key = user_input.strip().lower()

    # ✅ NEW: Nëse është profesion i saktë (1 fjalë) mos thirr GPT fare
    # (kjo s'prek asnjë pjesë tjetër të logjikës)
    if len(key.split()) == 1 and key in PROFESSIONS:
        refine_cache[key] = (key, key)
        return key, key

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
- Nëse kërkesa është profesion: lejo "marangoz, druri", "kurs anglisht, arsim".
- Nëse ka problem: "riparim bojleri, hidraulik".
- Të jetë shumë inteligjent me dialekte.
- MOS përdor fjalë si: dua, duhet, kam nevojë, problemi është, ndihmë.

Shembuj:
"bojleri nuk ngroh" -> "riparim bojleri, hidraulik"
"sdi qysh bajne dy plus 2" -> "mësim matematike, arsim"
"me duhet marangoz" -> "marangoz, druri"

Kërkesa: "{user_input}"
"""

    for _ in range(3):
        try:
            rsp = client.chat.completions.create(
                model=ONECALL_MODEL,
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

# =========================
# 2) EMBEDDING (IDENTIK)
# =========================
embed_cache = {}

def embed_query(text: str):
    key = text.lower()
    if key in embed_cache:
        return embed_cache[key]

    for _ in range(3):
        try:
            r = client.embeddings.create(
                model=EMBED_QUERY_MODEL,
                input=text
            )
            arr = np.array(r.data[0].embedding, dtype=np.float32)
            embed_cache[key] = arr
            return arr
        except:
            time.sleep(0.3)

    return None

# =========================
# 3) LOAD SERVICES (IDENTIK)
# =========================
SERVICES = []

def load_services():
    global SERVICES

    if not os.path.exists(SERVICES_CACHE):
        SERVICES = []
        print("❌ services_cache.json NUK ekziston")
        return

    data = json.load(open(SERVICES_CACHE, "r", encoding="utf-8"))
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
                continue

        out.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "category": s.get("category"),
            "keywords": [k.lower() for k in safe_list(s.get("keywords", []))],
            "embedding": emb,
            "uniqueid": s.get("uniqueid")
        })

    SERVICES = out
    print(f"✅ U ngarkuan {len(SERVICES)} shërbime")

load_services()

# =========================
# 4) GPT CHECK (IDENTIK)
# =========================
def gpt_check(query, service_name):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për kërkesën "{query}"? Kthe vetëm: po / jo.'

    try:
        rsp = client.chat.completions.create(
            model=CHECK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# =========================
# 5) SMART SEARCH (IDENTIK) + conditional skip GPT check kur profesion i saktë
# =========================
def smart_search(user_query):
    times = {}
    t0 = time.time()

    t = time.time()
    cleaned, refined = refine_query(user_query)
    times["refine"] = time.time() - t

    # ✅ NEW: vetëm për skip GPT check (nuk prek renditje/pragje)
    is_clean_profession = (len(refined.split()) == 1 and refined in PROFESSIONS)

    t = time.time()
    q_emb = embed_query(refined)
    times["embed"] = time.time() - t
    if q_emb is None:
        return [], times, cleaned, refined

    t = time.time()
    scored = []
    for s in SERVICES:
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
            # ✅ NEW: nëse refined është profesion i saktë → skip GPT check
            if is_clean_profession or gpt_check(refined, third[2]["name"]):
                final.append(third)

    else:
        chosen = yellows[:2]
        if len(yellows) >= 3:
            cand = yellows[2]
            # ✅ NEW: nëse refined është profesion i saktë → skip GPT check
            if is_clean_profession or gpt_check(refined, cand[2]["name"]):
                chosen.append(cand)
        final = chosen

    # elimino poshtë 0.60 në fund (pa prekur renditjen)
    final = [x for x in final if x[0] >= 0.60]

    times["total"] = time.time() - t0
    return final, times, cleaned, refined

# =========================
# UPLOAD SERVICES
# =========================
@app.post("/upload_services")
async def upload_services(file: UploadFile = File(...)):
    raw = await file.read()
    with open(SERVICES_CACHE, "wb") as f:
        f.write(raw)
    load_services()
    return {"status": "ok", "count": len(SERVICES)}

# =========================
# SEARCH ENDPOINT
# =========================
@app.post("/search")
async def search(body: dict):
    query = body.get("q", "").strip()
    if not query:
        return {"results": []}

    results, times, cleaned, refined = smart_search(query)

    out = []
    for sim01, sim_raw, s in results:
        out.append({
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "score": round(sim01, 3),
            "cosine": round(sim_raw, 3),
            "uniqueid": s["uniqueid"],
            "keywords": s["keywords"]
        })

    return {
        "results": out,
        "cleaned": cleaned,
        "refined": refined,
        "timings": times
    }
