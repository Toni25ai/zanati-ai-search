import os, json, time, re
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI

# =========================
# KONFIGURIME
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
# UTILS
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
# 1) REFINE (IDENTIK ME LOKAL)
# =========================
refine_cache = {}

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
# 2) EMBEDDING
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
# 3) LOAD SERVICES
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
# 4) GPT CHECK
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
            max_tokens=3
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# =========================
# 5) SMART SEARCH
# =========================
def smart_search(user_query):
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
    for s in SERVICES:
        sim_raw = cosine(q_emb, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))

    scored.sort(
        key=lambda x: (round(x[0], 6), round(x[1], 6)),
        reverse=True
    )

    times["sim"] = time.time() - t

    greens  = [x for x in scored if x[0] >= GREEN_TH]
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

    final = [x for x in final if x[0] >= 0.60]
    times["total"] = time.time() - t0

    return final, times, cleaned, refined

# =========================
# TEMP STORAGE FOR GET (UI ONLY)
# =========================
LAST_RESULTS = {
    "results": [],
    "cleaned": "",
    "refined": "",
    "timings": {}
}

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
# SEARCH ENDPOINT (POST)
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

    # ruaj për GET
    LAST_RESULTS["results"] = out
    LAST_RESULTS["cleaned"] = cleaned
    LAST_RESULTS["refined"] = refined
    LAST_RESULTS["timings"] = times

    return {
        "results": out,
        "cleaned": cleaned,
        "refined": refined,
        "timings": times
    }

# =========================
# GET RESULTS (FOR BUBBLE)
# =========================
@app.get("/results")
async def get_results():
    return LAST_RESULTS
