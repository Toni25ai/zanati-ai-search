import os, time, re, json
import numpy as np
from numpy.linalg import norm
from supabase import create_client, Client
from openai import OpenAI
from fastapi import FastAPI

# ========== FASTAPI ==========
app_api = FastAPI()
app = app_api   # mos e prek, ruaj identik

# ========== SUPABASE CONNECT ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OPENAI CONNECT ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== PARAMETRA ==========
GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.50

# ========== FUNKSIONE ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def gpt_check(service_name, query):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për "{query}"? Kthe vetëm: po/jo'
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=3
        )
        return r.choices[0].message.content.strip().lower() == "po"
    except:
        return False

# ========== ENDPOINTS ==========
@app_api.get("/health")
def health():
    return {"status": "ok", "time_sec": 0.0}

@app_api.get("/columns")
def list_columns():
    sample = supabase.from_("detailedtable").select("*").limit(1).execute().data
    if not sample:
        return []
    return list(sample[0].keys())

@app_api.get("/search")
def search_service(q: str):
    t0 = time.time()

    # 1) clean input
    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", q.lower()).strip()
    
    # 2) refine = saktësisht cleaned
    refined = cleaned  # identik me logjikën tënde

    # 3) embed query me large
    try:
        rsp = client.embeddings.create(model="text-embedding-3-large", input=refined)
        qemb = np.array(rsp.data[0].embedding, dtype=np.float32)
    except:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    # 4) load service rows nga supabase me ID të përbashkët
    try:
        rows = supabase.from_("detailedtable").select("id,name,embedding_large,keywords,uniqueid,category").execute().data
    except:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    scored = []
    for r in rows:
        e = r.get("embedding_large")
        emb = None

        if isinstance(e, list):
            emb = np.array(e, dtype=np.float32)
        elif isinstance(e, str):
            try:
                arr = json.loads(e)
                if isinstance(arr, list) and len(arr) > 1:
                    emb = np.array(arr, dtype=np.float32)
            except:
                continue

        if emb is None:
            continue

        sim_raw = cosine(qemb, emb)
        sim01 = scale01(sim_raw)

        if sim01 < RED_TH:
            continue

        scored.append((sim01, sim_raw, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    for sim01, sim_raw, r in scored[:4]:
        final.append({
            "id": r["id"],
            "name": r["name"],
            "score": round(sim01, 3),
            "uniqueid": r.get("uniqueid") or r.get("uniqueid", "")
        })

    # yellow chance me GPT-check vetëm nëse 1–2 green
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    if 1 <= len(final) < 3 and yellows:
        y = yellows[0]
        if gpt_check(r["name"], refined):
            final.append({
                "id": y[2]["id"],
                "name": y[2]["name"],
                "score": round(y[0], 3),
                "uniqueid": y[2].get("uniqueid", "")
            })

    t_total = time.time() - t0
    return {"results": final, "time_sec": round(t_total, 2)}
