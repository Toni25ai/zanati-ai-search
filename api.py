import os, time, re, json
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI

# ========== SUPABASE CONNECT ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OPENAI CONNECT ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== PRAGJET ==========
GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.50  # poshtë 0.50 eliminohen

app_api = FastAPI()

# ========== FUNKSIONE ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def gpt_check(service_name, query):
    pr = f'A është shërbimi "{service_name}" i përshtatshëm për "{query}"? Kthe vetëm: po/jo'
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": pr}],
            temperature=0, max_tokens=3
        )
        return r.choices[0].message.content.strip().lower() == "po"
    except:
        return False

# ========== ENDPOINT SEARCH ==========
@app_api.get("/search")
def search_service(q: str):
    t0 = time.time()

    # Pastrojmë queryn pa e ndryshuar kuptimin
    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", q.lower()).strip()
    refined = cleaned  # nuk e ndryshojmë më asnjëherë inputin

    # Marrim embedding_large të query
    rsp = client.embeddings.create(model="text-embedding-3-large", input=refined)
    qemb = np.array(rsp.data[0].embedding, dtype=np.float32)

    # Lexojmë embedding_large nga supabase column: "embedding_large"
    rows = supabase.from_("detailedtable").select("id,name,embedding_large,keywords,category").execute().data

    scored = []
    for r in rows:
        if r.get("embedding_large") is None:
            continue
        emb = np.array(r["embedding_large"], dtype=np.float32)
        sim = cosine(qemb, emb)
        sim01 = scale01(sim)

        # eliminojmë poshtë 0.5 sepse s’janë relevante
        if sim01 < RED_TH:
            continue

        # ruajmë vetëm ato që kanë shans ≥0.5
        scored.append((sim01, sim, r))

    # i rendisim sipas relevancës
    scored.sort(key=lambda x: x[0], reverse=True)

    # marrim max 4 shërbime
    final = []
    accepted = []
    greens = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    # 👉 CASE 1: Nese ka të paktën 1 GREEN
    if greens:
        for g in greens[:4]:
            accepted.append(g)
        # Nëse ka vetëm 1 ose 2 greens → bëjmë GPT-check vetem për një yellow
        if len(accepted) < 3 and yellows:
            third = yellows[0]
            if gpt_check(third[2]["name"], refined):
                accepted.append(third)

        accepted = accepted[:4]  # max 4
    else:
        # 👉 CASE 2: Nese nuk ka GREEN, por ka YELLOW
        for y in yellows[:2]:
            accepted.append(y)
        # Nese eshte i treti, e kontrollojmë me GPT
        if len(scored) >= 3:
            cand = scored[2]
            if YELLOW_TH <= cand[0] < GREEN_TH:
                if gpt_check(cand[2]["name"], refined):
                    accepted.append(cand)

    for sc, raw, s in accepted:
        final.append({
            "uniqueid": s["id"],  # lidhja me Bubble/Supabase bëhet me id
            "name": s["name"],
            "score": round(sc, 3)
        })

    t_total = time.time() - t0
    return {"results": final, "time_sec": round(t_total, 2)}

# ========= RUAJMË APP =========
app = app_api
