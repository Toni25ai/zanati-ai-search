import os, time, re, json
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI, Query

# ========== Lidhje API ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not openai_key:
    raise SystemExit("❌ KE HARUAR Environment Variables në Render!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=openai_key)

app = FastAPI()

# ========== CACHING NË RAM (Cloud-friendly, por jo lokal!) ==========
refine_cache = {}
embedding_cache = {}

# ========== Utils ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a, b)/(na*nb))

def scale01(x): return max(0.0, min(1.0,(x+1.0)/2.0))

def to_arr(x):
    if x is None: return None
    if isinstance(x, list):
        arr = np.array(x, dtype=np.float32).flatten()
        return arr if arr.size else None
    if isinstance(x, str):
        try:
            arr = np.array(json.loads(x), dtype=np.float32).flatten()
            return arr if arr.size else None
        except:
            nums = [float(n) for n in re.split(r"[,\s]+", x.strip("[] ")) if n]
            arr = np.array(nums, dtype=np.float32).flatten()
            return arr if arr.size else None
    return None

# ========== Refine inteligjent super i shkurtër ==========
def refine_query(q: str):
    key = q.lower().strip()
    if key in refine_cache:
        return refine_cache[key]

    prompt = f"""
Kthe vetëm JSON:
{{"cleaned":"{q}","refined":"{q}"}}

Mos ndrysho asgjë në kuptim, vetëm pastro nëse duhet dhe mbaje dyfjalësh 'veprim objekt, kategori' kur ka veprim.
Për profesion pa veprim: "hidraulik banjo, ndërtim"
Për probleme: "riparim bojleri, hidraulik"
Për kurse: "kurs matematike, arsim"
Kërkesa: "{q}"
"""
    try:
        rsp = client.chat.completions.create(
            model=ONECALL_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=35
        )
        data = json.loads(rsp.choices[0].message.content.strip())
        cleaned = data["cleaned"]
        refined = data["refined"]
    except:
        cleaned, refined = q, q

    refine_cache[key] = (cleaned, refined)
    return cleaned, refined

# ========== Embedding Cloud Supabase ==========
def embed_query(text: str):
    key = text.lower().strip()
    if key in embedding_cache:
        return embedding_cache[key]

    try:
        rsp = client.embeddings.create(model=EMBED_QUERY_MODEL, input=text)
        arr = np.array(rsp.data[0].embedding, dtype=np.float32).flatten()
        embedding_cache[key] = arr
        return arr
    except:
        return None

# ========== Smart search me pragje ========== 
@app.get("/search")
def search_service(q: str = Query(...)):
    t0 = time.time()

    # 1) CLEAN & REFINE
    cleaned, refined = refine_query(q)

    # 2) EMBEDDING QUERY
    q_emb = embed_query(refined)
    if q_emb is None:
        return {"results":[],"time_sec":round(time.time()-t0,2)}

    # 3) SEARCH SIMILARITY ndaj Supabase table
    rows = supabase.from_("detailedtable").select("id,name,category,embedding_large,keywords").execute().data
    scored = []

    for r in rows:
        emb = to_arr(r.get("embedding_large"))
        if emb is None: continue
        sim = cosine(q_emb, emb)
        sim01 = scale01(sim)

        # elimino poshte 0.6
        if sim01 < 0.6: continue
        # ruaj relevante
        scored.append((sim01, sim, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    # 4) GREEN zona ≥0.7
    for sim01, sim, r in scored[:4]:
        color = "🟢" if sim01>=0.7 else "🟡"
        final.append({"id":r["id"],"name":r["name"],"profession":r["category"],"score":sim01,"zone":color})

    t_total = time.time()-t0
    return {"results":final,"time_sec":round(t_total,2)}
