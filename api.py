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

# ========== PARAMETRA ==========
GREEN_TH  = 0.70
YELLOW_TH = 0.60   # do përdoret vetëm nëse ka 1–2 green
RED_TH    = 0.50   # poshtë 0.50 eliminohen

app_api = FastAPI()

def cosine(a, b):
    na, nb = norm(a), norm(b)
    return 0.0 if na==0 or nb==0 else float(np.dot(a,b)/(na*nb))

def scale01(x): return max(0.0, min(1.0,(x+1.0)/2.0))

def gpt_check(service_name, query):
    p = f'A është shërbimi "{service_name}" i përshtatshëm për "{query}"? Kthe: po/jo'
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":p}],
            temperature=0, max_tokens=3
        )
        return r.choices[0].message.content.strip().lower()=="po"
    except:
        return False

@app_api.get("/search")
def search_service(q: str):
    t0 = time.time()

    # clean pa kufizim + pa detyrim sinonimesh
    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+","",q.lower()).strip()
    
    # refined duhet 100% e njëjtë me input, nuk e ndryshojmë më
    refined = cleaned  

    # embed query
    rsp = client.embeddings.create(model="text-embedding-3-large",input=refined)
    qemb = np.array(rsp.data[0].embedding,dtype=np.float32)

    # kap embedding nga Supabase
    rows = supabase.from_("detailedtable").select("id,name,embedding_large").execute().data
    scored=[]
    for r in rows:
        emb = np.array(r["embedding_large"],dtype=np.float32)
        sim = cosine(qemb, emb)
        sim01 = scale01(sim)
        if sim01 < RED_TH: continue   # eliminohet poshtë 0.5
        scored.append((sim01,sim,r))
    scored.sort(key=lambda x:x[0],reverse=True)

    final=[]
    if scored:
        for sim01,sim,r in scored[:4]:
            final.append({"id":r["id"],"name":r["name"],"score":round(sim01,3)})

            if len(final)==2:
                # nëse kemi 1–2 green (≥0.6) → marrim një yellow me GPT
                if len(scored)>2:
                    y = scored[2]
                    if GREEN_TH > len(final) >= 1 and YELLOW_TH <= y[0] < GREEN_TH:
                        if gpt_check(y[2]["name"], refined):
                            final.append({"id":y[2]["id"],"name":y[2]["name"],"score":round(y[0],3)})
                break

    t_total = time.time()-t0
    return {"results":final,"time_sec":round(t_total,2)}

app = app_api
