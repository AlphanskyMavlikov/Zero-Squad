import os,re,json,hashlib
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

HERE=Path(__file__).parent
EMBED_MODEL="text-embedding-3-small"
EXPLAIN_MODEL=os.getenv("OPENAI_EXPLAIN_MODEL","gpt-5.6-luna")
CACHE=HERE/"embeddings_cache.json"
PATHS=[Path(r"C:\Users\admin\Downloads\hackathon dataset anonymized .csv"),Path(r"C:\Users\admin\Downloads\hackathon dataset anonymized.csv"),HERE/"hackathon dataset anonymized .csv"]

def norm(x): return re.sub(r"\s+"," ",str(x).strip().lower())
def parts(x): return [] if pd.isna(x) else [norm(v) for v in str(x).split("|") if v.strip()]
def key(): return os.getenv("OPENAI_API_KEY")
def cli(): return OpenAI(api_key=key()) if key() else None

@st.cache_data
def load(path):
    d=pd.read_csv(path)
    d["price_from_kzt"]=pd.to_numeric(d["price_from_kzt"],errors="coerce")
    d["max_hours"]=pd.to_numeric(d["max_hours"],errors="coerce")
    for c in ["synthetic","city_imputed","price_imputed"]:
        d[c]=d[c].astype(str).str.lower().isin(["true","1","yes"])
    return d

def cache_load():
    try: return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    except: return {}

def embed(text):
    c=cli()
    if not c:return None
    h=hashlib.sha256((EMBED_MODEL+text).encode()).hexdigest()
    ca=cache_load()
    if h not in ca:
        ca[h]=c.embeddings.create(model=EMBED_MODEL,input=text).data[0].embedding
        CACHE.write_text(json.dumps(ca),encoding="utf-8")
    return ca[h]

def cos(a,b):
    if a is None or b is None:return 0.
    a=np.array(a);b=np.array(b); den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else 0.

def qtext(fmt,cat,lang,hours):
    s=f"Нужен подрядчик категории {cat} для {fmt}."
    if lang:s+=f" Работа на языке {lang}."
    if hours:s+=f" Продолжительность {hours:g} часов."
    return s

def ptext(r):
    return f"Категория: {r.categories}. Форматы: {r.event_formats}. Языки: {r.languages}. Описание: {r.description}"

def failures(r,dt,fmt,budget,lang,hours):
    f=[]
    if dt.isoformat() in parts(r.busy_dates):f.append(("date",f"занят {dt:%d.%m.%Y}"))
    if pd.isna(r.price_from_kzt):f.append(("budget","цена не указана"))
    elif r.price_from_kzt>budget:f.append(("budget",f"цена превышает бюджет на {int(r.price_from_kzt-budget):,} ₸".replace(","," ")))
    if norm(fmt) not in parts(r.event_formats):f.append(("format",f"не берёт формат «{fmt}»"))
    if lang and norm(lang) not in parts(r.languages):f.append(("language",f"не работает на языке «{lang}»"))
    if hours and not pd.isna(r.max_hours) and r.max_hours<hours:f.append(("duration",f"максимум {r.max_hours:g} ч, требуется {hours:g} ч"))
    return f

def breakdown(r,budget,lang,hours,sim):
    budget_pts=round(20*max(0,1-r.price_from_kzt/max(budget,1)),1)
    format_pts=25.
    lang_pts=15. if lang else 10.
    duration_pts=10.
    if hours and not pd.isna(r.max_hours):duration_pts=round(7+3*min(max((r.max_hours-hours)/hours,0),1),1)
    semantic_pts=round(30*max(0,sim),1)
    x={"Бюджет":budget_pts,"Формат":format_pts,"Язык":lang_pts,"Длительность":duration_pts,"Смысл описания":semantic_pts}
    x["Итого"]=round(sum(x.values()),1)
    return x

def explain(r,dt,fmt,cat,budget,lang,hours,sim,br):
    c=cli()
    if not c:return "Добавьте OPENAI_API_KEY — тогда ИИ-агент сгенерирует персональное объяснение. Hard filters продолжают работать."
    ev={"name":r.anon_name,"category":cat,"city":r.city,"date":dt.isoformat(),"event_type":fmt,
        "budget_kzt":int(budget),"price_from_kzt":int(r.price_from_kzt),"requested_language":lang,
        "languages":parts(r.languages),"requested_hours":hours,"max_hours":None if pd.isna(r.max_hours) else float(r.max_hours),
        "description":str(r.description),"semantic_similarity":round(sim,4),"score":br,"availability":"free"}
    sys="""Ты агент объяснения рекомендаций event-подрядчиков. На русском напиши 1–2 конкретных предложения,
почему подрядчик попал в TOP. Используй ТОЛЬКО JSON-факты, ничего не придумывай.
Упомяни наиболее различающие факты: цену относительно бюджета, формат, язык/длительность если запрошены
и конкретный смысл description. Не называй similarity числом. Не используй общие фразы вроде «отличный выбор»."""
    return c.responses.create(model=EXPLAIN_MODEL,input=[{"role":"system","content":sys},{"role":"user","content":json.dumps(ev,ensure_ascii=False)}]).output_text.strip()

def recommend(df,city,dt,fmt,cat,budget,hours,lang):
    pool=df[df.city.map(norm).eq(norm(city)) & df.categories.apply(lambda x:norm(cat) in parts(x))]
    if pool.empty:return {"status":"none","pool":pool,"passed":[],"rejected":[],"counts":{}}
    passed=[];rej=[];counts={k:0 for k in ["date","budget","format","language","duration"]}
    for _,r in pool.iterrows():
        f=failures(r,dt,fmt,budget,lang,hours)
        if f:
            rej.append((r,f))
            for code,_ in f:counts[code]+=1
        else:passed.append(r)
    if not passed:return {"status":"filtered","pool":pool,"passed":[],"rejected":rej,"counts":counts}
    q=embed(qtext(fmt,cat,lang,hours))
    ranked=[]
    for r in passed:
        sim=cos(q,embed(ptext(r)))
        br=breakdown(r,budget,lang,hours,sim)
        ranked.append((br["Итого"],sim,str(r.id),r,br))
    ranked.sort(key=lambda x:(-x[0],-x[1],x[2]))
    return {"status":"ok","pool":pool,"passed":ranked,"rejected":rej,"counts":counts}

st.set_page_config(page_title="Contractor Matcher V2",page_icon="🧠",layout="wide")
st.title("🧠 Умный подбор подрядчиков — V2")
st.caption("Hard filters → semantic embeddings → прозрачный score → AI-agent explanation")

path=next((p for p in PATHS if p.exists()),None)
with st.sidebar:
    up=st.file_uploader("Другой CSV",type="csv")
    if up:
        path=HERE/"_uploaded.csv";path.write_bytes(up.getvalue())
    st.write("OpenAI API:","✅ подключён" if key() else "⚠️ OPENAI_API_KEY не задан")
    if not key():
        st.caption("Задайте ключ перед запуском: $env:OPENAI_API_KEY=\"sk-proj-...\"")
if not path:st.stop()
df=load(str(path))
cities=sorted(df.city.dropna().unique())
cats=sorted({v.strip() for x in df.categories.dropna() for v in str(x).split("|")})
fmts=sorted({v.strip() for x in df.event_formats.dropna() for v in str(x).split("|")})
langs=sorted({v.strip() for x in df.languages.dropna() for v in str(x).split("|")})

with st.form("form"):
    a,b,c=st.columns(3);city=a.selectbox("Город",cities);dt=b.date_input("Дата",date(2026,10,17),min_value=date(2026,9,23),max_value=date(2026,12,31));fmt=c.selectbox("Тип мероприятия",fmts)
    a,b,c=st.columns(3);cat=a.selectbox("Категория",cats);budget=b.number_input("Бюджет, ₸",1,20_000_000,800_000,50_000);lu=c.selectbox("Язык",["Неважно"]+langs)
    useh=st.checkbox("Указать длительность");hours=st.number_input("Часы",1.,24.,6.,.5,disabled=not useh)
    go=st.form_submit_button("Подобрать",type="primary",use_container_width=True)

if go:
    lang=None if lu=="Неважно" else lu;h=float(hours) if useh else None
    with st.spinner("Hard filters + semantic matching..."):res=recommend(df,city,dt,fmt,cat,float(budget),h,lang)
    st.subheader("Воронка отбора")
    n=len(res["pool"]);pn=len(res["passed"])
    st.write(f"**Город + категория:** {n} → **после hard filters:** {pn} → **TOP:** {min(3,pn)}")
    names={"date":"🔴 Заняты на дату","budget":"🟠 Не проходят бюджет","format":"🟡 Не берут формат","language":"🔵 Не подходят по языку","duration":"🟣 Не хватает длительности"}
    rows=[{"Причина":names[k],"Количество":v} for k,v in res["counts"].items() if v]
    if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption("Один профиль может иметь несколько причин отсева.")

    if res["status"]=="none":st.warning(f"В городе {city} категории «{cat}» нет.")
    elif res["status"]=="filtered":st.error("Кандидаты есть, но никто не проходит все hard-условия.")
    else:
        if len(res["passed"])<3:st.warning(f"Подходящих только {len(res['passed'])}; поэтому карточек меньше трёх.")
        cols=st.columns(min(3,len(res["passed"])))
        for col,(total,sim,_,r,br) in zip(cols,res["passed"][:3]):
            with col:
                st.subheader(str(r.anon_name));st.caption("🧪 СИНТЕТИЧЕСКИЙ" if r.synthetic else "👤 ИСХОДНЫЙ")
                st.metric("Совместимость",f"{total:.1f} / 100")
                st.write(f"**Цена от:** {int(r.price_from_kzt):,} ₸".replace(","," "))
                st.dataframe(pd.DataFrame([{"Фактор":k,"Баллы":v} for k,v in br.items() if k!="Итого"]),hide_index=True,use_container_width=True)
                with st.spinner("ИИ-агент пишет объяснение..."):st.info(explain(r,dt,fmt,cat,float(budget),lang,h,sim,br))
                with st.expander("Технически"):st.write(f"Cosine similarity: {sim:.4f}");st.write(f"Embedding: `{EMBED_MODEL}` · LLM: `{EXPLAIN_MODEL}`")

    st.subheader("Почему не этот подрядчик?")
    if not res["rejected"]:st.write("На hard filters никто не был исключён.")
    for r,fs in res["rejected"]:
        with st.expander(f"{r.anon_name} — исключён"):
            for _,reason in fs:st.write("❌ "+reason)
            st.caption("Embeddings не могут вернуть профиль: semantic ranking выполняется только после hard filters.")

with st.expander("Архитектура"):
    st.code("""CSV → City+Category → HARD FILTERS(date/budget/format/language/duration)
→ valid candidates only → embeddings semantic similarity → deterministic score → TOP 3
→ grounded AI explanation from structured evidence""")
