import os,re,json,hashlib
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIStatusError

HERE=Path(__file__).parent
EMBED_MODEL="text-embedding-3-small"
EXPLAIN_MODEL=os.getenv("OPENAI_EXPLAIN_MODEL","gpt-5.6-luna")
CACHE=HERE/"embeddings_cache.json"
PATHS=[Path(r"C:\Users\admin\Downloads\hackathon dataset anonymized .csv"),Path(r"C:\Users\admin\Downloads\hackathon dataset anonymized.csv"),HERE/"hackathon dataset anonymized .csv"]

def norm(x): return re.sub(r"\s+"," ",str(x).strip().lower())
def parts(x): return [] if pd.isna(x) else [norm(v) for v in str(x).split("|") if v.strip()]
def key():
    # Автоматически используем локальный .streamlit/secrets.toml.
    # Если файла нет — пробуем переменную среды. Ошибка отсутствия secrets не ломает приложение.
    try:
        value = st.secrets["OPENAI_API_KEY"]
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.getenv("OPENAI_API_KEY")
    return value.strip() if value else None
def cli(): return OpenAI(api_key=key()) if key() else None

def check_openai_connection():
    if not key(): return "missing","API-ключ не найден."
    try:
        cli().embeddings.create(model=EMBED_MODEL,input="connection test")
        return "ok",f"OpenAI API работает. Embedding model: {EMBED_MODEL}"
    except AuthenticationError: return "auth","API-ключ недействителен (401). Создайте новый ключ."
    except RateLimitError as e:
        m=str(e).lower()
        return ("quota","Ключ принят, но квота/баланс исчерпаны или биллинг не настроен.") if any(x in m for x in ["quota","billing","credit"]) else ("rate","Достигнут временный лимит запросов.")
    except APIConnectionError: return "network","Нет соединения с OpenAI API."
    except APIStatusError as e: return "api",f"OpenAI API вернул HTTP {e.status_code}."
    except Exception as e: return "other",f"{type(e).__name__}: {e}"

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

def embed_many(texts):
    c=cli()
    if not c:return [None for _ in texts]
    ca=cache_load()
    keys=[hashlib.sha256((EMBED_MODEL+t).encode()).hexdigest() for t in texts]
    missing=[(i,t,k) for i,(t,k) in enumerate(zip(texts,keys)) if k not in ca]
    if missing:
        response=c.embeddings.create(model=EMBED_MODEL,input=[t for _,t,_ in missing])
        for (i,t,k),item in zip(missing,response.data):
            ca[k]=item.embedding
        CACHE.write_text(json.dumps(ca),encoding="utf-8")
    return [ca[k] for k in keys]

def embed(text):
    return embed_many([text])[0]

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

def mode_config(mode):
    # Дата, город и категория никогда не ослабляются.
    # Бюджет: Hard = строго; Medium = до +15%; Light = до +30%.
    # Medium/Light также мягче относятся к языку и длительности, но формат остаётся обязательным.
    return {
        "Hard":   {"budget_over":0.00, "duration_shortfall":0.00, "language_hard":True},
        "Medium": {"budget_over":0.15, "duration_shortfall":0.15, "language_hard":False},
        "Light":  {"budget_over":0.30, "duration_shortfall":0.30, "language_hard":False},
    }[mode]

def failures(r,dt,fmt,budget,lang,hours,mode):
    cfg=mode_config(mode)
    f=[]
    if dt.isoformat() in parts(r.busy_dates):
        f.append(("date",f"занят {dt:%d.%m.%Y}"))
    if pd.isna(r.price_from_kzt):
        f.append(("budget","цена не указана"))
    elif r.price_from_kzt > budget*(1+cfg["budget_over"]):
        limit=int(budget*(1+cfg["budget_over"]))
        over_limit=int(r.price_from_kzt-limit)
        f.append(("budget",
            f"цена {int(r.price_from_kzt):,} ₸ выше допустимого лимита {limit:,} ₸ "
            f"на {over_limit:,} ₸".replace(","," ")))
    if norm(fmt) not in parts(r.event_formats):
        f.append(("format",f"не берёт формат «{fmt}»"))
    if lang and cfg["language_hard"] and norm(lang) not in parts(r.languages):
        f.append(("language",f"не работает на языке «{lang}»"))
    if hours and not pd.isna(r.max_hours):
        min_hours=hours*(1-cfg["duration_shortfall"])
        if r.max_hours < min_hours:
            f.append(("duration",f"максимум {r.max_hours:g} ч — слишком мало для запроса {hours:g} ч в режиме {mode}"))
    return f

def relaxed_notes(r,budget,lang,hours,mode):
    """Что именно было ослаблено для прошедшего кандидата."""
    notes=[]
    if r.price_from_kzt>budget:
        diff=int(r.price_from_kzt-budget)
        pct=(r.price_from_kzt/budget-1)*100
        notes.append(f"цена выше бюджета на {diff:,} ₸ ({pct:.1f}%), но это допустимо в режиме {mode}".replace(","," "))
    elif r.price_from_kzt<=budget:
        reserve=int(budget-r.price_from_kzt)
        notes.append(f"цена укладывается в бюджет; запас составляет {reserve:,} ₸".replace(","," "))
    if lang and norm(lang) not in parts(r.languages) and mode!="Hard":
        notes.append(f"язык «{lang}» не совпадает; в режиме {mode} язык считается мягким критерием")
    if hours and not pd.isna(r.max_hours) and r.max_hours<hours and mode!="Hard":
        notes.append(f"длительность меньше запроса на {hours-r.max_hours:g} ч, но находится в допустимом отклонении режима {mode}")
    return notes

def breakdown(r,budget,lang,hours,sim,mode):
    cfg=mode_config(mode)
    ratio=float(r.price_from_kzt)/max(float(budget),1)
    if ratio<=1:
        # По ТЗ бюджет — верхняя граница, а не целевая цена.
        # Любая цена <= бюджета полностью проходит бюджетный критерий.
        budget_pts=20.0
    else:
        # Medium/Light могут допустить контролируемое превышение бюджета.
        # Чем ближе цена к верхней границе допуска, тем меньше бюджетный балл.
        over=ratio-1
        tolerance=max(cfg["budget_over"],0.01)
        budget_pts=round(20*max(0,1-over/tolerance),1)
    format_pts=25.
    if not lang:
        lang_pts=15.0
    else:
        lang_pts=15.0 if norm(lang) in parts(r.languages) else 0.0
    duration_pts=10.0
    if hours and not pd.isna(r.max_hours):
        duration_pts=round(10*min(max(float(r.max_hours)/hours,0),1),1)
    semantic_pts=round(30*max(0,sim),1)
    x={"Бюджет":budget_pts,"Формат":format_pts,"Язык":lang_pts,"Длительность":duration_pts,"Смысл описания":semantic_pts}
    x["Итого"]=round(sum(x.values()),1)
    return x

def explain(r,dt,fmt,cat,budget,lang,hours,sim,br,mode):
    c=cli()
    if not c:return "ИИ-объяснение временно недоступно: API-ключ не настроен."
    ev={"name":r.anon_name,"category":cat,"city":r.city,"date":dt.isoformat(),"event_type":fmt,
        "budget_kzt":int(budget),"price_from_kzt":int(r.price_from_kzt),"requested_language":lang,
        "languages":parts(r.languages),"requested_hours":hours,"max_hours":None if pd.isna(r.max_hours) else float(r.max_hours),
        "description":str(r.description),"semantic_similarity":round(sim,4),"score":br,"availability":"free",
        "selection_mode":mode,"relaxed_conditions":relaxed_notes(r,budget,lang,hours,mode)}
    sys="""Ты агент объяснения рекомендаций event-подрядчиков. На русском напиши 1–2 конкретных предложения,
почему подрядчик попал в TOP. Используй ТОЛЬКО JSON-факты, ничего не придумывай.
Упомяни наиболее различающие факты: цену относительно бюджета, формат, язык/длительность если запрошены
и конкретный смысл description. Бюджет пользователя — это верхняя граница: цена значительно ниже бюджета
не является недостатком и не должна описываться как плохое совпадение. Не называй similarity числом.
Не используй общие фразы вроде «отличный выбор»."""
    return c.responses.create(model=EXPLAIN_MODEL,input=[{"role":"system","content":sys},{"role":"user","content":json.dumps(ev,ensure_ascii=False)}]).output_text.strip()

def recommend(df,city,dt,fmt,cat,budget,hours,lang,mode):
    pool=df[df.city.map(norm).eq(norm(city)) & df.categories.apply(lambda x:norm(cat) in parts(x))]
    if pool.empty:return {"status":"none","pool":pool,"passed":[],"rejected":[],"counts":{}}
    passed=[];rej=[];counts={k:0 for k in ["date","budget","format","language","duration"]}
    for _,r in pool.iterrows():
        f=failures(r,dt,fmt,budget,lang,hours,mode)
        if f:
            rej.append((r,f))
            for code,_ in f:counts[code]+=1
        else:passed.append(r)
    if not passed:return {"status":"filtered","pool":pool,"passed":[],"rejected":rej,"counts":counts}
    texts=[qtext(fmt,cat,lang,hours)]+[ptext(r) for r in passed]
    vectors=embed_many(texts)
    q=vectors[0]
    ranked=[]
    for r,pvec in zip(passed,vectors[1:]):
        sim=cos(q,pvec)
        br=breakdown(r,budget,lang,hours,sim,mode)
        ranked.append((br["Итого"],sim,str(r.id),r,br))
    ranked.sort(key=lambda x:(-x[0],-x[1],x[2]))
    return {"status":"ok","pool":pool,"passed":ranked,"rejected":rej,"counts":counts}


def near_misses(rejected,budget,limit=3):
    scored=[]
    weights={"date":4,"format":4,"budget":2,"language":1,"duration":2}
    for r,fs in rejected:
        penalty=sum(weights.get(code,1) for code,_ in fs)
        if not pd.isna(r.price_from_kzt) and budget:
            penalty += min(abs(float(r.price_from_kzt)-budget)/budget,1)
        scored.append((penalty,str(r.id),r,fs))
    scored.sort(key=lambda x:(x[0],x[1]))
    return scored[:limit]

def count_for_mode(df,city,dt,fmt,cat,budget,hours,lang,mode):
    pool=df[df.city.map(norm).eq(norm(city)) & df.categories.apply(lambda x:norm(cat) in parts(x))]
    return sum(1 for _,r in pool.iterrows() if not failures(r,dt,fmt,budget,lang,hours,mode))

def compare_top_agent(top,dt,fmt,cat,budget,lang,hours,mode):
    c=cli()
    if not c:return "AI-сравнение недоступно: API-ключ не настроен."
    items=[]
    for total,sim,_,r,br in top:
        items.append({"name":str(r.anon_name),"price":int(r.price_from_kzt),"budget":int(budget),
                      "languages":parts(r.languages),"max_hours":None if pd.isna(r.max_hours) else float(r.max_hours),
                      "description":str(r.description),"semantic_similarity":round(sim,4),
                      "score_breakdown":br,"mode":mode})
    prompt="""Сравни подрядчиков на русском языке, не выбирая победителя и не рекомендуя одного из них.
Опиши 2–4 конкретных различия: бюджет, смысл описания, язык, длительность и компромиссы.
Используй только JSON. Не пиши общих фраз. 2–4 предложения."""
    return c.responses.create(model=EXPLAIN_MODEL,input=[
        {"role":"system","content":prompt},
        {"role":"user","content":json.dumps(items,ensure_ascii=False)}
    ]).output_text.strip()

def pair_reason(a,b,budget):
    _,s1,_,r1,br1=a; _,s2,_,r2,br2=b
    diffs=[]
    d1=int(budget-r1.price_from_kzt); d2=int(budget-r2.price_from_kzt)
    if d1!=d2:
        diffs.append(f"Оба проходят бюджетный критерий; запас бюджета: {r1.anon_name} — {d1:,} ₸, {r2.anon_name} — {d2:,} ₸".replace(","," "))
    if abs(s1-s2)>.001:
        who=r1.anon_name if s1>s2 else r2.anon_name
        diffs.append(f"у {who} выше смысловое совпадение описания с запросом")
    best=[]
    for k in ["Язык","Длительность","Смысл описания","Бюджет"]:
        if br1[k]!=br2[k]:
            best.append(f"{k}: {br1[k]} против {br2[k]}")
    if best: diffs.append("; ".join(best[:2]))
    return ". ".join(diffs)+"."

st.set_page_config(page_title="Умный подбор подрядчиков",page_icon="🧠",layout="wide")
st.title("🧠 Умный подбор подрядчиков")
st.caption("Результаты обновляются автоматически при изменении параметров заказа.")

path=next((p for p in PATHS if p.exists()),None)
with st.sidebar:
    st.header("Настройки")
    up=st.file_uploader("Другой CSV",type=["csv"])
    if up:
        path=HERE/"_uploaded.csv"; path.write_bytes(up.getvalue())
    st.subheader("🎬 Демо для жюри")
    st.caption("Быстрые сценарии можно выбрать перед заполнением формы:")
    demo_choice=st.selectbox("Сценарий",["Ручной ввод","Плотная категория","Редкая категория","Нет результатов","Hard vs Medium","Сравнение дат"])
    st.subheader("OpenAI API")
    st.write("🔑 Локальный ключ найден" if key() else "⚠️ API-ключ не настроен")
    if st.button("Проверить подключение OpenAI",use_container_width=True):
        with st.spinner("Проверяю реальным API-запросом..."):
            status,msg=check_openai_connection()
        if status=="ok": st.success("✅ "+msg)
        elif status=="auth": st.error("❌ "+msg)
        else: st.warning("⚠️ "+msg)
    st.caption("Ключ хранится локально в .streamlit/secrets.toml.")

if not path:st.stop()
df=load(str(path))
cities=sorted(df.city.dropna().unique())
cats=sorted({v.strip() for x in df.categories.dropna() for v in str(x).split("|")})
fmts=sorted({v.strip() for x in df.event_formats.dropna() for v in str(x).split("|")})
langs=sorted({v.strip() for x in df.languages.dropna() for v in str(x).split("|")})

st.subheader("Каталог")
m1,m2,m3,m4=st.columns(4)
added_syn=int(df.id.astype(str).str.startswith("SYN-").sum())
source_profiles=len(df)-added_syn
m1.metric("Подрядчиков",len(df))
m2.metric("Из исходного файла",source_profiles)
m3.metric("Добавлено нами",added_syn)
m4.metric("Категорий",len(cats))
st.caption(f"В исходном файле: {int((~df.synthetic).sum())} реальных и {int((df.synthetic & ~df.id.astype(str).str.startswith('SYN-')).sum())} синтетических профилей. Все добавленные нами профили имеют synthetic=true.")

demo_defaults={
    "Плотная категория":("Алматы",date(2026,10,13),"корпоратив","Ведущий",2000000),
    "Редкая категория":("Астана",date(2026,10,24),"свадьба","Флорист",700000),
    "Нет результатов":("Алматы",date(2026,12,19),"корпоратив","Ведущий",100000),
    "Hard vs Medium":("Алматы",date(2026,11,14),"корпоратив","Ведущий",650000),
    "Сравнение дат":("Алматы",date(2026,11,14),"корпоратив","Фотограф",700000),
}
dd=demo_defaults.get(demo_choice)
def idx(values,value):
    try:return values.index(value)
    except:return 0

if "last_result" not in st.session_state:
    st.session_state.last_result=None
if "last_query" not in st.session_state:
    st.session_state.last_query=None
if "top_comparison" not in st.session_state:
    st.session_state.top_comparison=None

st.subheader("Параметры заказа")
a,b,c=st.columns(3)
city=a.selectbox("Город",cities,index=idx(cities,dd[0]) if dd else 0,key="city_live")
dt=b.date_input("Дата",dd[1] if dd else date(2026,10,17),min_value=date(2026,9,23),max_value=date(2026,12,31),key="date_live")
fmt=c.selectbox("Тип мероприятия",fmts,index=idx(fmts,dd[2]) if dd else 0,key="format_live")

a,b,c=st.columns(3)
cat=a.selectbox("Категория",cats,index=idx(cats,dd[3]) if dd else 0,key="category_live")
budget=b.number_input("Бюджет, ₸",1,20_000_000,dd[4] if dd else 800_000,50_000,key="budget_live")
lu=c.selectbox("Язык",["Неважно"]+langs,key="language_live")

useh=st.checkbox("Указать длительность",key="use_duration_live")
hours=st.number_input("Часы",1.,24.,6.,.5,disabled=not useh,key="hours_live")
mode=st.radio(
    "Режим отбора",
    ["Hard","Medium","Light"],
    horizontal=True,
    help="Hard — строгие условия. Medium — умеренные отклонения. Light — более гибкий поиск.",
    key="mode_live"
)
if mode=="Hard":
    st.caption("🔒 Hard: любая цена ≤ бюджета проходит; превышение бюджета не допускается. Язык и длительность — строгие условия.")
elif mode=="Medium":
    st.caption("⚖️ Medium: любая цена ниже бюджета проходит; сверх бюджета допускается до +15%. Также допускается до 15% нехватки по длительности; язык — мягкий критерий.")
else:
    st.caption("🪶 Light: любая цена ниже бюджета проходит; сверх бюджета допускается до +30%. Также допускается до 30% нехватки по длительности; язык — мягкий критерий.")

# Подбор выполняется автоматически после любого изменения параметров.
go=True

compare_dates=st.checkbox("📅 Сравнить с другой датой",value=(demo_choice=="Сравнение дат"))
second_date=st.date_input("Вторая дата",date(2026,11,21),min_value=date(2026,9,23),max_value=date(2026,12,31),disabled=not compare_dates)

# Автоподбор: изменение любого виджета автоматически запускает расчёт.
lang=None if lu=="Неважно" else lu
h=float(hours) if useh else None
try:
    with st.spinner("Обновляю подбор..."):
        res=recommend(df,city,dt,fmt,cat,float(budget),h,lang,mode)
except AuthenticationError:
    st.error("❌ OpenAI отклонил API-ключ (401). Создайте новый ключ и выполните настройку заново.")
    st.stop()
except RateLimitError as e:
    msg=str(e).lower()
    if any(x in msg for x in ["quota","billing","credit"]):
        st.error("⚠️ API-ключ распознан, но квота/баланс исчерпаны или биллинг не настроен.")
    else:
        st.error("⏳ Достигнут временный лимит запросов OpenAI. Попробуйте позже.")
    st.stop()
except APIConnectionError:
    st.error("❌ Нет соединения с OpenAI API. Проверьте интернет.")
    st.stop()
except Exception as e:
    st.error(f"Ошибка API: {type(e).__name__}: {e}")
    st.stop()

query_signature=(city,dt.isoformat(),fmt,cat,float(budget),h,lang,mode)
if st.session_state.get("comparison_query_signature") != query_signature:
    st.session_state.top_comparison=None
    st.session_state.comparison_query_signature=query_signature

st.session_state.last_result=res
st.session_state.last_query={
    "city":city,"dt":dt,"fmt":fmt,"cat":cat,"budget":float(budget),
    "h":h,"lang":lang,"mode":mode,"compare_dates":compare_dates,"second_date":second_date
}

if compare_dates:
    st.subheader("📅 Сравнение дат")
    second_count=count_for_mode(df,city,second_date,fmt,cat,float(budget),h,lang,mode)
    first_count=len(res["passed"])
    pool_n=len(res["pool"])
    first_busy=sum(1 for _,r in res["pool"].iterrows() if dt.isoformat() in parts(r.busy_dates))
    second_busy=sum(1 for _,r in res["pool"].iterrows() if second_date.isoformat() in parts(r.busy_dates))
    st.dataframe(pd.DataFrame([
        {"Показатель":f"Заняты из {pool_n}","Дата 1":first_busy,"Дата 2":second_busy},
        {"Показатель":"Прошли условия","Дата 1":first_count,"Дата 2":second_count},
    ],index=None),hide_index=True,use_container_width=True)
    st.info(f"На {dt:%d.%m.%Y} заняты {first_busy} из {pool_n}, после условий остаётся {first_count}. "
            f"На {second_date:%d.%m.%Y} заняты {second_busy}, после условий остаётся {second_count}.")

st.subheader("Воронка отбора")
n=len(res["pool"]);pn=len(res["passed"])
st.write(f"**Город + категория:** {n} → **после {mode}-отбора:** {pn} → **TOP:** {min(3,pn)}")
names={"date":"🔴 Заняты на дату","budget":"🟠 Не проходят бюджет","format":"🟡 Не берут формат","language":"🔵 Не подходят по языку","duration":"🟣 Не хватает длительности"}
rows=[{"Причина":names[k],"Количество":v} for k,v in res["counts"].items() if v]
if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.caption("Один профиль может иметь несколько причин отсева.")

if res["status"]=="none":st.warning(f"В городе {city} категории «{cat}» нет.")
elif res["status"]=="filtered":st.error(f"Кандидаты есть, но никто не проходит условия режима {mode}.")
else:
    if len(res["passed"])<3:st.warning(f"Подходящих только {len(res['passed'])}; поэтому карточек меньше трёх.")
    cols=st.columns(min(3,len(res["passed"])))
    for col,(total,sim,_,r,br) in zip(cols,res["passed"][:3]):
        with col:
            st.subheader(str(r.anon_name))
            if str(r.id).startswith("SYN-"):
                st.caption("🧪 ДОБАВЛЕННЫЙ СИНТЕТИЧЕСКИЙ ПРОФИЛЬ")
            elif r.synthetic:
                st.caption("🧪 СИНТЕТИЧЕСКИЙ ПРОФИЛЬ ИЗ ИСХОДНОГО ДАТАСЕТА")
            else:
                st.caption("👤 РЕАЛЬНЫЙ ПРОФИЛЬ ИЗ ИСХОДНОГО ДАТАСЕТА")
            pct=max(0,min(100,float(total)))
            st.metric("Совместимость",f"{pct:.1f}%")
            st.progress(pct/100)
            st.caption(f"Режим отбора: **{mode}**")
            notes=relaxed_notes(r,float(budget),lang,h,mode)
            if notes:
                if mode!="Hard" and any(r.price_from_kzt>float(budget) for _ in [0]):
                    pct_over=(float(r.price_from_kzt)/float(budget)-1)*100
                    st.warning(f"⚠️ Допущено отклонение: цена выше бюджета на {pct_over:.1f}%.")
                st.write("**Почему прошёл условия:**")
                for note in notes: st.write("• "+note)
            st.write(f"**Цена от:** {int(r.price_from_kzt):,} ₸".replace(","," "))
            maxpts={"Бюджет":20,"Формат":25,"Язык":15,"Длительность":10,"Смысл описания":30}
            st.dataframe(pd.DataFrame([{"Фактор":k,"Баллы":v,"Максимум":maxpts[k]} for k,v in br.items() if k!="Итого"]),hide_index=True,use_container_width=True)
            with st.spinner("ИИ-агент пишет объяснение..."):st.info(explain(r,dt,fmt,cat,float(budget),lang,h,sim,br,mode))
            with st.expander("Технически"):st.write(f"Cosine similarity: {sim:.4f}");st.write(f"Embedding: `{EMBED_MODEL}` · LLM: `{EXPLAIN_MODEL}`")

    if len(res["passed"])>=2:
        st.subheader("🔎 Почему №1 выше №2?")
        st.write(pair_reason(res["passed"][0],res["passed"][1],float(budget)))
    if len(res["passed"])>=2:
        if st.button("🤖 Сравнить TOP-подрядчиков",use_container_width=True):
            try:
                with st.spinner("AI сравнивает различия..."):
                    st.session_state.top_comparison=compare_top_agent(
                        res["passed"][:3],dt,fmt,cat,float(budget),lang,h,mode
                    )
            except Exception as e:
                st.session_state.top_comparison=f"AI-сравнение временно недоступно: {type(e).__name__}"
        if st.session_state.top_comparison:
            st.info(st.session_state.top_comparison)

# Если текущий режим ничего не дал — показать почти подходящих и следующий режим.
if res["status"]=="filtered":
    st.subheader("🥈 Ближе всего к запросу")
    for _,_,r,fs in near_misses(res["rejected"],float(budget),3):
        st.markdown(f"**{r.anon_name}** — {int(r.price_from_kzt):,} ₸".replace(","," "))
        for _,reason in fs: st.write("⚠️ "+reason)
    if mode=="Hard":
        medium_n=count_for_mode(df,city,dt,fmt,cat,float(budget),h,lang,"Medium")
        light_n=count_for_mode(df,city,dt,fmt,cat,float(budget),h,lang,"Light")
        if medium_n:
            st.info(f"При строгих условиях результатов нет. В режиме Medium найдено: {medium_n}. Переключите режим на Medium, чтобы посмотреть их.")
        elif light_n:
            st.info(f"Hard и Medium не дали результатов. В режиме Light найдено: {light_n}. Переключите режим на Light.")
    elif mode=="Medium":
        light_n=count_for_mode(df,city,dt,fmt,cat,float(budget),h,lang,"Light")
        if light_n: st.info(f"В режиме Light найдено: {light_n}. Переключите режим на Light, чтобы посмотреть их.")

st.subheader("Почему не этот подрядчик?")
if not res["rejected"]:st.write(f"На этапе {mode}-отбора никто не был исключён.")
for r,fs in res["rejected"]:
    with st.expander(f"{r.anon_name} — исключён"):
        for _,reason in fs:st.write("❌ "+reason)