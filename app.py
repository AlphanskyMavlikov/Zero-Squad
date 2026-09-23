from __future__ import annotations
import re
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
PATHS = [
    Path(r"C:\Users\admin\Downloads\hackathon dataset anonymized .csv"),
    Path(r"C:\Users\admin\Downloads\hackathon dataset anonymized.csv"),
    HERE / "hackathon dataset anonymized .csv",
    HERE / "hackathon dataset anonymized.csv",
]

def norm(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())

def parts(x):
    if pd.isna(x) or not str(x).strip():
        return []
    return [norm(v) for v in str(x).split("|") if v.strip()]

def find_csv():
    return next((p for p in PATHS if p.exists()), None)

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    required = {"id","anon_name","categories","city","price_from_kzt","event_formats",
                "languages","max_hours","busy_dates","description","synthetic",
                "city_imputed","price_imputed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Нет колонок: " + ", ".join(sorted(missing)))
    df["price_from_kzt"] = pd.to_numeric(df["price_from_kzt"], errors="coerce")
    df["max_hours"] = pd.to_numeric(df["max_hours"], errors="coerce")
    for c in ["synthetic","city_imputed","price_imputed"]:
        df[c] = df[c].astype(str).str.lower().isin(["true","1","yes"])
    return df

SEMANTIC = {
    "свадьба": ["свад", "невест", "жених", "церемон"],
    "той": ["той", "наурыз", "национал", "казах"],
    "корпоратив": ["корпоратив", "компан", "бизнес", "business", "бренд"],
    "конференция": ["конферен", "форум", "бизнес", "business", "workshop"],
    "юбилей": ["юбилей", "годовщ"],
    "день рождения": ["день рождения", "birthday", "праздник"],
}

def semantic(row, event_type, category):
    text = norm(row["description"])
    words = SEMANTIC.get(norm(event_type), []) + [w for w in norm(category).split() if len(w) >= 5]
    hits = list(dict.fromkeys(w for w in words if w in text))
    return min(len(hits), 3), hits[:2]

def score(row, budget, event_type, category, language, duration):
    price = float(row["price_from_kzt"])
    budget_fit = max(0, 1 - price / max(budget, 1))
    sem, _ = semantic(row, event_type, category)
    lang = 1 if language and norm(language) in parts(row["languages"]) else 0
    dur = 0
    if duration:
        dur = .5 if pd.isna(row["max_hours"]) else min(max((row["max_hours"]-duration)/max(duration,1),0),1)
    return 50 + 20*budget_fit + 10*lang + 8*dur + 4*sem

def explanation(row, event_date, budget, event_type, category, language, duration):
    price = int(row["price_from_kzt"])
    reserve = int(budget - price)
    facts = [f"Цена от {price:,} ₸ укладывается в бюджет и оставляет запас {reserve:,} ₸".replace(",", " "),
             f"подрядчик берёт формат «{event_type}»"]
    if language:
        facts.append(f"работает на языке «{language}»")
    if duration:
        if pd.isna(row["max_hours"]):
            facts.append("для этой услуги длительность не привязана к присутствию на площадке")
        else:
            facts.append(f"доступно до {int(row['max_hours'])} ч при запросе на {duration:g} ч")
    _, hits = semantic(row, event_type, category)
    if hits:
        facts.append("в описании найдены релевантные маркеры: " + ", ".join("«"+x+"»" for x in hits))
    return "; ".join(facts) + f". На {event_date.strftime('%d.%m.%Y')} профиль свободен."

def recommend(df, city, event_date, event_type, category, budget, duration=None, language=None):
    pool = df[
        df["city"].map(norm).eq(norm(city)) &
        df["categories"].apply(lambda x: norm(category) in parts(x))
    ].copy()

    if pool.empty:
        return "no_category", [], {}, f"В городе {city} категории «{category}» в каталоге нет."

    reasons = {"заняты на дату":0, "дороже бюджета":0, "не берут формат":0,
               "не подходят по языку":0, "не хватает длительности":0}
    passed = []
    day = event_date.isoformat()

    for _, row in pool.iterrows():
        failed = []
        if day in parts(row["busy_dates"]): failed.append("заняты на дату")
        if pd.isna(row["price_from_kzt"]) or row["price_from_kzt"] > budget: failed.append("дороже бюджета")
        if norm(event_type) not in parts(row["event_formats"]): failed.append("не берут формат")
        if language and norm(language) not in parts(row["languages"]): failed.append("не подходят по языку")
        if duration and not pd.isna(row["max_hours"]) and row["max_hours"] < duration:
            failed.append("не хватает длительности")
        if failed:
            for f in failed: reasons[f] += 1
        else:
            passed.append(row)

    if not passed:
        detail = "; ".join(f"{k}: {v}" for k,v in reasons.items() if v)
        return "filtered", [], reasons, "Кандидаты есть, но ни один не проходит все условия. " + detail + "."

    ranked = []
    for row in passed:
        s = score(row, budget, event_type, category, language, duration)
        ranked.append((s, str(row["id"]), row))
    ranked.sort(key=lambda x: (-x[0], x[1]))

    cards = []
    for s, _, row in ranked[:3]:
        cards.append({
            "name":row["anon_name"], "category":row["categories"], "city":row["city"],
            "price":int(row["price_from_kzt"]), "score":round(s,2),
            "synthetic":bool(row["synthetic"]), "price_imputed":bool(row["price_imputed"]),
            "city_imputed":bool(row["city_imputed"]),
            "explanation":explanation(row,event_date,budget,event_type,category,language,duration)
        })

    if len(cards) < 3:
        msg = f"Подобрано {len(cards)} из 3: после проверки всех условий осталось только {len(passed)} подходящих."
    else:
        msg = "Подобраны 3 наиболее подходящих свободных подрядчика."
    return "ok", cards, reasons, msg

st.set_page_config(page_title="Умный подбор подрядчиков", page_icon="🎯", layout="wide")
st.title("🎯 Умный подбор event-подрядчиков")
st.caption("Hard-фильтры → детерминированный скоринг → персональное объяснение")

csv_path = find_csv()
with st.sidebar:
    st.header("Датасет")
    upload = st.file_uploader("Загрузить другой CSV", type="csv")
    if upload:
        tmp = HERE / "_uploaded.csv"
        tmp.write_bytes(upload.getvalue())
        csv_path = tmp
    if csv_path: st.success(str(csv_path))
    else: st.error("CSV не найден.")

if not csv_path:
    st.stop()

try:
    df = load_data(str(csv_path))
except Exception as e:
    st.error(f"Ошибка CSV: {e}")
    st.stop()

cities = sorted(df["city"].dropna().astype(str).unique())
categories = sorted({v.strip() for x in df["categories"].dropna() for v in str(x).split("|")})
formats = sorted({v.strip() for x in df["event_formats"].dropna() for v in str(x).split("|")})
langs = sorted({v.strip() for x in df["languages"].dropna() for v in str(x).split("|")})

with st.form("query"):
    a,b,c = st.columns(3)
    city = a.selectbox("Город", cities)
    event_date = b.date_input("Дата", date(2026,10,17), min_value=date(2026,9,23), max_value=date(2026,12,31))
    event_type = c.selectbox("Тип мероприятия", formats)
    a,b,c = st.columns(3)
    category = a.selectbox("Категория", categories)
    budget = b.number_input("Бюджет, ₸", min_value=1, value=800000, step=50000)
    lang_ui = c.selectbox("Язык (опционально)", ["Неважно"] + langs)
    use_duration = st.checkbox("Указать длительность")
    duration = st.number_input("Длительность, ч", 1.0, 24.0, 6.0, .5, disabled=not use_duration)
    go = st.form_submit_button("Подобрать", type="primary", use_container_width=True)

st.caption(f"Профилей: {len(df)} · синтетических: {int(df['synthetic'].sum())}")

if go:
    language = None if lang_ui == "Неважно" else lang_ui
    dur = float(duration) if use_duration else None
    status, cards, reasons, message = recommend(df,city,event_date,event_type,category,float(budget),dur,language)

    if status == "ok":
        st.success(message)
        cols = st.columns(len(cards))
        for col, card in zip(cols,cards):
            with col:
                st.subheader(card["name"])
                st.caption("🧪 СИНТЕТИЧЕСКИЙ ПРОФИЛЬ" if card["synthetic"] else "👤 ИСХОДНЫЙ ПРОФИЛЬ")
                st.write("**Категория:**", card["category"])
                st.write("**Город:**", card["city"])
                st.write(f"**Цена от:** {card['price']:,} ₸".replace(","," "))
                st.info(card["explanation"])
                with st.expander("Почему такой порядок?"):
                    st.write("Детерминированный score:", card["score"])
                    st.write("При равном score порядок фиксируется по ID.")
                    flags=[]
                    if card["price_imputed"]: flags.append("цена восстановлена при подготовке датасета")
                    if card["city_imputed"]: flags.append("город восстановлен при подготовке датасета")
                    if flags: st.warning("; ".join(flags))
        if len(cards) < 3:
            st.warning("Карточек меньше трёх: других профилей, проходящих все hard-условия, нет.")
    elif status == "no_category":
        st.warning("Категории нет в этом городе")
        st.write(message)
    else:
        st.error("Кандидаты есть, но никто не прошёл условия")
        st.write(message)
        active={k:v for k,v in reasons.items() if v}
        st.dataframe(pd.DataFrame({"Причина":active.keys(),"Количество":active.values()}),hide_index=True,use_container_width=True)

with st.expander("Как работает pipeline"):
    st.markdown("""
1. Пул формируется по **городу и категории**.
2. Hard-фильтры проверяют **дату, бюджет, формат, язык и длительность**.
3. Прошедшие кандидаты получают прозрачный score: запас бюджета + язык + длительность + смысловые маркеры описания.
4. Сортировка: `score DESC`, затем `id ASC`. Случайности нет.
5. Объяснение строится из конкретных совпадений профиля с запросом.
""")
