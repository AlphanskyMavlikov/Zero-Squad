import ast, hashlib, random, calendar, re
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

APP=Path(__file__).with_name("app.py")
DATA=Path(__file__).with_name("hackathon dataset anonymized .csv")
tree=ast.parse(APP.read_text(encoding="utf-8"))
wanted={"norm","parts","mode_config","failures","relaxed_notes","breakdown","cos","qtext","ptext","recommend","near_misses","count_for_mode","pair_reason"}
body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
mod=ast.Module(body=body,type_ignores=[]); ast.fix_missing_locations(mod)
ns={"pd":pd,"np":np,"re":re,"date":date}
exec(compile(mod,"logic","exec"),ns)

def fake_embed_many(texts):
    out=[]
    for text in texts:
        r=random.Random(int(hashlib.sha256(text.encode()).hexdigest()[:16],16))
        out.append([r.random() for _ in range(48)])
    return out
ns["embed_many"]=fake_embed_many

df=pd.read_csv(DATA)
df["price_from_kzt"]=pd.to_numeric(df.price_from_kzt,errors="coerce")
df["max_hours"]=pd.to_numeric(df.max_hours,errors="coerce")
for c in ["synthetic","city_imputed","price_imputed"]:
    df[c]=df[c].astype(str).str.lower().isin(["true","1","yes"])

assert len(df)==166
assert df.id.astype(str).str.startswith("SYN-").sum()==100
assert df.synthetic.sum()==113

cities=list(df.city.dropna().unique())
cats=sorted({x.strip() for v in df.categories.dropna() for x in str(v).split("|")})
fmts=sorted({x.strip() for v in df.event_formats.dropna() for x in str(v).split("|")})
langs=sorted({x.strip() for v in df.languages.dropna() for x in str(v).split("|")})

rng=random.Random(7901)
for i in range(5000):
    city=rng.choice(cities); cat=rng.choice(cats); fmt=rng.choice(fmts)
    budget=rng.randint(1,30)*100000
    month=rng.choice([9,10,11,12])
    dt=date(2026,month,rng.randint(1,calendar.monthrange(2026,month)[1]))
    lang=rng.choice([None]+langs); hours=rng.choice([None,3.,6.,10.,12.])
    results=[]
    for mode in ["Hard","Medium","Light"]:
        a=ns["recommend"](df,city,dt,fmt,cat,budget,hours,lang,mode)
        b=ns["recommend"](df,city,dt,fmt,cat,budget,hours,lang,mode)
        assert [x[2] for x in a["passed"]]==[x[2] for x in b["passed"]], "non-deterministic order"
        for _,_,_,r,_ in a["passed"]:
            assert dt.isoformat() not in ns["parts"](r.busy_dates), "busy contractor passed"
        results.append(len(a["passed"]))
    assert results[0] <= results[1] <= results[2], ("mode monotonicity",results)

dense=ns["recommend"](df,"Алматы",date(2026,10,13),"корпоратив","Ведущий",2000000,None,None,"Hard")
assert dense["status"]=="ok" and len(dense["passed"])>=3, "dense demo must rank >=3"

rare=ns["recommend"](df,"Астана",date(2026,10,24),"свадьба","Флорист",700000,None,None,"Hard")
assert len(rare["pool"])>0, "rare demo pool missing"

empty=ns["recommend"](df,"Алматы",date(2026,12,19),"корпоратив","Ведущий",100000,None,None,"Hard")
assert empty["status"]=="filtered" and len(empty["passed"])==0, "empty demo must be empty"

d1=ns["recommend"](df,"Алматы",date(2026,11,14),"корпоратив","Фотограф",700000,None,None,"Hard")
d2=ns["recommend"](df,"Алматы",date(2026,11,21),"корпоратив","Фотограф",700000,None,None,"Hard")
assert [x[2] for x in d1["passed"]] != [x[2] for x in d2["passed"]], "date demo should differ"

print("PASS: 5000 randomized queries")
print("PASS: deterministic ranking")
print("PASS: busy-date exclusion")
print("PASS: Hard <= Medium <= Light")
print("PASS: dense / rare / empty demos")
print("PASS: two-date output differs")
