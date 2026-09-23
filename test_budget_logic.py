import ast,re,random,hashlib
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

tree=ast.parse(Path("app.py").read_text(encoding="utf-8"))
wanted={"norm","parts","mode_config","failures","breakdown"}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
mod=ast.Module(body=nodes,type_ignores=[]); ast.fix_missing_locations(mod)
ns={"pd":pd,"np":np,"re":re,"date":date}
exec(compile(mod,"budget_logic","exec"),ns)

class R:
    pass

r=R()
r.busy_dates=""
r.event_formats="корпоратив"
r.languages="русский"
r.max_hours=8
r.price_from_kzt=1_500_000

# 1.5m must pass against 5m in every mode.
for mode in ["Hard","Medium","Light"]:
    assert not any(code=="budget" for code,_ in ns["failures"](r,date(2026,10,1),"корпоратив",5_000_000,"русский",6,mode))
    br=ns["breakdown"](r,5_000_000,"русский",6,0.7,mode)
    assert br["Бюджет"]==20.0, (mode,br)

# Hard rejects over-budget.
r.price_from_kzt=5_100_000
assert any(code=="budget" for code,_ in ns["failures"](r,date(2026,10,1),"корпоратив",5_000_000,"русский",6,"Hard"))

# Medium accepts up to +15%, rejects above.
r.price_from_kzt=5_700_000
assert not any(code=="budget" for code,_ in ns["failures"](r,date(2026,10,1),"корпоратив",5_000_000,"русский",6,"Medium"))
r.price_from_kzt=5_800_000
assert any(code=="budget" for code,_ in ns["failures"](r,date(2026,10,1),"корпоратив",5_000_000,"русский",6,"Medium"))

# Light accepts up to +30%, rejects above.
r.price_from_kzt=6_500_000
assert not any(code=="budget" for code,_ in ns["failures"](r,date(2026,10,1),"корпоратив",5_000_000,"русский",6,"Light"))
r.price_from_kzt=6_600_000
assert any(code=="budget" for code,_ in ns["failures"](r,date(2026,10,1),"корпоратив",5_000_000,"русский",6,"Light"))

print("PASS: budget is an upper limit")
print("PASS: 1 500 000 passes against 5 000 000 in Hard/Medium/Light")
print("PASS: all <= budget candidates receive 20/20 budget points")
print("PASS: Medium +15% and Light +30% only relax the upper boundary")
