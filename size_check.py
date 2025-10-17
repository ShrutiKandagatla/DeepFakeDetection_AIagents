import os
from pathlib import Path
base = Path("Data Set 1")
def resolve(base,subset):
    p1 = base / subset
    p2 = base / "Data Set 1" / subset
    p3 = base / "data set 1" / subset
    return p1 if p1.exists() else (p2 if p2.exists() else p3.exists() and p3 or p1)
for subset in ("train","validation","t"):
    p = resolve(base, subset)
    print(subset, p)
    for c in ("real","fake"):
        d = p / c
        n = len(list(d.glob("*.*"))) if d.exists() else 0
        print("  ",c, n)