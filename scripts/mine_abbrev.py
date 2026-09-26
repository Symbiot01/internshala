import sys, pandas as pd
from collections import Counter
sys.path.insert(0, 'code/business_entity_resolution')
from src.textnorm import transliterate
import unicodedata, re
def basic(t):
    t = unicodedata.normalize("NFKC", t or "")
    t = transliterate(t).casefold().replace("&"," and ")
    return " ".join(re.sub(r"[^a-z0-9]+"," ",t).split())
from rapidfuzz import fuzz
cand = pd.read_csv("/tmp/gold_pairs_sample.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s1 = s1.set_index("entity_id")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt.matched_entity_ids.str.len()>0].sample(1500, random_state=0)
owner={}
for r in gt.itertuples(index=False):
    for mid in r.matched_entity_ids.split(","): owner[mid]=r.source1_entity_id
pairs = []
for r in cand.itertuples(index=False):
    if r.entity_id not in owner or owner[r.entity_id] not in s1.index: continue
    o = s1.loc[owner[r.entity_id]]
    pairs.append((basic(o.business_name), basic(r.business_name)))
    pairs.append((basic(o.business_address), basic(r.business_address)))
print("pairs:", len(pairs))
mapping = Counter()
for a,b in pairs:
    ta = set(w for w in a.split() if len(w)>1); tb = set(w for w in b.split() if len(w)>1)
    for w in ta-tb:
        best=None; bs=0
        for v in tb-ta:
            s = fuzz.ratio(w,v)
            if s>bs: bs=s; best=v
        if best and 55<=bs<100:
            mapping[(w,best)]+=1
for (w,v),c in mapping.most_common(50):
    print(f"  {w} <-> {v}  x{c}")
