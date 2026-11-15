import collections
old = open("paper/thesis_new/draft_new.md", encoding="utf-8").read().split("\n"); new = open("paper/thesis_new/draft_v3.md", encoding="utf-8").read()
nl = new.split("\n"); cnt = collections.Counter(l for l in nl if len(l) > 40)
miss = [(i + 1, l) for i, l in enumerate(old) if len(l) > 40 and l not in cnt and l.replace("(RQ3)", "").replace("(RQ2)", "") not in new]
print("missing old lines:", len(miss))
for i, l in miss: print(" ", i, l[:90])
d = [l for l, c in cnt.items() if c > 1]; print("duplicated lines:", len(d))
for l in d: print("  dup:", l[:90])
