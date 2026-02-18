# fix_visplot.py

from pathlib import Path

src = Path("collect/calibrators_original.txt")
dst = Path("collect/calibrators.txt")

lines = src.read_text(encoding="utf-8").splitlines()
out = []

for line in lines:
    s = line
    while "visplot" in s:
        i = s.find("visplot")
        before = s[:i]
        after = s[i + len("visplot"):]

        out.append(before + "visplot")

        # special case: new source glued after visplot
        if "J2000" in after:
            out.append("")   # extra blank line

        s = after  # <-- NO STRIPPING, NO TOUCHING

    if s != "":
        out.append(s)

dst.write_text("\n".join(out) + "\n", encoding="utf-8")

# quick_patch_lowercase_config.py
from pathlib import Path
p = Path("collect/calibrators.txt")
t = p.read_text(encoding="utf-8")
t = t.replace(" S S S s ", " S S S S ")
p.write_text(t, encoding="utf-8")
print("done")


print("done")
