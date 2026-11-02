"""Check every reference in main.tex against Crossref: paper/data/refs_check.json. Prints what needs a person's eye."""
import json
import re
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

P = Path(__file__).resolve().parents[1]
tex = (P / "main.tex").read_text(encoding="utf-8")
items = re.findall(r"\\bibitem\{(\w+)\}\s*(.+)", tex)
cited = set(k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", tex) for k in g.split(","))


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "RoughCutRefCheck/1.0 (academic reference check)"})
    return json.load(urllib.request.urlopen(req, timeout=30))


out = []
for key, text in items:
    m = re.search(r"``(.+?),?''", text)
    title = re.sub(r"[{}\\]", "", m.group(1)) if m else re.sub(r"[{}\\]", "", (re.search(r"\\emph\{(.+?)\}", text) or [0, text[:80]])[1])
    doi = (re.search(r"doi: ([^\s,]+?)\.?$", text.strip()) or [None, None])[1]
    doi = doi.replace("\\_", "_") if doi else None
    rec = {"key": key, "title": title, "doi": doi, "cited": key in cited}
    try:
        if doi:
            w = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
        else:
            w = get("https://api.crossref.org/works?rows=1&query.bibliographic=" + urllib.parse.quote(title))["message"]["items"][0]
        found = (w.get("title") or [""])[0]
        rec.update(found=found, match=round(SequenceMatcher(None, title.lower(), found.lower()).ratio(), 2),
                   year=(w.get("issued", {}).get("date-parts") or [[None]])[0][0], venue=(w.get("container-title") or [""])[0][:60])
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:80]
    out.append(rec)
    time.sleep(0.15)
(P / "data/refs_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(len(out), "refs;", sum(1 for r in out if not r["cited"]), "not cited")
for r in out:
    if r.get("match", 0) < 0.9 or not r["cited"]:
        print(r["key"], "| doi" if r["doi"] else "| no doi", "|", r.get("match"), "|", r["title"][:60], "->", r.get("found", r.get("error", ""))[:60], r.get("year"))
