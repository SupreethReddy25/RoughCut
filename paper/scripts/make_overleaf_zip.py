"""Package the paper for Overleaf: paper/RoughCut_paper_overleaf.zip (compile main.tex with pdfLaTeX)."""
import zipfile
from pathlib import Path

P = Path(__file__).resolve().parents[1]
files = [P / n for n in ("main.tex", "ieeeaccess.cls", "IEEEtran.cls", "logo.png", "notaglinelogo.png", "bullet.png")]
files += sorted((P / "figures").iterdir())
out = P / "RoughCut_paper_overleaf.zip"
out.unlink(missing_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(P).as_posix())
print(len(files), "files;", zipfile.ZipFile(out).namelist()[6])
