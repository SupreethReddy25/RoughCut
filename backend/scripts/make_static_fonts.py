"""One-off: build the static font files in backend/assets/fonts from the official Google Fonts (OFL) sources.

google/fonts only ships variable TTFs, and libass does not match variable-font named instances
("Montserrat ExtraBold" falls back to a default face). So we pin the weight axis and give each instance its
own family name. Used by libass (fontsdir) and by the browser preview (@font-face) so both render the same.

    backend\\.venv\\Scripts\\python.exe backend\\scripts\\make_static_fonts.py     (needs `pip install fonttools`)
"""
from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

OUT = Path(__file__).resolve().parents[1] / "assets" / "fonts"
BASE = "https://github.com/google/fonts/raw/main/ofl/"
# (url path, axis pins, family name, output file)
FONTS = [
    ("montserrat/Montserrat%5Bwght%5D.ttf", {"wght": 800}, "Montserrat ExtraBold", "Montserrat-ExtraBold.ttf"),
    ("inter/Inter%5Bopsz%2Cwght%5D.ttf", {"wght": 600, "opsz": 14}, "Inter SemiBold", "Inter-SemiBold.ttf"),
]


def rename(font: TTFont, family: str) -> None:
    ps = family.replace(" ", "-")
    name = font["name"]
    for rec in list(name.names):
        if rec.nameID in (16, 17, 21, 22, 25):  # typographic names would re-expose the base family
            name.removeNames(nameID=rec.nameID)
    for pid, eid, lid in ((3, 1, 0x409), (1, 0, 0)):
        name.setName(family, 1, pid, eid, lid)
        name.setName("Regular", 2, pid, eid, lid)
        name.setName(family, 4, pid, eid, lid)
        name.setName(ps, 6, pid, eid, lid)
        name.setName(f"{ps};static", 3, pid, eid, lid)
    font["OS/2"].fsSelection = (font["OS/2"].fsSelection & ~0b1100001) | 0b1000000  # REGULAR bit only
    font["head"].macStyle = 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for url, pins, family, out in FONTS:
            src = Path(td) / Path(url).name.replace("%5B", "[").replace("%5D", "]").replace("%2C", ",")
            urllib.request.urlretrieve(BASE + url, src)
            f = instancer.instantiateVariableFont(TTFont(src), pins)
            rename(f, family)
            f.save(OUT / out)
            print("wrote", OUT / out)


if __name__ == "__main__":
    main()
