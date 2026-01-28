#!/usr/bin/env python3
# md2gdt.py  (T2med SA 6310, Volltext via 6228, CP437-safe)

import argparse
import re
import textwrap
import unicodedata
from pathlib import Path

IMPORT_DIR = Path("/Users/thomaskienzle/Desktop/pdf-ki-out")
OUTPUT_NAME = "T2MD1.gdt"

SATZART = "6310"
DEFAULT_KENNFELD = "ALLG00"
DEFAULT_CODE = "KI01"
DEFAULT_NAME = "KI-Zusammenfassung"

MAX_6228_CHARS = 90


def gdt_line(field4: str, content: str) -> str:
    payload = f"{field4}{content}"
    length = len(payload) + 3
    return f"{length:03d}{payload}"


def strip_markdown(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"^\s*---\s*\n.*?\n---\s*\n", "", md, flags=re.DOTALL)
    md = re.sub(r"```[^\n]*\n(.*?)```", r"\1", md, flags=re.DOTALL)
    md = re.sub(r"`([^`]*)`", r"\1", md)
    md = re.sub(r"^\s{0,3}#{1,6}\s*", "", md, flags=re.MULTILINE)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    md = re.sub(r"^\s*[-*+]\s+", "", md, flags=re.MULTILINE)
    md = re.sub(r"^\s*\d+\.\s+", "", md, flags=re.MULTILINE)
    md = re.sub(r"\*\*(.*?)\*\*", r"\1", md)
    md = re.sub(r"\*(.*?)\*", r"\1", md)
    md = re.sub(r"_(.*?)_", r"\1", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def normalize_for_search(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ")
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = re.sub(r"[*_`]", "", s)
    return s


def extract_patient_number(md_raw: str) -> str | None:
    s = normalize_for_search(md_raw)
    patterns = [
        r"(?im)^\s*patientennummer\s*[:]?\s*([0-9]{1,10})\b",
        r"(?im)^\s*patientennr\.?\s*[:]?\s*([0-9]{1,10})\b",
        r"(?im)^\s*pat\.?\s*nr\.?\s*[:]?\s*([0-9]{1,10})\b",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    m = re.search(r"(?i)patientennummer\s*[:]?\s*([0-9]{1,10})\b", s)
    return m.group(1) if m else None


def sanitize_text_cp437(text: str) -> str:
    t = unicodedata.normalize("NFC", text)

    # Typografische Zeichen & alle Unicode-Bindestriche  ASCII
    repl = {
        "\u2010": "-",  # HYPHEN
        "\u2011": "-",  # NON-BREAKING HYPHEN
        "\u2012": "-",  # FIGURE DASH
        "\u2013": "-",  # EN DASH
        "\u2014": "-",  # EM DASH
        "\u2212": "-",  # MINUS SIGN
        "\u2043": "-",  # HYPHEN BULLET
        "\u2026": "...",
        "\u201e": '"', "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u00a0": " ",
    }
    for a, b in repl.items():
        t = t.replace(a, b)

    # Fallback: alles aus Unicode-Kategorie "Pd" sicher abfangen
    t = "".join("-" if unicodedata.category(ch) == "Pd" else ch for ch in t)

    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Finales "Einfrieren" als CP437
    t = t.encode("cp437", errors="replace").decode("cp437", errors="replace")
    return t


def wrap_6228(text: str, width: int) -> list[str]:
    out: list[str] = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            out.append(" ")
            continue
        out.extend(textwrap.wrap(
            para,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ))
    while out and out[0] == " ":
        out.pop(0)
    while out and out[-1] == " ":
        out.pop()
    return out or ["(leer)"]


def next_vorgangsnummer_from_existing(file_path: Path, encoding: str) -> int | None:
    if not file_path.exists():
        return None
    raw = file_path.read_bytes().decode(encoding, errors="ignore")
    for line in raw.splitlines():
        if len(line) >= 7 and line[3:7] == "8100":
            val = re.sub(r"\D", "", line[7:])
            if val:
                try:
                    return int(val) + 1
                except ValueError:
                    pass
    return None


def build_gdt(vorgang: int, patient_id: str, kennfeld: str, code: str, name: str, lines_6228: list[str]) -> str:
    vorgang_str = f"{vorgang:06d}"

    gdt: list[str] = []
    gdt.append(gdt_line("8000", SATZART))
    gdt.append(gdt_line("8100", vorgang_str))

    # Patientennummer in beide Felder
    gdt.append(gdt_line("0101", patient_id))
    gdt.append(gdt_line("3000", patient_id))

    gdt.append(gdt_line("9206", "2"))
    gdt.append(gdt_line("9218", "02.10"))

    gdt.append(gdt_line("8402", kennfeld))
    gdt.append(gdt_line("6200", code))
    gdt.append(gdt_line("6201", name))

    for l in lines_6228:
        gdt.append(gdt_line("6228", l))

    gdt.append(gdt_line("9999", ""))
    return "\n".join(gdt) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("md_file")
    ap.add_argument("--code", default=DEFAULT_CODE)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--kennfeld", default=DEFAULT_KENNFELD)
    ap.add_argument("--vorgang", type=int, default=None)
    ap.add_argument("--encoding", default="cp437", help="T2med zeigt Umlaute i.d.R. korrekt mit CP437")
    ap.add_argument("--outdir", default=str(IMPORT_DIR))
    args = ap.parse_args()

    md_path = Path(args.md_file)
    raw_md = md_path.read_text(encoding="utf-8", errors="replace")

    patient_id = extract_patient_number(raw_md)
    if not patient_id:
        raise SystemExit("Keine Patientennummer gefunden (z.B. 'Patientennummer: 246').")

    plain = strip_markdown(raw_md)
    plain = sanitize_text_cp437(plain)
    lines_6228 = wrap_6228(plain, MAX_6228_CHARS)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / OUTPUT_NAME

    vorgang = args.vorgang
    if vorgang is None:
        auto = next_vorgangsnummer_from_existing(out_path, args.encoding)
        vorgang = auto if auto is not None else 200

    gdt = build_gdt(vorgang, patient_id, args.kennfeld, args.code, args.name, lines_6228)
    out_path.write_bytes(gdt.encode(args.encoding, errors="replace"))

    print(f"OK: 0101={patient_id} 3000={patient_id} 8100={vorgang:06d} -> {out_path} (enc={args.encoding})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
