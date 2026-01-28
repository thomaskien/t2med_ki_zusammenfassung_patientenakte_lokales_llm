#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version 1.4.5
#
# Vollständiger Changelog seit 1.2
# --------------------------------
# 1.2 (Baseline)
# - PDF Watcher + PDF->Text (pypdf), Cleaning, Chunking, Stage1+Stage2 via LMStudio/OpenAI SDK
# - Patientendaten-Extraktion (konservativ) + Header-Erzwingung
# - md-Ausgabe + md2gdt + done/error Ordner
#
# 1.3
# - OpenAI-Client als Singleton (weniger Overhead)
# - Retry/Backoff für LLM-Calls (robuster bei LMStudio-Hängern)
# - Stage2-Input-Limit/Kompression (gegen Context/Token-Probleme)
# - Bugfix Name: kontextbasierte/validierte Namensheuristik statt freiem "Nachname, Vorname"
#
# 1.4
# - Neue Prompt-Logik Kopf+Kartei und getrennte Verarbeitung; Stage1 Chunk1 Kopf+Kartei, Rest nur Kartei
# - Stage2 Formatvorlage mit Rubriken CAVE/Allergien/Dauerdiagnosen/Akutdiagnosen + Kartei-Kategorien
#
# 1.4.1
# - Robustheit: Stage1-Output-Validierung + 1x Strict Re-Prompt
# - Parser toleranter
# - Fallback Kopf aus PDF-Text (bis "Alle Karteikarten-Einträge")
# - Patientennummer auch aus Folgelinie (Patientennummer:\n7975) + Bullet-Name/Geburtsdatum
#
# 1.4.2
# - Kopf-Rubriken deterministisch geparst (Allergien/Akutdiagnosen zuverlässig)
# - Akutdiagnosen gezählt/sortiert ("Diagnose (Nx)")
# - Vor jeder Überschrift "---" als Trenner
# - Stage2-LLM nur noch für Kartei-Kategorien
#
# 1.4.3
# - Neuer harter Trenner in Exportvorlage: "<<< KARTEIKARTE >>>" => deterministischer Split Kopf/Kartei
# - Kopf kommt direkt aus PDF-Text (bis Trenner), Stage1 nur für Kartei
# - Fix gegen "leere Datei": Ausgabe wird auch bei leerer Kartei erzeugt
#
# 1.4.4
# - Fix: Trenner vor "Relevante Vorerkrankungen" erzwungen (Postprocessing)
# - Fix: entfernt "seit" ohne Datum (Postprocessing + Prompt)
# - Exakter Titel für Akutdiagnosen
#
# 1.4.5 (dieses Release)
# - Titel: "Akutdiagnosen (nach Haeufigkeit)" (ASCII)
# - Wichtige Einzelereignisse: max. 20 (statt 10)
# - Filter: Fehleinträge wie "[Seite 2]" werden in Kopf/Kartei/Counts entfernt
# - Modellzeile in Markdown-Header enthält zusätzlich Programmversion

import time
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pypdf import PdfReader
from openai import OpenAI


# =====================================================
# Programm / Konfiguration
# =====================================================
PROGRAM_VERSION = "1.4.5"

WATCH_DIR = Path("/Users/thomaskienzle/Desktop/pdf-ki").resolve()
OUT_DIR = Path("/Users/thomaskienzle/Desktop/pdf-ki-out").resolve()

DONE_PDF_DIR = WATCH_DIR / "done"
ERROR_PDF_DIR = WATCH_DIR / "error"
DONE_MD_DIR = OUT_DIR / "done"

LOG_FILE = WATCH_DIR / "run.log"

MD2GDT_SCRIPT = Path("/Users/thomaskienzle/md2gdt.py").resolve()
PYTHON_BIN = "python3"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL = "openai/gpt-oss-20b"

TEMPERATURE = 0.1
TOP_P = 0.9

MAX_TOKENS_STAGE1 = 900
MAX_TOKENS_STAGE2 = 1400  # etwas Luft für 20 Ereignisse

CHUNK_MAX_CHARS = 12000
MAX_INPUT_CHARS = 800_000

STAGE2_KARTE_MAX_CHARS = 220_000

LLM_RETRIES = 5
LLM_RETRY_BASE_SLEEP = 1.0

# Export-Vorlagen-Trenner
KARTE_TRENNER = "<<< KARTEIKARTE >>>"

# Überschrift: ASCII gewünscht
AKUT_TITLE = "Akutdiagnosen (nach Haeufigkeit)"

# Seite-Marker, die NICHT als Diagnosen/Ereignisse zählen sollen
PAGE_MARKER_RE = re.compile(r"^\s*\[Seite\s+\d+\]\s*$", re.IGNORECASE)


# =====================================================
# Prompts
# =====================================================

SYSTEM_PROMPT_STAGE1_KARTEI_ONLY = r"""Du bist ein medizinischer Dokumentationsassistent.

AUFGABE
Extrahiere aus dem folgenden Text ALLE medizinisch relevanten Fakten aus den Karteieinträgen/Verlauf.

AUSGABEFORMAT (EXAKT, reiner Text, KEIN Markdown)

[KARTEIKARTE]
PAT | <Patientennummer> | <Nachname> | <Vorname> | <Geburtsdatum>
DX  | <Datum> | <Diagnose/Text> | <ICD falls vorhanden>
EV  | <Datum> | <Ereignis/Besonderheit>
ALL | <Datum> | <Allergie>
MED | <Datum> | <Medikation>

REGELN
- Gib NUR den Block [KARTEIKARTE] aus.
- Keine Zusatztexte.
- Ignoriere Seitenmarker wie "[Seite 2]" vollständig (nicht als Diagnose/Ereignis ausgeben).
"""

SYSTEM_PROMPT_STAGE2_KARTEI_ONLY = r"""Du bist ein medizinischer Dokumentationsassistent.

INPUT
Du erhältst genau einen Block [KARTEIKARTE] mit Faktenzeilen:
PAT | ...
DX  | <Datum> | <Diagnose/Text> | <ICD falls vorhanden>
EV  | <Datum> | <Ereignis/Besonderheit>
ALL | <Datum> | <Allergie>
MED | <Datum> | <Medikation>

AUFGABE
Erzeuge NUR diese drei Abschnitte aus den gelieferten Fakten:
- Relevante Vorerkrankungen
- Wiederkehrende Probleme
- Wichtige Einzelereignisse (mit Datum)

REGELN
- Verwende ausschließlich gelieferte Inhalte.
- Erfinde nichts.
- Keine medizinische Interpretation.
- Keine Meta-Ereignisse wie "KI-Zusammenfassung".
- Maximal 20 Einzelereignisse.
- Wenn ein Abschnitt keine Inhalte hat, MUSS dort genau eine Zeile stehen: "- nicht dokumentiert"
- Verwende das Wort "seit" nur, wenn direkt ein Datum/Jahr genannt werden kann. Sonst "seit" weglassen.
- Ignoriere Seitenmarker wie "[Seite 2]" vollständig.

AUSGABEFORMAT (EXAKT, KEIN MARKDOWN)

---
Relevante Vorerkrankungen
==========================
- <Eintrag oder "nicht dokumentiert">

---
Wiederkehrende Probleme
==========================
- <Eintrag oder "nicht dokumentiert">

---
Wichtige Einzelereignisse (mit Datum)
==========================
- <Datum>: <Ereignis>
"""


# =====================================================
# Hilfsfunktionen
# =====================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_until_file_stable(path: Path, checks=4, delay=0.5, timeout=30):
    start = time.time()
    last = -1
    stable = 0
    while time.time() - start < timeout:
        if not path.exists():
            return False
        size = path.stat().st_size
        if size == last:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
            last = size
        time.sleep(delay)
    return False


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages.append(f"[Seite {i+1}]\n{txt}")
    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()

        # Seitenmarker komplett entfernen, damit sie nirgends als Inhalt landen
        if PAGE_MARKER_RE.match(s):
            continue

        if not s:
            lines.append(line)
            continue

        if re.fullmatch(r"[0-9 ,.;:/\\\-()]+", s):
            continue
        if s.lower().startswith("null:"):
            continue

        lines.append(line)
    return "\n".join(lines)


def chunk_text(text: str):
    chunks = []
    pos = 0
    while pos < len(text):
        chunks.append(text[pos:pos + CHUNK_MAX_CHARS])
        pos += CHUNK_MAX_CHARS
    return chunks


# =====================================================
# LLM (Singleton + Retry)
# =====================================================

_CLIENT = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")


def call_llm(system_prompt: str, user_text: str, max_tokens: int) -> str:
    last_err = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            resp = _CLIENT.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            sleep_s = min(LLM_RETRY_BASE_SLEEP * (2 ** (attempt - 1)), 20.0) + (0.1 * attempt)
            log(f"[LLM-RETRY] attempt={attempt}/{LLM_RETRIES} err={repr(e)} sleep={sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {LLM_RETRIES} retries: {repr(last_err)}")


def write_output(pdf_path: Path, summary: str, meta: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"{pdf_path.stem}.md"
    meta_path = OUT_DIR / f"{pdf_path.stem}.meta.json"

    header = [
        "# KI-Zusammenfassung",
        "",
        f"- Quelle: {pdf_path.name}",
        f"- Modell: {MODEL} (pdf_ki {PROGRAM_VERSION})",
        f"- Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    md_path.write_text("\n".join(header) + summary + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path


def move_file(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    src.replace(dst)


def run_md2gdt(md_path: Path):
    subprocess.run([PYTHON_BIN, str(MD2GDT_SCRIPT), str(md_path)], check=True)


def dedupe_preserve_order(lines):
    seen = set()
    out = []
    for ln in lines:
        key = ln.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out


def parse_karte_facts_from_stage1(text: str):
    """
    Erwartet [KARTEIKARTE] Block. Tolerant: sammelt Faktenzeilen auch ohne Marker.
    Filtert Seitenmarker konsequent.
    """
    facts = []
    in_block = False

    def norm_marker(s: str) -> str:
        return re.sub(r"\s+", "", s.strip().upper())

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if PAGE_MARKER_RE.match(s):
            continue
        if norm_marker(s) == "[KARTEIKARTE]":
            in_block = True
            continue

        if in_block or True:
            if s.startswith(("PAT |", "DX", "EV", "ALL", "MED")):
                # Seitenmarker auch innerhalb von Textteilen verhindern
                if "[Seite" in s:
                    continue
                facts.append(s)

    return facts


def compact_karte_block(fact_lines, max_chars: int):
    fact_lines = dedupe_preserve_order(fact_lines)
    kartei_block = "[KARTEIKARTE]\n" + ("\n".join(fact_lines).strip() + "\n" if fact_lines else "")
    if len(kartei_block) <= max_chars:
        return kartei_block
    tail = kartei_block[-max_chars:]
    if "[KARTEIKARTE]" not in tail:
        tail = "[KARTEIKARTE]\n...\n" + tail
    return tail


# =====================================================
# Patient Info Extraction (robust)
# =====================================================

_DATE_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$")

def extract_patient_info_from_text(text: str) -> dict:
    info = {
        "patientennummer": "nicht dokumentiert",
        "nachname": "nicht dokumentiert",
        "vorname": "nicht dokumentiert",
        "geburtsdatum": "nicht dokumentiert",
    }

    head = text[:12000]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]

    def set_pnr(val: str):
        if val and any(ch.isdigit() for ch in val):
            info["patientennummer"] = val.strip()

    def set_geb(val: str):
        if val and _DATE_RE.match(val.strip()):
            info["geburtsdatum"] = val.strip()

    def set_name(val: str):
        v = val.strip()
        m = re.search(r"\b([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{1,40})\s*,\s*([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{1,40})\b", v)
        if m:
            info["nachname"] = m.group(1).strip()
            info["vorname"] = m.group(2).strip()
            return
        parts = [p for p in re.split(r"\s+", v) if p]
        if len(parts) == 2:
            info["vorname"] = parts[0].strip()
            info["nachname"] = parts[1].strip()

    for i, ln in enumerate(lines[:400]):
        low = ln.lower()

        if "patientennummer" in low:
            m = re.search(r"patientennummer\s*:\s*(.+)$", ln, re.IGNORECASE)
            if m and m.group(1).strip():
                set_pnr(m.group(1).strip())
            else:
                if i + 1 < len(lines):
                    nxt = lines[i + 1]
                    if any(ch.isdigit() for ch in nxt):
                        set_pnr(nxt)
            continue

        if "geburtsdatum" in low or low.startswith("dob"):
            m = re.search(r"(geburtsdatum|dob)\s*:\s*(.+)$", ln, re.IGNORECASE)
            if m and m.group(2).strip():
                set_geb(m.group(2).strip())
            else:
                if i + 1 < len(lines):
                    set_geb(lines[i + 1])
            continue

        if re.match(r"^-?\s*name\s*:", ln, re.IGNORECASE):
            m = re.search(r"name\s*:\s*(.+)$", ln, re.IGNORECASE)
            if m and m.group(1).strip():
                set_name(m.group(1).strip())
            else:
                if i + 1 < len(lines):
                    set_name(lines[i + 1])
            continue

    return info


def enforce_patient_header(summary: str, patient_info: dict) -> str:
    pnr = patient_info["patientennummer"]
    nach = patient_info["nachname"]
    vor = patient_info["vorname"]
    geb = patient_info["geburtsdatum"]
    header_line1 = f"Patientennummer: {pnr}"
    header_line2 = f"{nach}, {vor}, {geb}"

    lines = summary.splitlines()
    idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Patientennummer:"):
            idx = i
            break

    if idx is None:
        return header_line1 + "\n" + header_line2 + "\n" + summary.lstrip()

    lines[idx] = header_line1
    if idx + 1 < len(lines):
        lines[idx + 1] = header_line2
    else:
        lines.append(header_line2)

    return "\n".join(lines)


# =====================================================
# Kopf deterministisch aus Text: CAVE/Allergien/Dauerdiagnosen/Akutdiagnosen
# =====================================================

RUB_CAVE = re.compile(r"^\s*cave\s*:?\s*$", re.IGNORECASE)
RUB_ALLERG = re.compile(r"^\s*allergien\s*:?\s*$", re.IGNORECASE)
RUB_DAUER = re.compile(r"^\s*dauerdiagnosen\b.*:?\s*$", re.IGNORECASE)
RUB_AKUT = re.compile(r"^\s*akutdiagnosen\b.*:?\s*$", re.IGNORECASE)

DATE_PREFIX = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{2,4})\s+")
ICD_PARENS = re.compile(r"\s*\([A-Z]\d{1,2}\.\d+[^\)]*\)\s*$")

def normalize_akut_diag(s: str) -> str:
    t = s.strip()
    if PAGE_MARKER_RE.match(t):
        return ""
    t = DATE_PREFIX.sub("", t).strip()
    t = ICD_PARENS.sub("", t).strip()
    t = re.sub(r"^\s*[A-Z]\d{1,2}\.\d+\s*[A-Z]?\s*", "", t).strip()
    # nochmal Seitenmarker abfangen (falls eingebettet)
    if "[Seite" in t:
        return ""
    return t

def parse_head_sections_from_head_text(head_text: str):
    lines = []
    for ln in head_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if PAGE_MARKER_RE.match(s):
            continue
        lines.append(s)

    sections = {"cave": [], "allergien": [], "dauer": [], "akut": []}
    current = None

    for ln in lines:
        if RUB_CAVE.match(ln):
            current = "cave"
            continue
        if RUB_ALLERG.match(ln):
            current = "allergien"
            continue
        if RUB_DAUER.match(ln):
            current = "dauer"
            continue
        if RUB_AKUT.match(ln):
            current = "akut"
            continue

        if current:
            # Seitenmarker/Artefakte nicht übernehmen
            if PAGE_MARKER_RE.match(ln) or "[Seite" in ln:
                continue
            sections[current].append(ln)

    akut_counts = {}
    for item in sections["akut"]:
        diag = normalize_akut_diag(item)
        if not diag:
            continue
        akut_counts[diag] = akut_counts.get(diag, 0) + 1

    akut_sorted = sorted(akut_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    return sections, akut_sorted

def format_head_block(sections, akut_sorted):
    def bullets(lines):
        if not lines:
            return ["- nicht dokumentiert"]
        out = []
        seen = set()
        for x in lines:
            k = x.strip()
            if not k:
                continue
            if PAGE_MARKER_RE.match(k) or "[Seite" in k:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(f"- {k}")
        return out if out else ["- nicht dokumentiert"]

    out = []

    out.append("---")
    out.append("CAVE")
    out.append("==========================")
    out.extend(bullets(sections["cave"]))
    out.append("")

    out.append("---")
    out.append("Allergien")
    out.append("==========================")
    out.extend(bullets(sections["allergien"]))
    out.append("")

    out.append("---")
    out.append("Dauerdiagnosen")
    out.append("==========================")
    out.extend(bullets(sections["dauer"]))
    out.append("")

    out.append("---")
    out.append(AKUT_TITLE)
    out.append("==========================")
    if not akut_sorted:
        out.append("- nicht dokumentiert")
    else:
        for diag, n in akut_sorted:
            # Seitenmarker sicher raus
            if not diag or PAGE_MARKER_RE.match(diag) or "[Seite" in diag:
                continue
            out.append(f"- {diag} ({n}x)")
    if out[-1] == "==========================":
        out.append("- nicht dokumentiert")
    out.append("")

    return "\n".join(out).rstrip() + "\n"


# =====================================================
# Split Kopf/Kartei
# =====================================================

def split_head_karte(cleaned_text: str):
    if KARTE_TRENNER in cleaned_text:
        head, karte = cleaned_text.split(KARTE_TRENNER, 1)
        return head.strip(), karte.strip(), "marker"

    stop_re = re.compile(r"(alle\s+karteikarten|karteikarten\-eintr|ausgewählte\s+karteikarten)", re.IGNORECASE)
    lines = cleaned_text.splitlines()
    head_lines = []
    karte_lines = []
    in_karte = False
    for ln in lines:
        if not in_karte and stop_re.search(ln):
            in_karte = True
            continue
        (karte_lines if in_karte else head_lines).append(ln)
    return "\n".join(head_lines).strip(), "\n".join(karte_lines).strip(), "fallback"


# =====================================================
# Postprocessing: Seitenmarker / "seit" ohne Datum / Trenner
# =====================================================

_RE_SEIT_TRAIL = re.compile(r"(\s+seit)\s*$", re.IGNORECASE)

def fix_stage2_text(stage2_text: str) -> str:
    lines = stage2_text.splitlines()

    # 1) hängendes "seit" entfernen
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("- "):
            lines[i] = _RE_SEIT_TRAIL.sub("", ln).rstrip()

    # 2) Seitenmarker entfernen
    lines = [ln for ln in lines if not PAGE_MARKER_RE.match(ln.strip()) and "[Seite" not in ln]

    # 3) Trenner vor "Relevante Vorerkrankungen" erzwingen
    for i, ln in enumerate(lines):
        if ln.strip() == "Relevante Vorerkrankungen":
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j < 0 or lines[j].strip() != "---":
                lines.insert(i, "---")
            break

    return "\n".join(lines).strip() + "\n"


# =====================================================
# Hauptverarbeitung
# =====================================================

def process_pdf(pdf_path: Path):
    if pdf_path.parent.name in ("done", "error"):
        return

    log(f"[NEW] {pdf_path.name}")

    if not wait_until_file_stable(pdf_path):
        log(f"[WAIT-FAIL] {pdf_path.name}")
        return

    try:
        raw_text = pdf_to_text(pdf_path)
        if not raw_text:
            raise RuntimeError("Kein Text extrahiert (Scan-PDF? Dann OCR nötig).")

        if len(raw_text) > MAX_INPUT_CHARS:
            raw_text = raw_text[:MAX_INPUT_CHARS]

        cleaned = clean_text(raw_text)

        head_text, karte_text, split_mode = split_head_karte(cleaned)
        log(f"[INFO] split_mode={split_mode} head_chars={len(head_text)} karte_chars={len(karte_text)}")

        patient_info = extract_patient_info_from_text(head_text if head_text else cleaned)

        # Kopf deterministisch
        head_sections, akut_sorted = parse_head_sections_from_head_text(head_text)
        head_block_text = format_head_block(head_sections, akut_sorted)

        # Kartei extrahieren
        all_fact_lines = []
        if karte_text.strip():
            chunks = chunk_text(karte_text)
            log(f"[INFO] Kartei-Chunks: {len(chunks)}")
            tmp = []
            for ci, chunk in enumerate(chunks, 1):
                log(f"[STAGE1-KARTEI] Chunk {ci}/{len(chunks)}")
                out = call_llm(SYSTEM_PROMPT_STAGE1_KARTEI_ONLY, chunk, MAX_TOKENS_STAGE1)
                facts = parse_karte_facts_from_stage1(out)
                tmp.extend(facts)
            all_fact_lines = dedupe_preserve_order(tmp)
        else:
            log("[WARN] Kartei-Text nach Trenner ist leer -> Stage2 bekommt leeren Kartei-Block")

        kartei_block = compact_karte_block(all_fact_lines, STAGE2_KARTE_MAX_CHARS)

        # Stage2
        log("[STAGE2] Kartei-Kategorien")
        kartei_summary_raw = call_llm(SYSTEM_PROMPT_STAGE2_KARTEI_ONLY, kartei_block, MAX_TOKENS_STAGE2).strip()
        kartei_summary = fix_stage2_text(kartei_summary_raw).rstrip()

        # Gesamtausgabe
        full = []
        full.append(f"Patientennummer: {patient_info['patientennummer']}")
        full.append(f"{patient_info['nachname']}, {patient_info['vorname']}, {patient_info['geburtsdatum']}")
        full.append("")
        full.append(head_block_text.rstrip())
        full.append("")
        full.append(kartei_summary)

        summary = "\n".join(full).strip() + "\n"
        summary = enforce_patient_header(summary, patient_info)

        meta = {
            "pdf": str(pdf_path),
            "program_version": PROGRAM_VERSION,
            "model": MODEL,
            "patient_info": patient_info,
            "split_mode": split_mode,
            "head_chars": len(head_text),
            "karte_chars": len(karte_text),
            "fact_lines": len(all_fact_lines),
            "karte_block_chars": len(kartei_block),
        }

        md_path = write_output(pdf_path, summary, meta)

        run_md2gdt(md_path)

        move_file(md_path, DONE_MD_DIR)
        move_file(md_path.with_suffix(".meta.json"), DONE_MD_DIR)
        move_file(pdf_path, DONE_PDF_DIR)

        log(f"[OK] {pdf_path.name} vollständig verarbeitet")

    except Exception as e:
        log(f"[ERR] {pdf_path.name}: {repr(e)}")
        try:
            move_file(pdf_path, ERROR_PDF_DIR)
        except Exception:
            pass


# =====================================================
# Watchdog
# =====================================================

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            time.sleep(0.2)
            process_pdf(Path(event.src_path))


def main():
    for d in (WATCH_DIR, OUT_DIR, DONE_PDF_DIR, DONE_MD_DIR, ERROR_PDF_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log("=== PDF-KI Chunked Watcher gestartet ===")

    for pdf in sorted(WATCH_DIR.glob("*.pdf")):
        process_pdf(pdf)

    observer = Observer()
    observer.schedule(Handler(), str(WATCH_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
