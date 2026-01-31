#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version 1.5.3
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
# 1.4.5
# - Titel: "Akutdiagnosen (nach Haeufigkeit)" (ASCII)
# - Wichtige Einzelereignisse: max. 20
# - Filter: Fehleinträge wie "[Seite 2]" entfernt (clean + parsing + post)
# - Modellzeile in Markdown-Header enthält zusätzlich Programmversion
#
# 1.5
# - Zwei optionale Übergaben (beide unabhängig schaltbar):
#   (1) Übergabe basierend auf der gesamten Kartei (Rohtext in Reihenfolge)
#   (2) Übergabe basierend auf der strukturierten Zusammenfassung
#
# 1.5.1
# - Refusal-Handling für Übergabe aus voller Kartei (automatischer Re-Prompt, entschärfter Prompt)
# - Übergabe aus Zusammenfassung als Fließtext (Validator + Auto-Retry gegen Markdown/Listen)
# - Input-Diet für Übergabe aus Zusammenfassung (Top Akut + Kernpunkte statt kompletter Listenwiederholung)
#
# 1.5.2
# - Kurzakte-Modus (automatisch anhand karte_chars / fact_lines)
#   * niedrigere Tokenlimits + kürzere Ziellängen (Stage2/Stage3), um "Aufblasen" bei kurzen Akten zu verhindern
# - PROMPT: Stage2 unterscheidet akut vs. vorerkrankung (Akutdiagnosen nicht stumpf kopieren)
# - FILTER: Kopfblock-Bullets unterdrücken Rubrikenzeilen wie "Weitere Dauerdiagnosen (gesamt):"
# - FILTER: Übergabe aus voller Kartei wird zusätzlich vorgefiltert (Codes, Bild:/Spiro, Labor-Rohlisten, Impfstatus, Vitalwerte)
#
# 1.5.3 (dieses Release)
# - ÄNDERUNG: Bei Kurzakte (short_case=True) werden KEINE Übergaben erzeugt und NICHT ausgegeben.
# - VERBESSERUNG: clean_text() global verschärft (Labor-Rohwertlisten, Codes-only, Bild:/Spiro, Scheckheft),
#   um Stage1/Stage2 Qualität insgesamt zu erhöhen.
# - BUGFIX: Flags ENABLE_HANDOVER_* wirken wieder für Langakten (keine Platzhalter-Ausgabe mehr).

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
PROGRAM_VERSION = "1.5.3"

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
MAX_TOKENS_STAGE2_LONG = 1400
MAX_TOKENS_STAGE2_SHORT = 750

# Stage3 (Übergaben)
MAX_TOKENS_HANDOVER_KARTEI_LONG = 900
MAX_TOKENS_HANDOVER_KARTEI_SHORT = 380

MAX_TOKENS_HANDOVER_SUMMARY_LONG = 650
MAX_TOKENS_HANDOVER_SUMMARY_SHORT = 320

CHUNK_MAX_CHARS = 12000
MAX_INPUT_CHARS = 800_000
STAGE2_KARTE_MAX_CHARS = 220_000

# Übergabe aus Kartei: begrenzen (head+tail), um Reihenfolge-Charakter zu behalten
HANDOVER_KARTEI_HEAD_CHARS = 30_000
HANDOVER_KARTEI_TAIL_CHARS = 90_000

LLM_RETRIES = 5
LLM_RETRY_BASE_SLEEP = 1.0

KARTE_TRENNER = "<<< KARTEIKARTE >>>"
AKUT_TITLE = "Akutdiagnosen (nach Haeufigkeit)"

# Optionen: unabhängig schaltbar (wirken nur bei Langakten, Kurzakte skippt Stage3 komplett)
ENABLE_HANDOVER_FROM_FULL_KARTEI = True
ENABLE_HANDOVER_FROM_STRUCTURED_SUMMARY = False

# Kurzakte-Schwellen (kombiniert)
SHORT_CASE_MAX_KARTE_CHARS = 15_000
SHORT_CASE_MAX_FACT_LINES = 60


# =====================================================
# Filter / Regex
# =====================================================
PAGE_MARKER_RE = re.compile(r"^\s*\[Seite\s+\d+\]\s*$", re.IGNORECASE)
INLINE_PAGE_RE = re.compile(r"\[Seite\s+\d+\]", re.IGNORECASE)

CODES_ONLY_RE = re.compile(r"^\s*\d{4,5}(\s*,\s*\d{4,5})*\s*$")
BILD_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{2,4}\s+)?Bild\s*:.*$", re.IGNORECASE)
SPIRO_RE = re.compile(r"\bspiro\b", re.IGNORECASE)
SCHECKHEFT_RE = re.compile(r"scheckheft", re.IGNORECASE)

LAB_TOKEN_RE = re.compile(r"\b[A-ZÄÖÜ]{2,8}\s*:\s*[-+]?\d", re.IGNORECASE)

# Impfstatus / Vitalwerte / Maße (für Übergabe-Filter)
VITALS_RE = re.compile(r"\b(rr\b|p\s*\d+|puls|blutdruck)\b", re.IGNORECASE)
MEAS_RE = re.compile(r"\b(gewicht|größe|groesse|cm|kg|m)\b", re.IGNORECASE)
IMPF_RE = re.compile(r"\bimpfstatus\b|\bimpf\w*\b", re.IGNORECASE)

HEAD_RUBRIC_LINE_RE = re.compile(
    r"^\s*(weitere\s+dauerdiagnosen.*|dau(er)?diagnosen.*\(gesamt\).*|akutdiagnosen.*\(.*\).*|allergien\s*:|cave\s*:)\s*$",
    re.IGNORECASE
)


def is_labs_raw_line(s: str) -> bool:
    """
    Robust gegen verschiedene Laborformate:
    - viele Param:Wert Paare
    - lange Zeilen mit vielen ':' / ',' / Ziffern
    """
    t = s.strip()
    if not t:
        return False

    lab_tokens = len(LAB_TOKEN_RE.findall(t))
    colon_count = t.count(":")
    comma_count = t.count(",")

    if lab_tokens >= 4:
        return True

    if len(t) >= 140 and (colon_count >= 6 or comma_count >= 10):
        return True

    if len(t) >= 180 and sum(ch.isdigit() for ch in t) >= 20 and (colon_count + comma_count) >= 8:
        return True

    return False


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
- Ignoriere Seitenmarker wie "[Seite 2]" vollständig (nicht ausgeben).
- Reine Abrechnungsziffern / Ziffernkolonnen ignorieren.
- Labor-Rohwertlisten ignorieren, sofern keine Diagnose/Beurteilung genannt ist.
"""

SYSTEM_PROMPT_STAGE2_KARTEI_LONG = r"""Du bist ein medizinischer Dokumentationsassistent.

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

WICHTIG (Klassifikation)
- "Relevante Vorerkrankungen" = eher chronisch/rezidivierend/dauerhaft oder klar als Vorerkrankung dokumentiert.
  Akutdiagnosen NICHT einfach kopieren, wenn sie nur einmalig akut erscheinen.
- "Wiederkehrende Probleme" = mehrfach dokumentiert / wiederholt / rezidivierend (oder als Dauerdiagnose aktiviert).
- "Wichtige Einzelereignisse" = zeitlich klar datierbare, relevante Ereignisse.

REGELN
- Verwende ausschließlich gelieferte Inhalte.
- Erfinde nichts. Keine medizinische Interpretation. Keine Ergänzungen.
- Keine Meta-Ereignisse wie "KI-Zusammenfassung".
- Maximal 20 Einzelereignisse.
- Wenn ein Abschnitt keine Inhalte hat: "- nicht dokumentiert"
- "seit" nur bei vorhandenem Datum/Jahr.
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

SYSTEM_PROMPT_STAGE2_KARTEI_SHORT = r"""Du bist ein medizinischer Dokumentationsassistent.

INPUT
Du erhältst genau einen Block [KARTEIKARTE] mit Faktenzeilen.

AUFGABE (KURZAKTE)
Erzeuge NUR diese drei Abschnitte – sehr knapp:
- Relevante Vorerkrankungen
- Wiederkehrende Probleme
- Wichtige Einzelereignisse (mit Datum)

WICHTIG
- Bei kurzer Akte NICHT aufblasen.
- Akutdiagnosen NICHT einfach als "Vorerkrankung" kopieren, wenn sie nur einmalig akut erscheinen.
- Maximal 8 Einzelereignisse.

REGELN
- Nur gelieferte Inhalte, nichts erfinden.
- Keine Interpretation, keine Ergänzungen.
- Wenn leer: "- nicht dokumentiert"
- Kein Markdown.

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

SYSTEM_PROMPT_HANDOVER_FROM_FULL_KARTEI_LONG = r"""Du bist ein medizinischer Dokumentationsassistent.

INPUT
Du erhältst den Verlauf / die Karteieinträge in natürlicher Reihenfolge.

AUFGABE
Formuliere eine kurze Übergabe in natürlicher Sprache (Arzt-zu-Arzt), ca. 6–10 Sätze.

INHALTLICH
- Nenne die wichtigsten Diagnosen/Probleme und relevante Zeitpunkte/Verläufe.
- Nenne Allergien/CAVE, falls dokumentiert; sonst "CAVE/Allergien: nicht dokumentiert".
- Beruf/Familie/Sozialanamnese NUR, wenn explizit dokumentiert; sonst "Sozial/Beruf/Familie: nicht dokumentiert".

REGELN
- Nur umformulieren/auswählen. KEINE Empfehlungen. KEINE Therapieentscheidungen.
- Keine Interpretation, keine neuen Diagnosen, keine Ergänzungen.
- KEINE Aufzählungen, KEIN Markdown, keine Überschriften – nur ein Fließtext-Absatz.

AUSGABEFORMAT (EXAKT)
<ein Absatz Fließtext>
"""

SYSTEM_PROMPT_HANDOVER_FROM_FULL_KARTEI_SHORT = r"""Du bist ein medizinischer Dokumentationsassistent.

INPUT
Kurze Akte / wenig Verlauf.

AUFGABE
Formuliere eine sehr kurze Übergabe in natürlicher Sprache (max. 4–6 Sätze).
- Nur wichtigste Diagnosen/Probleme + 1–3 Schlüsseldaten/Ereignisse.
- CAVE/Allergien erwähnen, wenn vorhanden; sonst "CAVE/Allergien: nicht dokumentiert".
- Sozial/Beruf/Familie nur wenn dokumentiert; sonst "Sozial/Beruf/Familie: nicht dokumentiert".
- Keine Interpretation, keine Ergänzungen.
- Keine Listen, kein Markdown, nur ein Absatz.

AUSGABEFORMAT
<ein Absatz>
"""

SYSTEM_PROMPT_HANDOVER_FROM_FULL_KARTEI_STRICT = r"""Gib GENAU EINEN Absatz Fließtext (keine Listen, kein Markdown, keine Überschriften).
Nur Inhalte aus dem Input verwenden. Keine Empfehlungen. Keine Interpretation. Keine Ergänzungen.
<ein Absatz>
"""

SYSTEM_PROMPT_HANDOVER_FROM_STRUCTURED_SUMMARY_LONG = r"""Du bist ein medizinischer Dokumentationsassistent.

INPUT
Du erhältst eine kompakte strukturierte Zusammenfassung.

AUFGABE
Formuliere eine kurze Übergabe in natürlicher Sprache (ca. 4–7 Sätze).

REGELN
- Nur gelieferte Informationen.
- Keine Empfehlungen, keine Interpretation, keine Ergänzungen.
- Keine Listen, kein Markdown, nur ein Absatz.
- CAVE/Allergien erwähnen (auch wenn "nicht dokumentiert").
- Sozial/Beruf/Familie nur wenn explizit dokumentiert; sonst "Sozial/Beruf/Familie: nicht dokumentiert".

AUSGABEFORMAT
<ein Absatz>
"""

SYSTEM_PROMPT_HANDOVER_FROM_STRUCTURED_SUMMARY_SHORT = r"""Du bist ein medizinischer Dokumentationsassistent.

INPUT
Kurze Akte / wenig Informationen.

AUFGABE
Formuliere eine sehr kurze Übergabe (max. 3–5 Sätze).
- Nur die wichtigsten Punkte.
- CAVE/Allergien erwähnen (auch wenn "nicht dokumentiert").
- Sozial/Beruf/Familie nur wenn dokumentiert; sonst "Sozial/Beruf/Familie: nicht dokumentiert".
- Keine Listen, kein Markdown, nur ein Absatz.

AUSGABEFORMAT
<ein Absatz>
"""

SYSTEM_PROMPT_HANDOVER_FROM_STRUCTURED_SUMMARY_STRICT = r"""Gib GENAU EINEN Absatz Fließtext.
Keine Listen, kein Markdown, keine Überschriften.
Nur Inhalte aus dem Input verwenden. Keine Empfehlungen. Keine Interpretation.
<ein Absatz>
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
    """
    Globaler Vorfilter: entfernt typische Noise-Zeilen, bevor Stage1/Stage2/Parser arbeiten.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()

        if PAGE_MARKER_RE.match(s):
            continue

        if not s:
            out.append(line)
            continue

        if INLINE_PAGE_RE.search(line):
            line = INLINE_PAGE_RE.sub("", line)
            s = line.strip()
            if not s:
                continue

        if re.fullmatch(r"[0-9 ,.;:/\\\-()]+", s):
            continue
        if s.lower().startswith("null:"):
            continue

        if CODES_ONLY_RE.match(s):
            continue

        if BILD_RE.match(s) or SPIRO_RE.search(s) or SCHECKHEFT_RE.search(s):
            continue

        if is_labs_raw_line(s):
            continue

        out.append(line)

    return "\n".join(out)


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


# =====================================================
# Validator / Refusal-Detection
# =====================================================

_REFUSAL_PATTERNS = [
    "kann diese anfrage nicht erfüllen",
    "kann dabei nicht helfen",
    "cannot comply",
    "i can't help with that",
    "sorry",
    "entschuldigung",
]

def looks_like_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return any(p in t for p in _REFUSAL_PATTERNS)

def looks_like_markdown_or_list(text: str) -> bool:
    t = text or ""
    if "**" in t or "```" in t or "#" in t:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    listish = 0
    for ln in lines:
        if ln.startswith("- ") or re.match(r"^\d+\.\s+", ln):
            listish += 1
    return listish >= 2

def ensure_single_paragraph(text: str) -> str:
    t = (text or "").replace("**", "").replace("`", "")
    t = re.sub(r"\n{2,}", "\n", t).strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return "nicht dokumentiert"
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^\-\s+", "", ln)
        ln = re.sub(r"^\d+\.\s+", "", ln)
        cleaned.append(ln)
    para = " ".join(cleaned)
    para = re.sub(r"\s{2,}", " ", para).strip()
    return para if para else "nicht dokumentiert"

def call_llm_with_guard(primary_system: str, strict_system: str, user_text: str, max_tokens: int) -> str:
    out = call_llm(primary_system, user_text, max_tokens)
    if looks_like_refusal(out) or looks_like_markdown_or_list(out):
        log("[HANDOVER-GUARD] retry with STRICT prompt")
        out2 = call_llm(strict_system, user_text, max_tokens)
        if looks_like_refusal(out2):
            return "nicht dokumentiert"
        out = out2
    return ensure_single_paragraph(out)


# =====================================================
# Stage1 Parser
# =====================================================

def parse_karte_facts_from_stage1(text: str):
    facts = []
    in_block = False

    def norm_marker(s: str) -> str:
        return re.sub(r"\s+", "", s.strip().upper())

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if PAGE_MARKER_RE.match(s) or "[Seite" in s:
            continue
        if norm_marker(s) == "[KARTEIKARTE]":
            in_block = True
            continue

        if in_block or True:
            if s.startswith(("PAT |", "DX", "EV", "ALL", "MED")):
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
# Patient Info Extraction
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
# Kopf-Sektion (deterministisch)
# =====================================================

RUB_CAVE = re.compile(r"^\s*cave\s*:?\s*$", re.IGNORECASE)
RUB_ALLERG = re.compile(r"^\s*allergien\s*:?\s*$", re.IGNORECASE)
RUB_DAUER = re.compile(r"^\s*dauerdiagnosen\b.*:?\s*$", re.IGNORECASE)
RUB_AKUT = re.compile(r"^\s*akutdiagnosen\b.*:?\s*$", re.IGNORECASE)

DATE_PREFIX = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{2,4})\s+")
ICD_PARENS = re.compile(r"\s*\([A-Z]\d{1,2}\.\d+[^\)]*\)\s*$")

def normalize_akut_diag(s: str) -> str:
    t = s.strip()
    if PAGE_MARKER_RE.match(t) or "[Seite" in t:
        return ""
    t = DATE_PREFIX.sub("", t).strip()
    t = ICD_PARENS.sub("", t).strip()
    t = re.sub(r"^\s*[A-Z]\d{1,2}\.\d+\s*[A-Z]?\s*", "", t).strip()
    if "[Seite" in t:
        return ""
    return t

def parse_head_sections_from_head_text(head_text: str):
    lines = []
    for ln in head_text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if PAGE_MARKER_RE.match(s) or "[Seite" in s:
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
            if HEAD_RUBRIC_LINE_RE.match(k):
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
        wrote_any = False
        for diag, n in akut_sorted:
            if not diag or PAGE_MARKER_RE.match(diag) or "[Seite" in diag:
                continue
            out.append(f"- {diag} ({n}x)")
            wrote_any = True
        if not wrote_any:
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
# Postprocessing Stage2
# =====================================================

_RE_SEIT_TRAIL = re.compile(r"(\s+seit)\s*$", re.IGNORECASE)

def fix_stage2_text(stage2_text: str) -> str:
    lines = stage2_text.splitlines()

    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("- "):
            lines[i] = _RE_SEIT_TRAIL.sub("", ln).rstrip()

    lines = [ln for ln in lines if not PAGE_MARKER_RE.match(ln.strip()) and "[Seite" not in ln]

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
# Übergabe: Vorfilter Kartei (zusätzlich zu clean_text)
# =====================================================

def should_drop_for_handover(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if PAGE_MARKER_RE.match(s) or "[Seite" in s:
        return True
    if CODES_ONLY_RE.match(s):
        return True
    if BILD_RE.match(s) or SPIRO_RE.search(s) or SCHECKHEFT_RE.search(s):
        return True
    if is_labs_raw_line(s):
        return True
    if IMPF_RE.search(s):
        return True
    if VITALS_RE.search(s):
        return True
    if MEAS_RE.search(s) and re.search(r"\d", s):
        return True
    return False

def build_filtered_karte_for_handover(karte_text: str) -> str:
    kept = []
    for ln in karte_text.splitlines():
        if should_drop_for_handover(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


# =====================================================
# Übergabe: Input-Diet für structured summary
# =====================================================

def extract_section_lines(text: str, title: str):
    lines = text.splitlines()
    out = []
    in_sec = False
    for ln in lines:
        if ln.strip() == title:
            in_sec = True
            continue
        if in_sec:
            if ln.strip() == "---":
                break
            out.append(ln)
    return [x.strip() for x in out if x.strip()]

def build_handover_input_from_structured(structured_summary_text: str, short_case: bool) -> str:
    lines = [ln for ln in structured_summary_text.splitlines() if ln.strip()]
    header = lines[:6]

    cave = extract_section_lines(structured_summary_text, "CAVE")[:3 if short_case else 5]
    allerg = extract_section_lines(structured_summary_text, "Allergien")[:4 if short_case else 8]
    dauer = extract_section_lines(structured_summary_text, "Dauerdiagnosen")[:6 if short_case else 10]
    akut = extract_section_lines(structured_summary_text, AKUT_TITLE)
    akut_bullets = [ln for ln in akut if ln.startswith("- ")][:5 if short_case else 10]
    recur = extract_section_lines(structured_summary_text, "Wiederkehrende Probleme")[:6 if short_case else 10]
    events = extract_section_lines(structured_summary_text, "Wichtige Einzelereignisse (mit Datum)")
    event_lines = [ln for ln in events if ln.startswith("- ")]
    event_lines = event_lines[:5 if short_case else 10]

    def pack(title, items):
        out = [title + ":"]
        out.extend(items if items else ["nicht dokumentiert"])
        return out

    blocks = []
    blocks.extend(header)
    blocks.append("")
    blocks.extend(pack("CAVE", cave))
    blocks.append("")
    blocks.extend(pack("Allergien", allerg))
    blocks.append("")
    blocks.extend(pack("Dauerdiagnosen", dauer))
    blocks.append("")
    blocks.extend(pack("Akutdiagnosen Top", akut_bullets))
    blocks.append("")
    blocks.extend(pack("Wiederkehrend", recur))
    blocks.append("")
    blocks.extend(pack("Ereignisse (Auswahl)", event_lines))

    return "\n".join(blocks).strip()


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

        head_sections, akut_sorted = parse_head_sections_from_head_text(head_text)
        head_block_text = format_head_block(head_sections, akut_sorted)

        # Stage1 Kartei -> Fakten
        all_fact_lines = []
        if karte_text.strip():
            chunks = chunk_text(karte_text)
            log(f"[INFO] Kartei-Chunks: {len(chunks)}")
            tmp = []
            for ci, chunk in enumerate(chunks, 1):
                log(f"[STAGE1-KARTEI] Chunk {ci}/{len(chunks)}")
                out = call_llm(SYSTEM_PROMPT_STAGE1_KARTEI_ONLY, chunk, MAX_TOKENS_STAGE1)
                tmp.extend(parse_karte_facts_from_stage1(out))
            all_fact_lines = dedupe_preserve_order(tmp)
        else:
            log("[WARN] Kartei-Text nach Trenner ist leer -> Stage2 bekommt leeren Kartei-Block")

        # Kurzakte bestimmen
        short_case = (len(karte_text) <= SHORT_CASE_MAX_KARTE_CHARS) or (len(all_fact_lines) <= SHORT_CASE_MAX_FACT_LINES)
        log(f"[INFO] short_case={short_case} fact_lines={len(all_fact_lines)}")

        kartei_block = compact_karte_block(all_fact_lines, STAGE2_KARTE_MAX_CHARS)

        stage2_prompt = SYSTEM_PROMPT_STAGE2_KARTEI_SHORT if short_case else SYSTEM_PROMPT_STAGE2_KARTEI_LONG
        stage2_tokens = MAX_TOKENS_STAGE2_SHORT if short_case else MAX_TOKENS_STAGE2_LONG

        log("[STAGE2] Kartei-Kategorien")
        kartei_summary_raw = call_llm(stage2_prompt, kartei_block, stage2_tokens).strip()
        kartei_summary = fix_stage2_text(kartei_summary_raw).rstrip()

        structured = []
        structured.append(f"Patientennummer: {patient_info['patientennummer']}")
        structured.append(f"{patient_info['nachname']}, {patient_info['vorname']}, {patient_info['geburtsdatum']}")
        structured.append("")
        structured.append(head_block_text.rstrip())
        structured.append("")
        structured.append(kartei_summary)
        structured_summary_text = "\n".join(structured).strip() + "\n"
        structured_summary_text = enforce_patient_header(structured_summary_text, patient_info)

        # =================================================
        # Stage3 Übergaben
        # 1.5.3: Bei Kurzakte KEINE Übergaben erzeugen
        # =================================================
        handover_full_karte = None
        handover_structured = None

        if not short_case:
            if ENABLE_HANDOVER_FROM_FULL_KARTEI:
                log("[STAGE3] Übergabe aus voller Kartei (gefiltert)")
                filtered_karte = build_filtered_karte_for_handover(karte_text)

                if len(filtered_karte) > (HANDOVER_KARTEI_HEAD_CHARS + HANDOVER_KARTEI_TAIL_CHARS + 200):
                    head = filtered_karte[:HANDOVER_KARTEI_HEAD_CHARS]
                    tail = filtered_karte[-HANDOVER_KARTEI_TAIL_CHARS:]
                    filtered_karte = head + "\n...\n" + tail

                if not filtered_karte.strip():
                    handover_full_karte = "nicht dokumentiert"
                else:
                    # Langakte => LONG Prompt/Tokens
                    handover_full_karte = call_llm_with_guard(
                        SYSTEM_PROMPT_HANDOVER_FROM_FULL_KARTEI_LONG,
                        SYSTEM_PROMPT_HANDOVER_FROM_FULL_KARTEI_STRICT,
                        filtered_karte,
                        MAX_TOKENS_HANDOVER_KARTEI_LONG
                    )

            if ENABLE_HANDOVER_FROM_STRUCTURED_SUMMARY:
                log("[STAGE3] Übergabe aus strukturierter Zusammenfassung")
                handover_input = build_handover_input_from_structured(structured_summary_text, short_case=False)

                handover_structured = call_llm_with_guard(
                    SYSTEM_PROMPT_HANDOVER_FROM_STRUCTURED_SUMMARY_LONG,
                    SYSTEM_PROMPT_HANDOVER_FROM_STRUCTURED_SUMMARY_STRICT,
                    handover_input,
                    MAX_TOKENS_HANDOVER_SUMMARY_LONG
                )
        else:
            log("[STAGE3] SKIP: Kurzakte -> keine Übergaben")

        # Finaler Output
        full_out = [structured_summary_text.rstrip()]

        if not short_case:
            if ENABLE_HANDOVER_FROM_FULL_KARTEI:
                full_out.append("")
                full_out.append("---")
                full_out.append("Uebergabe (aus gesamter Kartei, natuerliche Sprache)")
                full_out.append("==========================")
                full_out.append(handover_full_karte or "nicht dokumentiert")

            if ENABLE_HANDOVER_FROM_STRUCTURED_SUMMARY:
                full_out.append("")
                full_out.append("---")
                full_out.append("Uebergabe (aus Zusammenfassung, natuerliche Sprache)")
                full_out.append("==========================")
                full_out.append(handover_structured or "nicht dokumentiert")

        summary = "\n".join(full_out).strip() + "\n"
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
            "short_case": short_case,
            "handover_full_karte_enabled": ENABLE_HANDOVER_FROM_FULL_KARTEI,
            "handover_structured_enabled": ENABLE_HANDOVER_FROM_STRUCTURED_SUMMARY,
            "handover_ran_this_file": (not short_case),
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
