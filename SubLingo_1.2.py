#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traduccion_subtitulos_ass_version4_interactivo.py
------------------------------------------------
Versión interactiva con:
1. Contexto narrativo ingresado por el usuario (mejora de estilo).
2. Detección automática de idioma + confirmación y selección de idioma objetivo.
3. "Doble‑origen": si un diálogo contiene texto original entre llaves {...} y
   una traducción intermedia (p.e. japonés + inglés), se utilizan ambos como
   entrada para Gemini, mejorando precisión y tono.
4. Interfaz enriquecida mediante Rich:
   - Listado de archivos encontrados en la carpeta y subcarpetas.
   - Barra de progreso de archivos y de líneas.
   - Paneles de estado y temporizador para back‑off.
5. Soporte por lotes (.ass y .srt).  Traduce, valida y corrige cada archivo,
   generando un informe breve al finalizar.

Requisitos:
    pip install pysubs2 rich langdetect google-generativeai

Uso:
    python3 traduccion_subtitulos_ass_version4_interactivo.py

Todos los archivos traducidos se guardan en la carpeta ./output_traducidos
con la misma estructura de carpetas que la original.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import difflib
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import pysubs2  # Manejo de .ass y .srt
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    print("Falta instalar google-generativeai → pip install google-generativeai", file=sys.stderr)
    sys.exit(1)

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# ---------- CONFIGURACIÓN --------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY", "")  # Cargar desde .env
DEFAULT_MODEL_NAME = "gemini-2.0-flash-lite"
BATCH_SIZE = 15
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5
TIME_TOLERANCE_MS = 5     # ±5 ms se considera igual
OUTPUT_ROOT = Path("output_traducidos")
OUTPUT_ROOT.mkdir(exist_ok=True)
# ---------------------------------------------------------------------------

console = Console()

# ---------- UTILIDADES -----------------------------------------------------

def find_subtitle_files(root: Path) -> List[Path]:
    """Devuelve todos los .ass y .srt dentro de root y sus subcarpetas."""
    exts = {".ass", ".srt"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def detect_language_sample(file: Path, sample_size: int = 20) -> str:
    """Intenta detectar idioma basándose en las primeras N líneas traducibles."""
    try:
        subs = pysubs2.load(file, encoding="utf-8")
    except Exception:
        return "en"

    texts = []
    for ev in subs:
        if ev.type != "Dialogue":
            continue
        txt = ev.plaintext.strip()
        if txt:
            texts.append(txt)
        if len(texts) >= sample_size:
            break
    joined = "\n".join(texts) if texts else "Hello world"
    try:
        return detect(joined)
    except LangDetectException:
        return "en"


def sleep_with_progress(seconds: int, description: str = "Espera") -> None:
    """Bloquea de forma síncrona mostrando una barrita/contador."""
    with Progress(
        SpinnerColumn(),
        TextColumn(description + " • "),
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task("sleep", total=seconds)
        while not prog.finished:
            time.sleep(0.1)
            prog.update(task, advance=0.1)

# ---------- FILTRO DE LÍNEAS A TRADUCIR ------------------------------------

def should_translate_line(ev: pysubs2.Event) -> bool:
    """Regla simple: omite líneas vacías o claramente de créditos."""
    text_lower = ev.plaintext.lower().strip()
    if not text_lower:
        return False
    CREDIT_KEYWORDS = [
        "translated by", "traducido por", "episode", "next episode", "http", "www.",
        "discord.gg", "patreon", "fansub", "special thanks", "credits", "typeset",
    ]
    return not any(k in text_lower for k in CREDIT_KEYWORDS)

# ---------- GEMINI ---------------------------------------------------------

class GeminiTranslator:
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("Se requiere API‑KEY de Gemini (env GEMINI_API_KEY).")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        console.print(Panel(f"[bold green]Proceso iniciado, Modelo usado:[/] {model_name}", expand=False))

    async def translate_batch(self, blocks: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """Traduce una lista de bloques y devuelve lista de traducciones."""
        prompt = (
            "Eres un traductor profesional. Cada bloque incluye contexto antes y "
            "después de la línea a traducir.\n"
            f"Traduce únicamente la sección marcada como [LINEA] del idioma fuente ({src_lang}) al destino ({tgt_lang}).\n"
            "Analiza primero el contexto para mantener coherencia y luego entrega solo la traducción de cada bloque,"
            " separando las traducciones con DOS saltos de línea.\n\n"
            "## Bloques:\n" + "\n\n".join(blocks)
        )
        retries = 0
        while retries < MAX_RETRIES:
            try:
                resp = await self.model.generate_content_async(
                    [prompt],
                    generation_config=genai.types.GenerationConfig(temperature=0.7),
                    safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in (
                        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
                )
                text = getattr(resp, "text", "") or "".join(p.text for p in resp.parts if hasattr(p, "text"))
                return [t.strip() for t in text.strip().split("\n\n")]
            except ResourceExhausted as e:
                retries += 1
                delay = INITIAL_BACKOFF_SECONDS * (2 ** (retries - 1))
                console.print(f"[yellow]Cuota agotada, reintento {retries}/{MAX_RETRIES} en {delay}s…[/]")
                sleep_with_progress(delay, "Back‑off Gemini")
            except Exception as e:
                console.print(f"[red]Error Gemini:[/] {e}")
                return []
        return []

# ---------- VALIDACIÓN -----------------------------------------------------

def validate_and_fix(original: pysubs2.SSAFile, translated: pysubs2.SSAFile) -> Tuple[int, int]:
    """Corrige tiempos desplazados y cuenta líneas sin traducir."""
    corr_times = 0
    retrad_needed = 0
    for ev_o, ev_t in zip(original, translated):
        if ev_o.type != "Dialogue":
            continue
        if abs(ev_o.start - ev_t.start) > TIME_TOLERANCE_MS or abs(ev_o.end - ev_t.end) > TIME_TOLERANCE_MS:
            ev_t.start = ev_o.start
            ev_t.end = ev_o.end
            corr_times += 1
        if should_translate_line(ev_o) and (not ev_t.plaintext.strip() or ev_t.plaintext == ev_o.plaintext):
            retrad_needed += 1
    return corr_times, retrad_needed

# ---------- TRADUCCIÓN DE UN ARCHIVO ---------------------------------------

async def translate_file(path: Path, translator: GeminiTranslator, src_lang: str, tgt_lang: str, progress: Progress, parent_task: TaskID):
    try:
        subs = pysubs2.load(path, encoding="utf-8")
    except Exception as e:
        console.print(f"[red]No se pudo abrir {path}: {e}")
        return

    # Mapeo de diálogos traducibles con contexto
    dialog_positions = []
    dialog_texts = []
    for idx, ev in enumerate(subs):
        if ev.type == "Dialogue":
            dialog_positions.append(idx)
            dialog_texts.append(ev.plaintext.strip())

    text_blocks = []
    indices = []
    for pos_idx, sub_idx in enumerate(dialog_positions):
        ev = subs[sub_idx]
        if not should_translate_line(ev):
            continue

        before = [dialog_texts[i] for i in range(max(0, pos_idx - 10), pos_idx)]
        after = [dialog_texts[i] for i in range(pos_idx + 1, min(len(dialog_texts), pos_idx + 11))]

        m = re.match(r"\{([^}]+)\}\s*(.*)", ev.plaintext.strip())
        if m:
            jp, en_text = m.groups()
            line = f"[ORIG]{jp}\n[TRAD_INTERMEDIA]{en_text}"
        else:
            line = ev.plaintext.strip()

        block = (
            "[ANTERIOR]\n" + "\n".join(b.replace("\n", " ") for b in before) +
            "\n[LINEA]\n" + line.replace("\n", "\\n") +
            "\n[POSTERIOR]\n" + "\n".join(a.replace("\n", " ") for a in after)
        )
        text_blocks.append(block)
        indices.append(sub_idx)

    total_blocks = len(text_blocks)
    if not total_blocks:
        console.print(f"[yellow]No hay diálogos traducibles en {path.name}.[/]")
        return

    line_task = progress.add_task("líneas", total=total_blocks, parent=parent_task)

    translations: List[str] = []
    for i in range(0, total_blocks, BATCH_SIZE):
        batch = text_blocks[i:i+BATCH_SIZE]
        translated_batch = await translator.translate_batch(batch, src_lang, tgt_lang)
        # Normalizar tamaño
        if len(translated_batch) != len(batch):
            translated_batch = [tb if k < len(translated_batch) else b for k, (tb, b) in enumerate(zip(translated_batch + batch, batch))]
        translations.extend([t.replace("\\n", "\n") for t in translated_batch])
        progress.update(line_task, advance=len(batch))
        if i + BATCH_SIZE < total_blocks:
            await asyncio.sleep(3)  # Espera proactiva entre lotes

    # Insertar en subs
    for idx, new_text in zip(indices, translations):
        subs[idx].text = new_text

    # Validar
    original = pysubs2.load(path, encoding="utf-8")
    fixed_times, missing = validate_and_fix(original, subs)

    # Guardar
    out_path = OUTPUT_ROOT / path.relative_to(Path.cwd())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Insertar el código de idioma en el nombre para que MKVToolNix lo detecte
    out_path = out_path.with_name(f"{out_path.stem}.{tgt_lang}{out_path.suffix}")
    subs.save(
        out_path,
        encoding="utf-8",
        format_="ass" if path.suffix.lower() == ".ass" else "srt",
    )

    progress.update(parent_task, advance=1)

    # Informe
    changes_panel = Panel(
        f"[bold]{path.name}[/]\n"
        f"Tiempos corregidos : [cyan]{fixed_times}[/]\n"
        f"Líneas sin traducir : [cyan]{missing}[/]",
        title="Resultado",
        border_style="blue",
        expand=False,
    )
    # console.print(changes_panel)
    progress.console.log(changes_panel)


# ---------- MAIN -----------------------------------------------------------

async def main():
    start_time = time.time()  # <-- Agrega esto
    console.print(Panel("[bold magenta]SubLingo – Traductor de subtítulos – Versión 4.1[/]", expand=False))

    # 1. Buscar archivos
    files = find_subtitle_files(Path.cwd())
    if not files:
        console.print("[red]No se encontraron archivos de subtítulos (.ass o .srt) en la carpeta actual ni en subcarpetas.[/]")
        return

    if len(files) == 1:
        console.print("\n[bold]Archivo detectado para traducir:[/]")
    else:
        console.print(f"\n[bold]{len(files)} archivos detectados para traducir:[/]")
    for idx, f in enumerate(files, 1):
        parent = "(carpeta) " if f.parent != Path.cwd() else "(archivo) "
        console.print(f"{idx}. {parent}{f.relative_to(Path.cwd())}")

    # 2. Detectar idioma base con el primer archivo
    detected_lang = detect_language_sample(files[0])
    lang_prompt = Prompt.ask(
        f"Idioma detectado en el primer archivo: [bold]{detected_lang}[/]. ¿Es correcto? (s/n o escribe el código ISO)",
        default="s"
    )
    if lang_prompt.lower() in {"s", "sí", "si", "y", "yes"}:
        src_lang = detected_lang
    else:
        src_lang = lang_prompt.lower().strip()

    # 3. Definir idioma destino
    tgt_lang = Prompt.ask("¿A qué idioma deseas traducir los subtítulos? (código ISO, ej. 'es-419')", default="es-419")

    translator = GeminiTranslator(API_KEY, DEFAULT_MODEL_NAME)

    # 4. Progreso general
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        file_task = progress.add_task("Archivos", total=len(files))
        for f in files:
            await translate_file(f, translator, src_lang, tgt_lang, progress, file_task)

    console.print(Panel("[bold green]Proceso terminado.[/]\nArchivos traducidos en ./output_traducidos", expand=False))
    elapsed = time.time() - start_time  # <-- Agrega esto
    mins, secs = divmod(int(elapsed), 60)
    console.print(f"[cyan]Tiempo de ejecución: {mins} min {secs} s[/]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Interrumpido por el usuario.[/]")
