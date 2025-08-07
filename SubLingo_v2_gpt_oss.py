#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubLingo_v2_gpt_oss.py
------------------------------------------------
Versión mejorada con optimizaciones especiales para japonés:
1. Contexto narrativo ingresado por el usuario (mejora de estilo).
2. Detección automática de idioma + confirmación y selección de idioma objetivo.
3. "Doble‑origen": si un diálogo contiene texto original entre llaves {...} y
   una traducción intermedia (p.e. japonés + inglés), se utilizan ambos como
   entrada para el modelo, mejorando precisión y tono.
4. Interfaz enriquecida mediante Rich:
   - Listado de archivos encontrados en la carpeta y subcarpetas.
   - Barra de progreso de archivos y de líneas.
   - Paneles de estado y temporizador para back‑off.
5. Soporte por lotes (.ass y .srt).  Traduce, valida y corrige cada archivo,
   generando un informe breve al finalizar.
6. MEJORAS PARA JAPONÉS:
   - Detección de caracteres japoneses (hiragana, katakana, kanji)
   - Filtros especializados que preservan expresiones cortas japonesas
   - Preprocesamiento de nombres con furigana (宇佐美(うさみ) -> 宇佐美)
   - Prompts especializados para onomatopeyas e interjecciones japonesas
   - Sistema de 3 intentos con modo agresivo para líneas problemáticas
   - Manejo específico de expresiones como あ, ん, なっ, は?, etc.

Requisitos:
    pip install pysubs2 rich langdetect cerebras-cloud-sdk python-dotenv

Uso:
    python3 SubLingo_v2_gpt_oss.py

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
load_dotenv()  # <--- Asegura que las variables del .env se cargan
from cerebras.cloud.sdk import Cerebras
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

# ---------- CONFIGURACIÓN --------------------------------------------------
API_KEY = os.getenv("CEREBRAS_API_KEY", "")  # Cargar desde .env
DEFAULT_MODEL_NAME = "gpt-oss-120b"
BATCH_SIZE = 100  # Tamaño del lote para procesamiento
MAX_RETRIES = 10  # Número máximo de reintentos
INITIAL_BACKOFF_SECONDS = 10  # Tiempo de espera inicial para reintentos
TIME_TOLERANCE_MS = 20    # ±20 ms se considera igual
BATCH_SLEEP_SECONDS = 10    # Espera entre lotes de traducción
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
    """Regla mejorada: omite líneas vacías o claramente de créditos, pero preserva expresiones japonesas."""
    text_original = ev.plaintext.strip()
    text_lower = text_original.lower()
    
    if not text_original:
        return False
    
    # Detectar si contiene caracteres japoneses
    has_japanese = any('\u3040' <= char <= '\u309F' or  # Hiragana
                      '\u30A0' <= char <= '\u30FF' or  # Katakana
                      '\u4E00' <= char <= '\u9FAF'     # Kanji
                      for char in text_original)
    
    # Si tiene japonés, ser más permisivo con líneas cortas
    if has_japanese:
        # Solo filtrar créditos obvios en japonés
        jp_credit_keywords = ["http", "www.", "discord", "patreon", "fansub", "字幕", "翻訳"]
        return not any(k in text_lower for k in jp_credit_keywords)
    
    # Para otros idiomas, usar filtros normales
    CREDIT_KEYWORDS = [
        "translated by", "traducido por", "episode", "next episode", "http", "www.",
        "discord.gg", "patreon", "fansub", "special thanks", "credits", "typeset",
    ]
    return not any(k in text_lower for k in CREDIT_KEYWORDS)

# ---------- CEREBRAS ------------------------------------------------------

class CerebrasTranslator:
    def __init__(self, api_key: str, model_name: str, narrative_ctx: str):
        if not api_key:
            raise ValueError("Se requiere API‑KEY de Cerebras (env CEREBRAS_API_KEY).")
        self.client = Cerebras(api_key=api_key)
        self.model_name = model_name
        self.narrative_ctx = narrative_ctx
        console.print(Panel(f"[bold green]Proceso iniciado, Modelo usado:[/] {model_name}", expand=False))


    async def translate_batch(self, blocks: List[str], src_lang: str, tgt_lang: str, aggressive_mode: bool = False) -> List[str]:
        import asyncio
        
        # Prompt especializado para japonés
        if src_lang == "ja" or "jp" in src_lang.lower():
            if aggressive_mode:
                prompt = (
                    f"TRADUCTOR ESPECIALIZADO EN JAPONÉS. Serie: {self.narrative_ctx}\n\n"
                    f"INSTRUCCIONES CRÍTICAS:\n"
                    f"1. DEBES traducir TODO el texto japonés al {tgt_lang}, sin excepción\n"
                    f"2. Texto marcado con [JP] es japonés puro que DEBE traducirse completamente\n"
                    f"3. Onomatopeyas japonesas (あ, ん, なっ, etc.) = expresiones emocionales en {tgt_lang}\n"
                    f"4. Interjecciones cortas (は?, ん?) = equivalentes naturales en {tgt_lang}\n"
                    f"5. Frases incompletas con '...' = traduce el contenido y mantén los puntos suspensivos\n"
                    f"6. NUNCA dejes texto sin traducir\n"
                    f"7. Ignora el marcador [JP] en tu respuesta, solo traduce el contenido\n\n"
                    f"EJEMPLOS:\n"
                    f"- '[JP]あ......' → 'Ah...'\n"
                    f"- '[JP]なっ......' → 'Qué...'\n"
                    f"- '[JP]は?' → '¿Eh?'\n"
                    f"- '[JP]ん?' → '¿Mm?'\n\n"
                    f"TEXTO A TRADUCIR:\n" + "\n\n".join(blocks)
                )
            else:
                prompt = (
                    f"Traductor especializado en japonés. Serie: {self.narrative_ctx}\n\n"
                    f"Traduce del japonés al {tgt_lang}:\n"
                    f"- Texto marcado con [JP] es japonés puro que DEBE traducirse\n"
                    f"- Onomatopeyas y expresiones emocionales japonesas deben traducirse naturalmente\n"
                    f"- Mantén el tono y estilo emocional\n"
                    f"- Traduce interjecciones cortas (あ, ん, は, etc.) a equivalentes naturales\n"
                    f"- Ignora el marcador [JP] en tu respuesta, solo traduce el contenido\n\n"
                    f"Texto:\n" + "\n\n".join(blocks)
                )
        else:
            prompt = (
                f"Eres un traductor profesional. El texto forma parte de una serie descrita así: \n"
                f"'''{self.narrative_ctx}'''\n\n"
                f"Tareas: \n1. Traduce fielmente del idioma fuente ({src_lang}) al destino ({tgt_lang})\n"
                "2. Respeta estilo, tono y emociones.\n"
                "3. Devuelve sólo la traducción de cada bloque; separa bloques con DOS saltos de línea.\n\n"
                "## Bloques:\n" + "\n\n".join(blocks)
            )
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=True,
                max_completion_tokens=65_536,
                temperature=0.2,
                top_p=1, 
                reasoning_effort="high"
                # reasoning_effort="high"## Opcional: "low", "medium", "high"
            )
            output = ""
            # Cerebras SDK no es async, así que usamos run_in_executor
            loop = asyncio.get_event_loop()
            def get_stream():
                return [chunk.choices[0].delta.content or "" for chunk in stream]
            chunks = await loop.run_in_executor(None, get_stream)
            output = "".join(chunks)
            # Limpiar marcadores [JP] de las traducciones
            translations = [t.strip() for t in output.strip().split("\n\n")]
            return [t.replace("[JP]", "").strip() for t in translations]
        except Exception as e:
            console.print(f"[red]Error Cerebras:[/] {e}")
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

# ---------- VALIDACIÓN ESTRUCTURAL LIGERA ----------------------------------
import random  # <-- encabezado del archivo

def sample_structure_check(
        original: pysubs2.SSAFile,
        translated: pysubs2.SSAFile,
        samples: int = 10,
        tol_ms: int = TIME_TOLERANCE_MS) -> list[str]:
    """
    Revisa que la estructura general se conserve tomando 'samples' checkpoints.
    Devuelve una lista de mensajes de error (vacía = todo ok).
    """
    issues = []

    # 0. Conteo de eventos
    if len(original) != len(translated):
        issues.append(f"Cantidad de eventos distinta: "
                      f"orig={len(original)} vs trad={len(translated)}")

    # 1. Muestreo cada ~10 %
    step = max(len(original) // samples, 1)
    for idx in range(0, len(original), step):
        if idx >= len(translated):
            issues.append(f"Bloque {idx+1} falta en traducido.")
            continue

        o, t = original[idx], translated[idx]

        # 1a. Tiempos
        if abs(o.start - t.start) > tol_ms or abs(o.end - t.end) > tol_ms:
            issues.append(f"Tiempos desfasados en bloque {idx+1} "
                          f"({o.start}->{t.start} ms)")

        # 1b. Semejanza de texto
        ratio = difflib.SequenceMatcher(None,
                                        o.plaintext.strip(),
                                        t.plaintext.strip()).ratio()
        if ratio < 0.30:  # muy diferente
            issues.append(f"Contenido muy diferente en bloque {idx+1} "
                          f"(sim={ratio:.2f})")

    return issues


# ---------- TRADUCCIÓN DE UN ARCHIVO ---------------------------------------

async def translate_file(path: Path, translator: CerebrasTranslator, src_lang: str, tgt_lang: str, progress: Progress, parent_task: TaskID):
    try:
        subs = pysubs2.load(path, encoding="utf-8")
    except Exception as e:
        console.print(f"[red]No se pudo abrir {path}: {e}")
        return

    # Mapeo de diálogos traducibles
    text_blocks = []
    indices = []
    for i, ev in enumerate(subs):
        if ev.type != "Dialogue" or not should_translate_line(ev):
            continue
        
        text = ev.plaintext.strip()
        
        # Doble‑origen:   {JP} EN
        m = re.match(r"\{([^}]+)\}\s*(.*)", text)
        if m:
            jp, en_text = m.groups()
            combo = f"[ORIG]{jp}\n[TRAD_INTERMEDIA]{en_text}"
            text_blocks.append(combo.replace("\n", "\\n"))
        else:
            # Preprocesamiento especial para japonés
            if src_lang == "ja" or "jp" in src_lang.lower():
                # Simplificar nombres con furigana: 宇佐美(うさみ) -> 宇佐美
                text = re.sub(r'([\u4E00-\u9FAF]+)\([\u3040-\u309F]+\)', r'\1', text)
                # Marcar texto como japonés para el traductor
                text = f"[JP]{text}"
            
            text_blocks.append(text.replace("\n", "\\n"))
        indices.append(i)

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
            await asyncio.sleep(BATCH_SLEEP_SECONDS)  # Espera proactiva entre lotes

    # Insertar en subs
    for idx, new_text in zip(indices, translations):
        subs[idx].text = new_text

    # Validar
    original = pysubs2.load(path, encoding="utf-8")
    fixed_times, missing = validate_and_fix(original, subs)
    # Chequeo estadístico de estructura
    structural_issues = sample_structure_check(original, subs)
    if structural_issues:
        panel = Panel(
            "\n".join(structural_issues),
            title="Chequeo de estructura (10 %)",
            border_style="yellow",
            expand=False,
        )
        progress.console.log(panel)
        # Si quieres detener el proceso automáticamente:
        # raise ValueError("Estructura incoherente; abortando guardado.")


    # --- Nueva ronda para líneas no traducidas ---
    if missing > 0:
        untranslated_indices = []
        untranslated_blocks = []
        for i, (ev_o, ev_t) in enumerate(zip(original, subs)):
            if ev_o.type != "Dialogue":
                continue
            if should_translate_line(ev_o) and (not ev_t.plaintext.strip() or ev_t.plaintext == ev_o.plaintext):
                untranslated_indices.append(i)
                untranslated_blocks.append(ev_o.plaintext.strip().replace("\n", "\\n"))
        if untranslated_blocks:
            console.print(f"[yellow]Segundo intento: traduciendo {len(untranslated_blocks)} líneas faltantes en {path.name}...[/]")
            # Traducir en lotes si son muchas
            for i in range(0, len(untranslated_blocks), BATCH_SIZE):
                batch = untranslated_blocks[i:i+BATCH_SIZE]
                translated_batch = await translator.translate_batch(batch, src_lang, tgt_lang)
                if len(translated_batch) != len(batch):
                    translated_batch = [tb if k < len(translated_batch) else b for k, (tb, b) in enumerate(zip(translated_batch + batch, batch))]
                for idx, new_text in zip(untranslated_indices[i:i+BATCH_SIZE], translated_batch):
                    subs[idx].text = new_text.replace("\\n", "\n")
                if i + BATCH_SIZE < len(untranslated_blocks):
                    await asyncio.sleep(BATCH_SLEEP_SECONDS)
            # Revalidar
            fixed_times2, missing2 = validate_and_fix(original, subs)
            fixed_times += fixed_times2
            missing = missing2
            
            # --- Tercer intento con modo agresivo para japonés ---
            if missing2 > 0 and (src_lang == "ja" or "jp" in src_lang.lower()):
                final_untranslated_indices = []
                final_untranslated_blocks = []
                for i, (ev_o, ev_t) in enumerate(zip(original, subs)):
                    if ev_o.type != "Dialogue":
                        continue
                    if should_translate_line(ev_o) and (not ev_t.plaintext.strip() or ev_t.plaintext == ev_o.plaintext):
                        final_untranslated_indices.append(i)
                        final_untranslated_blocks.append(ev_o.plaintext.strip().replace("\n", "\\n"))
                if final_untranslated_blocks:
                    console.print(f"[red]Tercer intento AGRESIVO para japonés: {len(final_untranslated_blocks)} líneas restantes...[/]")
                    # Usar modo agresivo específico para japonés
                    for i in range(0, len(final_untranslated_blocks), BATCH_SIZE):
                        batch = final_untranslated_blocks[i:i+BATCH_SIZE]
                        translated_batch = await translator.translate_batch(batch, src_lang, tgt_lang, aggressive_mode=True)
                        if len(translated_batch) != len(batch):
                            translated_batch = [tb if k < len(translated_batch) else b for k, (tb, b) in enumerate(zip(translated_batch + batch, batch))]
                        for idx, new_text in zip(final_untranslated_indices[i:i+BATCH_SIZE], translated_batch):
                            subs[idx].text = new_text.replace("\\n", "\n")
                        if i + BATCH_SIZE < len(final_untranslated_blocks):
                            await asyncio.sleep(BATCH_SLEEP_SECONDS)
                    # Revalidar final
                    fixed_times3, missing3 = validate_and_fix(original, subs)
                    fixed_times += fixed_times3
                    missing = missing3

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
    progress.console.log(changes_panel)

    # --- Resumen de líneas no traducidas ---
    if missing > 0:
        untranslated_summary = []
        for i, (ev_o, ev_t) in enumerate(zip(original, subs)):
            if ev_o.type != "Dialogue":
                continue
            if should_translate_line(ev_o) and (not ev_t.plaintext.strip() or ev_t.plaintext == ev_o.plaintext):
                motivo = "Posible canción/segmento no traducible" if any(word in ev_o.plaintext.lower() for word in ["♪", "♫", "music", "musica", "歌", "カラオケ"]) else "No traducido tras dos intentos"
                untranslated_summary.append(f"Línea {i+1}: '{ev_o.plaintext.strip()[:60]}...'  [Motivo: {motivo}]")
        if untranslated_summary:
            resumen_panel = Panel(
                "\n".join(untranslated_summary),
                title=f"Líneas no traducidas en {path.name}",
                border_style="red",
                expand=False,
            )
            progress.console.log(resumen_panel)

    # --- Resumen de líneas no traducidas con explicación del modelo ---
    if missing > 0:
        untranslated_summary = []
        for i, (ev_o, ev_t) in enumerate(zip(original, subs)):
            if ev_o.type != "Dialogue":
                continue
            if should_translate_line(ev_o) and (not ev_t.plaintext.strip() or ev_t.plaintext == ev_o.plaintext):
                # Pedir explicación al modelo
                prompt_explain = (
                    f"A continuación se muestra una línea de subtítulo que no fue traducida: \n"
                    f"'{ev_o.plaintext.strip()}'\n"
                    "¿Por qué no se debe traducir esta línea? Si debe traducirse, responde: 'Debe traducirse'. Si no, explica brevemente el motivo (por ejemplo: es una canción, onomatopeya, nombre propio, etc.)."
                )
                explicacion = await translator.translate_batch([prompt_explain], src_lang, tgt_lang)
                explicacion_str = explicacion[0] if explicacion else "Sin explicación del modelo"
                untranslated_summary.append(f"Línea {i+1}: '{ev_o.plaintext.strip()[:60]}...'\n[Modelo]: {explicacion_str}")
        if untranslated_summary:
            resumen_panel = Panel(
                "\n\n".join(untranslated_summary),
                title=f"Líneas no traducidas en {path.name}",
                border_style="red",
                expand=False,
            )
            progress.console.log(resumen_panel)
# ---------- MAIN -----------------------------------------------------------

async def main():
    start_time = time.time()
    console.print(Panel("[bold magenta]SubLingo – Traductor de subtítulos – Versión 4.1[/]", expand=False))

    # --- Selección de modo ---
    modo_auto = Confirm.ask("¿Deseas usar el modo automático? (acepta todos los valores por defecto)", default=False)

    if modo_auto:
        narrative_ctx = "Una historia con diálogos naturales y estilo narrativo neutral."
        files = find_subtitle_files(Path.cwd())
        if not files:
            console.print("[red]No se encontraron archivos de subtítulos (.ass o .srt) en la carpeta actual ni en subcarpetas.[/]")
            return
        src_lang = detect_language_sample(files[0])
        tgt_lang = "es-419"
        API_KEY = os.getenv("CEREBRAS_API_KEY", "")
        DEFAULT_MODEL_NAME = "gpt-oss-120b"
        translator = CerebrasTranslator(API_KEY, DEFAULT_MODEL_NAME, narrative_ctx)
    else:
        # 1. Contexto narrativo
        narrative_ctx = Prompt.ask(
            "Describe brevemente la obra (serie, película, etc.) y su tono general",
            default="Una historia con diálogos naturales y estilo narrativo neutral."
        )
        # 2. Buscar archivos
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
        # 3. Detectar idioma base con el primer archivo
        detected_lang = detect_language_sample(files[0])
        lang_prompt = Prompt.ask(
            f"Idioma detectado en el primer archivo: [bold]{detected_lang}[/]. ¿Es correcto? (s/n o escribe el código ISO)",
            default="s"
        )
        if lang_prompt.lower() in {"s", "sí", "si", "y", "yes"}:
            src_lang = detected_lang
        else:
            src_lang = lang_prompt.lower().strip()
        tgt_lang = Prompt.ask("¿A qué idioma deseas traducir los subtítulos? (código ISO, ej. 'es-419')", default="es-419")
        API_KEY = os.getenv("CEREBRAS_API_KEY", "")
        DEFAULT_MODEL_NAME = "gpt-oss-120b"
        translator = CerebrasTranslator(API_KEY, DEFAULT_MODEL_NAME, narrative_ctx)

    # 5. Progreso general
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
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    console.print(f"[cyan]Tiempo de ejecución: {mins} min {secs} s[/]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Interrumpido por el usuario.[/]")
