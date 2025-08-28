#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SubLingo – Traductor de subtítulos – Versión 5.0 (Agentes)
-----------------------------------------------------------
Refactor compacto con dos agentes:
- TranslatorAgent: traduce por lotes con alineación por IDs [[n]] y reglas estrictas.
- ValidatorAgent: valida la salida, detecta huecos/idénticos y dispara una segunda pasada agresiva.

Objetivo: mismo flujo que la versión anterior, menos líneas y más robustez.

Requisitos:
- pip install cerebras-cloud-sdk pysubs2 rich python-dotenv (opcional)
- export/set CEREBRAS_API_KEY

Uso:
    python SubLingo_v3_agents.py
"""
from __future__ import annotations
import os, re, sys, asyncio
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
try:
    from cerebras.cloud.sdk import Cerebras
except Exception as e:  # Mensaje claro si falta SDK
    raise SystemExit("[ERROR] Instala 'cerebras-cloud-sdk' (pip install cerebras-cloud-sdk)")
try:
    import pysubs2
except Exception:
    raise SystemExit("[ERROR] Instala 'pysubs2' (pip install pysubs2)")

# --- Configuración ---
MODEL_NAME   = os.getenv("SUBLINGO_MODEL", "gpt-oss-120b")
BATCH_SIZE   = int(os.getenv("SUBLINGO_BATCH", 120))
SLEEP        = float(os.getenv("SUBLINGO_SLEEP", 0.15))
OUTPUT_ROOT  = Path(os.getenv("SUBLINGO_OUT", "output_traducidos")).resolve()
SHOW_HEADER  = True

# Cargar .env si existe (sin dependencia obligatoria)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # Fallback simple .env
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
console = Console()

# --- Timer util ---
import time
start_time = time.perf_counter

def fmt_duration(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


# --------------------------- Utilidades CLI ---------------------------
def ask(msg: str, default: str|None=None) -> str:
    prompt = f"{msg} " + (f"({default}): " if default is not None else ": ")
    ans = input(prompt).strip()
    return ans or (default or "")

def banner(title: str):
    if not SHOW_HEADER: return
    console.print(Panel.fit(title, border_style="green"))

# Detecta idioma básico (ja, en, es) sin dependencias
def quick_lang_detect(text: str) -> str:
    if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF' for c in text):
        return "ja"
    # Heurística simple: contiene muchas palabras comunes del inglés
    en_tokens = sum(w in {"the","and","you","what","is","are","not","i","it","to"} for w in re.findall(r"[a-zA-Z']+", text.lower()))
    return "en" if en_tokens >= 1 else "es-419"

# Selecciona líneas a traducir (evita URLs/creditos)
def should_translate(text: str) -> bool:
    t = text.strip()
    if not t: return False
    low = t.lower()
    skip = ("http" in low) or ("www." in low) or ("discord" in low) or ("patreon" in low) or ("credits" in low) or ("special thanks" in low) or ("@" in low) or any(x in low for x in [".com",".net",".org"])
    return not skip

# -------------------------- Cliente LLM --------------------------
@dataclass
class LLM:
    api_key: str
    model: str
    def __post_init__(self):
        key = self.api_key or os.getenv("CEREBRAS_API_KEY", "")
        if not key:
            key = os.getenv("CEREBRAS_API_KEY", "")
        if not key:
            console.print("[yellow]No encontré CEREBRAS_API_KEY. Puedes pegarla ahora (solo para esta ejecución).[/]")
            try:
                key = ask("Clave CEREBRAS_API_KEY", "")
            except Exception:
                key = ""
            if key:
                os.environ["CEREBRAS_API_KEY"] = key
        if not key:
            msg = (
                "[ERROR] Falta CEREBRAS_API_KEY. Opciones:\n"
                " - Crea .env junto al script: CEREBRAS_API_KEY=sk-XXX\n"
                " - En PowerShell (solo sesión actual): $env:CEREBRAS_API_KEY=\"sk-XXX\" ; python SubLingo_v3_agents.py\n"
                " - Persistente (nueva terminal): setx CEREBRAS_API_KEY \"sk-XXX\"\n"
            )
            raise SystemExit(msg)
        self.api_key = key
        self.client = Cerebras(api_key=self.api_key)

    async def complete(self, prompt: str, temperature: float=0.2, max_tokens: int=4096) -> str:
        # Config dinámica para GPT-OSS vs otros modelos
        model_lower = (self.model or "").lower()
        env_max = os.getenv("SUBLINGO_MAX_TOKENS")
        max_comp = int(env_max) if env_max else (65536 if "gpt-oss" in model_lower else max_tokens)
        reasoning = os.getenv("SUBLINGO_REASONING", "medium") if "gpt-oss" in model_lower else None

        def collect():
            chunks = []
            kwargs = dict(
                messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                model=self.model,
                stream=True,
                temperature=temperature,
                max_completion_tokens=max_comp,
                top_p=1,
            )
            if reasoning is not None:
                kwargs["reasoning_effort"] = reasoning
            try:
                iterator = self.client.chat.completions.create(**kwargs)
            except Exception:
                # Reintento sin reasoning_effort por compatibilidad retro
                kwargs.pop("reasoning_effort", None)
                iterator = self.client.chat.completions.create(**kwargs)

            for ch in iterator:
                d = ch.choices[0].delta.content
                if d:
                    chunks.append(d)
            return "".join(chunks)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, collect)

@dataclass
class TranslatorAgent:
    llm: LLM
    narrative_ctx: str
    def _rules(self, src: str, tgt: str, aggressive: bool) -> str:
        base = f'''Eres un traductor profesional. Contexto: {self.narrative_ctx}
    Idioma fuente: {src} → destino: {tgt}

    REGLAS:
    1) TRADUCE TODO, sin excepciones (onomatopeyas y rótulos incluidos).
    2) Si ya está en destino, reescribe (paráfrasis breve).
    3) Conserva saltos de línea y tono.
    4) Responde con una línea por bloque: [[id]] <traducción>
    5) No inventes ni borres IDs.
    '''
        if aggressive:
            base += '''MODO AGRESIVO: nunca dejes texto idéntico al original;
    interjecciones tipo 'Huh?', 'Eek...', y rótulos como 'INTERNET & COMIC CAFE' se traducen.
    '''
        if src == "ja":
            base += "Para japonés: convierte onomatopeyas a equivalentes naturales en el destino.\n"
        return base

    async def translate(self, blocks: List[str], src: str, tgt: str, aggressive: bool=False) -> List[str]:
        numbered = [f"[[{i}]] {b}" for i, b in enumerate(blocks, 1)]

        prompt = self._rules(src, tgt, aggressive) + f'''\nBLOQUES:
    {"\n".join(numbered)}

    Responde con [[id]] <texto>.'''

        out = await self.llm.complete(prompt)

        # Regex robusto: raw string + multiline + dotall
        patt = re.compile(r'^\s*\[\[(\d+)\]\]\s*(.*?)(?=\n\[\[|\Z)', re.M | re.S)
        mapping = {int(m.group(1)): m.group(2).strip() for m in patt.finditer(out)}
        return [mapping.get(i, "").strip() for i in range(1, len(blocks) + 1)]


@dataclass
class ValidatorAgent:
    translator: TranslatorAgent
    def _needs_fix(self, src: str, out: str) -> bool:
        if not should_translate(src):
            return False
        s1, s2 = src.strip(), out.strip()
        if not s2: return True
        # igual salvo puntuación/espacios
        norm = lambda s: re.sub(r"\W+","", s).lower()
        if norm(s1) == norm(s2): return True
        return False
    def find_issues(self, src_blocks: List[str], out_blocks: List[str]) -> Tuple[List[int], List[str]]:
        idxs, blocks = [], []
        for i,(s,o) in enumerate(zip(src_blocks,out_blocks)):
            if self._needs_fix(s,o): idxs.append(i); blocks.append(s)
        return idxs, blocks
    async def repair(self, src_blocks: List[str], out_blocks: List[str], src: str, tgt: str) -> List[str]:
        idxs, blocks = self.find_issues(src_blocks, out_blocks)
        if not blocks: return out_blocks
        fix = []
        for i in range(0, len(blocks), BATCH_SIZE):
            part = blocks[i:i+BATCH_SIZE]
            fixed = await self.translator.translate(part, src, tgt, aggressive=True)
            fix.extend(fixed)
            if i + BATCH_SIZE < len(blocks):
                await asyncio.sleep(SLEEP)
        for j, val in zip(idxs, fix):
            out_blocks[j] = val.strip()
        return out_blocks

# -------------------------- Núcleo de proceso --------------------------
async def process_subtitle(path: Path, llm: LLM, src_lang: str, tgt_lang: str, narrative: str):
    subs = pysubs2.load(path, encoding="utf-8")
    events = [ev for ev in subs if ev.type == "Dialogue" and should_translate(ev.plaintext)]
    if not events:
        console.print(f"[yellow]No hay diálogos traducibles en {path.name}[/]")
        return 0, 0

    # Bloques de texto de origen
    src_blocks, assembly = [], []  # src_blocks: sublíneas; assembly: [(idx_evento, num_sublíneas, prefijos)]
    for i, ev in enumerate(subs):
        if ev in events:
            raw = ev.plaintext.replace("\r\n", "\n").replace("\r", "\n").strip()
            parts = raw.split("\n")

            prefixes = []   # p.ej. ["- ", "", "- "]
            clean = []      # sublíneas sin el prefijo (para que el modelo no lo borre)
            for p in parts:
                m = re.match(r'^(\s*[-–—•]\s+)?(.*)$', p)
                prefixes.append(m.group(1) or "")
                clean.append(m.group(2))
            assembly.append((i, len(parts), prefixes))
            src_blocks.extend(clean)


    banner(f"Proceso iniciado, Modelo usado: {MODEL_NAME}")
    translator = TranslatorAgent(LLM(os.getenv("CEREBRAS_API_KEY"), MODEL_NAME), narrative)
    validator  = ValidatorAgent(translator)

    # Pasada 1: traducción normal
    out_blocks: List[str] = []
    with Progress(TextColumn("[bold]lotes[/]"), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as bar:
        t = bar.add_task("traduciendo", total=len(src_blocks))
        for i in range(0, len(src_blocks), BATCH_SIZE):
            batch = src_blocks[i:i+BATCH_SIZE]
            out = await translator.translate(batch, src_lang, tgt_lang, aggressive=False)
            out_blocks.extend(out)
            bar.update(t, advance=len(batch))
            if i + BATCH_SIZE < len(src_blocks): await asyncio.sleep(SLEEP)

    # Pasada 2: reparación por el validador
    out_blocks = await validator.repair(src_blocks, out_blocks, src_lang, tgt_lang)

    # Escribir back  (MULTILINE-FIX: rearmar eventos multilinea)
    missing = 0
    pos = 0
    for idx_event, count, prefixes in assembly:
        lines = []
        for j in range(count):
            out_line = (out_blocks[pos + j] if pos + j < len(out_blocks) else "").strip()
            src_line = (src_blocks[pos + j] if pos + j < len(src_blocks) else "").strip()

            if not out_line:
                # cuenta como faltante y caemos al original limpio (sublínea)
                missing += 1
                out_line = src_line

            # Si la original tenía guion/prefijo y la traducción no, lo restauramos
            if prefixes[j] and not re.match(r'^\s*[-–—•]\s+', out_line):
                out_line = prefixes[j] + out_line

            lines.append(out_line)
        pos += count

        # Unir con salto de línea real dentro del mismo evento SRT
        subs[idx_event].text = "\n".join(lines)

    # Guardar
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / (path.stem + f".{tgt_lang}" + path.suffix)
    subs.save(out_path, encoding="utf-8", format_="srt")

    # Reporte
    console.log(Panel(
        f"[bold]{path.name}[/]\nTiempos corregidos : [cyan]0[/]\nLíneas sin traducir : [cyan]{missing}[/]",
        title="Resultado", border_style="blue", expand=False
    ))
    return 0, missing

# ------------------------------ Flujo CLI ------------------------------
def gather_files(cwd: Path) -> List[Path]:
    return sorted([p for p in cwd.glob("*.srt")] + [p for p in cwd.glob("*.ass")])

async def main():
    console.print(Panel.fit("SubLingo – Traductor de subtítulos – Versión 5.0", border_style="cyan"))

    auto = ask("¿Deseas usar el modo automático? (acepta todos los valores por defecto) [y/n]", "n").lower().startswith("y")
    narrative = "Una historia con diálogos naturales y estilo narrativo neutral."
    if not auto:
        narrative = ask("Describe brevemente la obra (serie, película, etc.) y su tono general", narrative)

    files = gather_files(Path.cwd())
    if not files:
        console.print("[red]No se hallaron archivos .srt/.ass en la carpeta actual.[/]")
        return

    console.print("\n[bold]Archivo(s) detectado(s) para traducir:[/]")
    for i, f in enumerate(files, 1):
        console.print(f"{i}. {f.name}")


    # Detección de idioma fuente
    sample_text = "".join([ev.plaintext for ev in pysubs2.load(files[0], encoding="utf-8") if ev.type=="Dialogue"])[:2000]
    src_guess = quick_lang_detect(sample_text)
    src_ans = ask(f"Idioma detectado en el primer archivo: {src_guess}. ¿Es correcto? (s/n o escribe el código ISO)", "s")
    if src_ans.lower() not in {"s","si","sí","y","yes"}:
        src_lang = src_ans.strip() or src_guess
    else:
        src_lang = src_guess

    tgt_lang = ask("¿A qué idioma deseas traducir los subtítulos? (código ISO, ej. 'es-419')", "es-419").strip()

    # console.print(Panel.fit(f"Proceso iniciado, Modelo usado: {MODEL_NAME}", border_style="magenta"))

    total_t0 = start_time()
    llm = LLM(os.getenv("CEREBRAS_API_KEY"), MODEL_NAME)
    for f in files:
        t0 = start_time()
        await process_subtitle(f, llm, src_lang, tgt_lang, narrative)
        dt = start_time() - t0
        console.print(f"[bold cyan]⏱ Tiempo {f.name}:[/] [green]{fmt_duration(dt)}[/]")

    dt_total = start_time() - total_t0
    console.print(f"[bold green]Trabajo finalizado[/]  [dim]| total:[/] [bold magenta]{fmt_duration(dt_total)}[/]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelado por el usuario.[/]")
