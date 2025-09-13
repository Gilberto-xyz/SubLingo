#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sublingo Ultra – Traductor de subtítulos unificado
--------------------------------------------------
Unifica las variantes Kimi (Groq), GPT-OSS (Cerebras) y Llama (Cerebras)
en un solo programa con selección de modelo/proveedor y modo de vista
de proceso opcional.

Características:
- Soporta SRT y ASS (conserva tags y prefijos de diálogo).
- Selección de modelo por alias: kimi, gpt-oss, llama31-8b, llama-scout.
- Proveedor automático (Groq/Cerebras) según el modelo elegido.
- Razonamiento opcional (Cerebras) vía --razonamiento o env SUBLINGO_REASONING.
- Vista de proceso (previews) con --p / --proceso / --t (desactivado por defecto).

Requisitos (según proveedor usado):
- Groq: pip install groq ; export/set GROQ_API_KEY
- Cerebras: pip install cerebras-cloud-sdk ; export/set CEREBRAS_API_KEY
- Común: pip install pysubs2 rich python-dotenv (opcional .env)

Ejemplos:
  python Sublingo_Ultra.py -m kimi --tgt es-419 --p
  python Sublingo_Ultra.py -m gpt-oss --razonamiento none --tgt es-419
  python Sublingo_Ultra.py -m llama31-8b --tgt pt-BR --dir SUBS_BULK

"""
from __future__ import annotations
import os, re, sys, asyncio, argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Cargar .env si existe (sin dependencia obligatoria)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
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


# --------------------------- Utilidades generales ---------------------------
class QuotaExceededError(Exception):
    pass

import time
start_time = time.perf_counter

def fmt_duration(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"

def trunc(s: str, n: int = 80) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return (s[: n - 1] + "…") if len(s) > n else s

def color_lang(lang: str, style: str = "bold magenta") -> str:
    return f"[{style}]{lang}[/]"

def ask(msg: str, default: str|None=None) -> str:
    prompt = f"{msg} " + (f"({default}): " if default is not None else ": ")
    console.print(prompt, end="")
    try:
        ans = input().strip()
    except EOFError:
        ans = ""
    return ans or (default or "")

def banner(title: str, enabled: bool=True):
    if not enabled: return
    console.print(Panel.fit(title, border_style="green"))

def quick_lang_detect(text: str) -> str:
    if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF' for c in text):
        return "ja"
    en_tokens = sum(w in {"the","and","you","what","is","are","not","i","it","to"} for w in re.findall(r"[a-zA-Z']+", text.lower()))
    return "en" if en_tokens >= 1 else "es-419"

def should_translate(text: str) -> bool:
    t = text.strip()
    if not t: return False
    low = t.lower()
    skip = ("http" in low) or ("www." in low) or ("discord" in low) or ("patreon" in low) or ("credits" in low) or ("special thanks" in low) or ("@" in low) or any(x in low for x in [".com",".net",".org"])
    return not skip


# --------------------------- Normalización ASS ---------------------------
def preserve_edge_spaces(original: str, translated: str) -> str:
    """Mantiene los espacios de borde del texto original en la traducción.
    No toca los espacios internos; solo replica los del inicio/fin del segmento.
    """
    lead = re.match(r"^\s*", original).group(0)
    tail = re.search(r"\s*$", original).group(0)
    core = translated.strip()
    return f"{lead}{core}{tail}"

def fix_ass_italic_spacing(line: str) -> str:
    r"""Asegura espacios visuales alrededor de bloques en cursiva {\i1}...{\i0}.
    - Inserta espacio antes de {\i1} solo si lo antecede un carácter de palabra.
    - Inserta espacio después de {\i0} solo si lo sigue un carácter de palabra.
    Evita introducir espacio antes de signos de puntuación.
    """
    # Antes de apertura de itálica: solo si hay una letra/dígito antes
    line = re.sub(r"(?<=\w)(\{\\i1\})", r" \1", line)
    # Después del cierre de itálica: solo si hay letra/dígito después
    line = re.sub(r"(\{\\i0\})(?=\w)", r"\1 ", line)
    return line

def fix_es_ass_idioms(line: str) -> str:
    r"""Correcciones específicas al español para malas traducciones literales comunes.
    Caso principal: "smooth operator" → "{\i1}verdadero seductor{\i0}" (o "verdadera seductora").
    Soporta variantes con/sin espacios indebidos alrededor de las etiquetas.
    """
    def repl_smooth(m: re.Match) -> str:
        art = (m.group('art') or '').strip()
        tail = m.group('tail') or ''
        fem = bool(re.search(r"\b(una|toda\s+una)\b", art, re.I))
        frase = 'verdadera seductora' if fem else 'verdadero seductor'
        return f"{art} {{\\i1}}{frase}{{\\i0}}{tail}"

    # un{\i1}suave{\i0}operador → un {\i1}verdadero seductor{\i0}
    patt = re.compile(r"(?P<art>\b(?:todo\s+un|toda\s+una|un|una))\s*\{\\i1\}\s*suave\s*\{\\i0\}\s*operador(a)?(?P<tail>\b|\s|\W)", re.I)
    line = patt.sub(repl_smooth, line)
    return line


# --------------------------- Cliente LLM unificado ---------------------------
SUPPORTED_ALIASES: Dict[str, Dict[str, str]] = {
    # alias: {provider, model}
    "kimi":      {"provider": "groq",     "model": "moonshotai/kimi-k2-instruct"},
    "gpt-oss":   {"provider": "cerebras", "model": "gpt-oss-120b"},
    # Llama 3.1 8B (Cerebras)
    "llama31-8b": {"provider": "cerebras", "model": "llama-3.1-8b-instruct"},
    # Compatibilidad con la versión previa de llama
    "llama-scout": {"provider": "cerebras", "model": "llama-4-scout-17b-16e-instruct"},
}

@dataclass
class LLMClient:
    provider: str               # "groq" | "cerebras"
    model: str
    api_key: Optional[str] = None
    reasoning: Optional[str] = None  # none|low|medium|high (solo cerebras si el modelo soporta)

    def _ensure_key(self):
        if self.provider == "groq":
            key_env = "GROQ_API_KEY"
            missing_msg = (
                "[ERROR] Falta GROQ_API_KEY. Opciones:\n"
                " - Crea .env: GROQ_API_KEY=gsk-XXX\n"
                " - PowerShell (sesión actual): $env:GROQ_API_KEY=\"gsk-XXX\"\n"
                " - Persistente: setx GROQ_API_KEY \"gsk-XXX\""
            )
        else:
            key_env = "CEREBRAS_API_KEY"
            missing_msg = (
                "[ERROR] Falta CEREBRAS_API_KEY. Opciones:\n"
                " - Crea .env: CEREBRAS_API_KEY=sk-XXX\n"
                " - PowerShell (sesión actual): $env:CEREBRAS_API_KEY=\"sk-XXX\"\n"
                " - Persistente: setx CEREBRAS_API_KEY \"sk-XXX\""
            )
        key = self.api_key or os.getenv(key_env, "")
        if not key:
            console.print(f"[yellow]No encontré {key_env}. Puedes pegarla ahora (solo para esta ejecución).[/]")
            pasted = ask(f"Clave {key_env}", "")
            if pasted:
                os.environ[key_env] = pasted
                key = pasted
        if not key:
            raise SystemExit(missing_msg)
        self.api_key = key

    async def complete(self, prompt: str, temperature: float=0.2, max_tokens: int=4096) -> str:
        self._ensure_key()
        env_max = os.getenv("SUBLINGO_MAX_TOKENS")
        max_comp = int(env_max) if env_max else max_tokens

        if self.provider == "groq":
            # Import diferido para no requerirlo si no se usa
            try:
                from groq import Groq
            except Exception:
                raise SystemExit("[ERROR] Instala 'groq' (pip install groq)")
            client = Groq(api_key=self.api_key)

            def collect():
                chunks: List[str] = []
                kwargs = dict(
                    messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                    model=self.model,
                    stream=True,
                    temperature=temperature,
                    max_completion_tokens=max_comp,
                    top_p=1,
                )
                try:
                    iterator = client.chat.completions.create(**kwargs)
                except Exception:
                    # Fallback sin stream
                    resp = client.chat.completions.create(
                        messages=kwargs["messages"],
                        model=self.model,
                        temperature=temperature,
                        max_completion_tokens=max_comp,
                        top_p=1,
                        stream=False,
                    )
                    return resp.choices[0].message.content or ""
                try:
                    for ch in iterator:
                        d = ch.choices[0].delta.content
                        if d:
                            chunks.append(d)
                except Exception as e:
                    if '429' in str(e).lower() or 'rate' in str(e).lower():
                        raise QuotaExceededError("Groq: cuota/límite de tasa alcanzado (429).") from e
                    raise
                return "".join(chunks)

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, collect)

        # provider == cerebras
        try:
            from cerebras.cloud.sdk import Cerebras
        except Exception:
            raise SystemExit("[ERROR] Instala 'cerebras-cloud-sdk' (pip install cerebras-cloud-sdk)")
        client = Cerebras(api_key=self.api_key)

        def collect_cb():
            chunks: List[str] = []
            kwargs = dict(
                messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                model=self.model,
                stream=True,
                temperature=temperature,
                max_completion_tokens=max_comp,
                top_p=1,
            )
            # reasoning opcional si viene definido y distinto de none
            if self.reasoning and str(self.reasoning).lower() != "none":
                kwargs["reasoning_effort"] = self.reasoning
            try:
                iterator = client.chat.completions.create(**kwargs)
            except Exception as e:
                # Reintento sin reasoning por compatibilidad
                kwargs.pop("reasoning_effort", None)
                try:
                    iterator = client.chat.completions.create(**kwargs)
                except Exception as e2:
                    if 'RateLimitError' in e2.__class__.__name__ or '429' in str(e2):
                        raise QuotaExceededError("Cerebras: cuota diaria de tokens excedida o límite de tasa.") from e2
                    raise

            try:
                for ch in iterator:
                    d = ch.choices[0].delta.content
                    if d:
                        chunks.append(d)
            except Exception as e:
                if 'RateLimitError' in e.__class__.__name__ or '429' in str(e):
                    raise QuotaExceededError("Cerebras: cuota diaria de tokens excedida o límite de tasa.") from e
                raise
            return "".join(chunks)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, collect_cb)


# --------------------------- Agentes ---------------------------
@dataclass
class TranslatorAgent:
    llm: LLMClient
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
        prompt = self._rules(src, tgt, aggressive) + f"\nBLOQUES:\n{os.linesep.join(numbered)}\n\nResponde con [[id]] <texto>."
        out = await self.llm.complete(prompt)
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
        batch_size = int(os.getenv("SUBLINGO_BATCH", 60))
        sleep_s = float(os.getenv("SUBLINGO_SLEEP", 0.15))
        fix = []
        for i in range(0, len(blocks), batch_size):
            part = blocks[i:i+batch_size]
            fixed = await self.translator.translate(part, src, tgt, aggressive=True)
            fix.extend(fixed)
            if i + batch_size < len(blocks):
                await asyncio.sleep(sleep_s)
        for j, val in zip(idxs, fix):
            out_blocks[j] = val.strip()
        return out_blocks


# --------------------------- Núcleo de proceso ---------------------------
def derive_narrative_from_filename(path: Path) -> str:
    name = path.stem
    name = re.sub(r"\[.*?\]|\(.*?\)", " ", name)
    name = re.sub(r"[._-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = path.stem
    return (
        f"Título/archivo: '{name}'. Usa este título como contexto de la obra "
        f"para mantener coherencia de nombres propios, terminología y tono."
    )

async def process_subtitle(
    path: Path,
    llm: LLMClient,
    src_lang: str,
    tgt_lang: str,
    preview: bool=False,
    only_styles: Optional[set] = None,
    skip_styles: Optional[set] = None,
):
    import pysubs2  # aseguramos import local; se valida al inicio de main
    subs = pysubs2.load(path, encoding="utf-8")
    is_ass = path.suffix.lower() == ".ass"

    # Filtrado por tipo y estilo (si aplica)
    def style_allowed(ev) -> bool:
        if only_styles and ev.style not in only_styles:
            return False
        if skip_styles and ev.style in skip_styles:
            return False
        return True

    events = [ev for ev in subs if ev.type == "Dialogue" and style_allowed(ev) and should_translate(ev.plaintext)]
    if not events:
        console.print(f"[yellow]No hay diálogos traducibles en {path.name}[/]")
        return 0, 0

    def tokenize_ass_text(text: str):
        tokens = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "{":
                j = text.find("}", i + 1)
                if j == -1:
                    tokens.append(("text", text[i:]))
                    break
                tokens.append(("tag", text[i:j+1]))
                i = j + 1
                continue
            if ch == "\\" and i + 1 < n and text[i+1] in ("N", "n"):
                tokens.append(("newline", text[i:i+2]))
                i += 2
                continue
            j = i
            while j < n and text[j] != "{" and not (text[j] == "\\" and j + 1 < n and text[j+1] in ("N", "n")):
                j += 1
            tokens.append(("text", text[i:j]))
            i = j
        return tokens

    src_blocks: List[str] = []
    missing = 0

    if is_ass:
        assembly_ass = []
        for i, ev in enumerate(subs):
            if ev in events:
                toks = tokenize_ass_text(ev.text)
                token_objs = []
                shape_on = False
                for ttype, tval in toks:
                    if ttype == "text":
                        m = re.match(r"^(\s*[-–—•]\s+)?(.*)$", tval, re.S)
                        prefix = m.group(1) or ""
                        content = (m.group(2) or "")
                        # Si está en modo shape (\p>0), nunca se traduce
                        if shape_on:
                            block_idx = None
                        else:
                            if should_translate(content):
                                block_idx = len(src_blocks)
                                src_blocks.append(content)
                            else:
                                block_idx = None
                        token_objs.append({
                            "type": "text",
                            "prefix": prefix,
                            "content": content,
                            "block_idx": block_idx,
                            "shape": shape_on,
                        })
                    elif ttype == "tag":
                        # Actualizar estado de shape si aparece \pN
                        try:
                            mshape = re.search(r"\\p(\d+)", tval)
                            if mshape:
                                shape_on = int(mshape.group(1)) > 0
                        except Exception:
                            pass
                        token_objs.append({"type": "tag", "text": tval})
                    else:
                        token_objs.append({"type": "newline", "text": tval})
                assembly_ass.append((i, token_objs))
    else:
        assembly = []  # [(idx_evento, num_sublíneas, prefijos)]
        for i, ev in enumerate(subs):
            if ev in events:
                raw = ev.plaintext.replace("\r\n", "\n").replace("\r", "\n").strip()
                parts = raw.split("\n")
                prefixes, clean = [], []
                for p in parts:
                    m = re.match(r'^(\s*[-–—•]\s+)?(.*)$', p)
                    prefixes.append(m.group(1) or "")
                    clean.append(m.group(2))
                assembly.append((i, len(parts), prefixes))
                src_blocks.extend(clean)

    batch_size = int(os.getenv("SUBLINGO_BATCH", 60))
    sleep_s   = float(os.getenv("SUBLINGO_SLEEP", 0.15))

    narrative = derive_narrative_from_filename(path)
    translator = TranslatorAgent(llm, narrative)
    validator  = ValidatorAgent(translator)

    out_blocks: List[str] = []
    with Progress(TextColumn("[bold]lotes[/]"), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as bar:
        t = bar.add_task("traduciendo", total=len(src_blocks))
        for i in range(0, len(src_blocks), batch_size):
            batch = src_blocks[i:i+batch_size]
            out = await translator.translate(batch, src_lang, tgt_lang, aggressive=False)
            out_blocks.extend(out)

            if preview:
                previews = []
                for s, o in zip(batch[:4], out[:4]):
                    previews.append(f"- {trunc(s, 40)} -> {trunc((o or ''), 40)}")
                if previews:
                    bar.console.print("\n".join(previews))

            bar.update(t, advance=len(batch))
            if i + batch_size < len(src_blocks):
                await asyncio.sleep(sleep_s)

    out_blocks = await validator.repair(src_blocks, out_blocks, src_lang, tgt_lang)

    if is_ass:
        for idx_event, token_objs in assembly_ass:
            rendered = []
            for tok in token_objs:
                if tok["type"] in {"tag", "newline"}:
                    rendered.append(tok["text"])
                else:
                    block_idx = tok["block_idx"]
                    prefix = tok["prefix"]
                    if block_idx is None:
                        rendered.append(prefix + tok["content"])
                        continue
                    out_line_raw = (out_blocks[block_idx] if block_idx < len(out_blocks) else "")
                    src_line_raw = (src_blocks[block_idx] if block_idx < len(src_blocks) else "")
                    out_line = out_line_raw.strip()
                    if not out_line:
                        missing += 1
                        out_line = src_line_raw
                    else:
                        # Conserva los espacios de borde del segmento original
                        out_line = preserve_edge_spaces(src_line_raw, out_line)
                    # Reaplica prefijos si existen (guiones, etc.)
                    if prefix and not re.match(r'^\s*[-–—•]\s+', out_line):
                        out_line = prefix + out_line
                    else:
                        out_line = prefix + out_line
                    rendered.append(out_line)
            # Unir y normalizar espaciado/idiomas
            final_text = "".join(rendered)
            final_text = fix_ass_italic_spacing(final_text)
            if tgt_lang.lower().startswith("es"):
                final_text = fix_es_ass_idioms(final_text)
            subs[idx_event].text = final_text
    else:
        pos = 0
        for idx_event, count, prefixes in assembly:
            lines = []
            for j in range(count):
                out_line = (out_blocks[pos + j] if pos + j < len(out_blocks) else "").strip()
                src_line = (src_blocks[pos + j] if pos + j < len(src_blocks) else "").strip()
                if not out_line:
                    missing += 1
                    out_line = src_line
                if prefixes[j] and not re.match(r'^\s*[-–—•]\s+', out_line):
                    out_line = prefixes[j] + out_line
                lines.append(out_line)
            pos += count
            subs[idx_event].text = "\n".join(lines)

    subs_dir = Path(os.getenv("SUBS_BULK_DIR", "SUBS_BULK")).resolve()
    out_dir = subs_dir / "TRADUCIDOS" / tgt_lang
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + f".{tgt_lang}" + path.suffix)

    subs.save(out_path, encoding="utf-8", format_=("ass" if is_ass else "srt"))

    console.log(Panel(
        f"[bold]{path.name}[/]\nGuardado en: [green]{out_path}[/]\nTiempos corregidos : [cyan]0[/]\nLíneas sin traducir : [cyan]{missing}[/]",
        title="Resultado", border_style="blue", expand=False
    ))
    return 0, missing


def gather_files(cwd: Path) -> List[Path]:
    return sorted([p for p in cwd.glob("*.srt")] + [p for p in cwd.glob("*.ass")])


# --------------------------- CLI ---------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sublingo Ultra – Traductor unificado")
    p.add_argument("--modelo", "-m", default=os.getenv("SUBLINGO_MODEL", "gpt-oss"),
                   help="Alias de modelo: " + ", ".join(SUPPORTED_ALIASES.keys()) + " o id explícito")
    p.add_argument("--proveedor", "--provider", choices=["auto","groq","cerebras"], default=os.getenv("SUBLINGO_PROVIDER", "auto"),
                   help="Proveedor LLM (auto por defecto)")
    p.add_argument("--model-id", default=os.getenv("SUBLINGO_MODEL_ID", ""),
                   help="Model ID explícito (anula el alias si se provee)")
    p.add_argument("--src", default=os.getenv("SUBLINGO_SRC", ""), help="Idioma fuente (ISO). Si vacío, se detecta")
    p.add_argument("--tgt", required=False, default=os.getenv("SUBLINGO_TGT", "es-419"), help="Idioma destino (ISO)")
    p.add_argument("--dir", default=os.getenv("SUBS_BULK_DIR", "SUBS_BULK"), help="Carpeta de subtítulos de entrada")
    p.add_argument("--batch", type=int, default=int(os.getenv("SUBLINGO_BATCH", 60)), help="Tamaño de lote")
    p.add_argument("--sleep", type=float, default=float(os.getenv("SUBLINGO_SLEEP", 0.15)), help="Pausa entre lotes (s)")
    p.add_argument("--razonamiento", "--reasoning", default=os.getenv("SUBLINGO_REASONING", "none"),
                   help="Razonamiento (none|low|medium|high) – solo Cerebras si aplica")
    p.add_argument("--p", "--proceso", "--t", dest="preview", action="store_true", help="Mostrar vista de proceso (previews)")
    p.add_argument("--no-header", action="store_true", help="Ocultar banner inicial")
    p.add_argument("--only-styles", default=os.getenv("SUBLINGO_ONLY_STYLES", ""),
                   help="Traducir solo estos estilos (coma-separados), ej: 'Default,OP English'")
    p.add_argument("--skip-styles", default=os.getenv("SUBLINGO_SKIP_STYLES", ""),
                   help="Omitir estos estilos (coma-separados), ej: 'Signs,OP Romaji'")
    p.add_argument("--sin-filtro", dest="disable_filter", action="store_true",
                   help="Desactiva el filtro inteligente de estilos (por defecto activado)")
    return p.parse_args(argv)


async def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Aplicar overrides al entorno para mantener consistencia interna
    os.environ["SUBLINGO_BATCH"] = str(args.batch)
    os.environ["SUBLINGO_SLEEP"] = str(args.sleep)
    os.environ["SUBS_BULK_DIR"] = args.dir
    os.environ["SUBLINGO_REASONING"] = str(args.razonamiento)

    # Validar dependencia común
    try:
        import pysubs2  # noqa: F401
    except Exception:
        raise SystemExit("[ERROR] Instala 'pysubs2' (pip install pysubs2)")

    subs_dir = Path(args.dir).resolve()
    subs_dir.mkdir(parents=True, exist_ok=True)

    show_header = not args.no_header
    banner("SubLingo – Traductor de subtítulos – Ultra", enabled=show_header)

    files = gather_files(subs_dir)
    if not files:
        console.print(f"[red]No se hallaron archivos .srt/.ass en la carpeta: {subs_dir}[/]")
        console.print("[dim]Coloca los subtítulos fuente en esa carpeta e intenta de nuevo.[/]")
        return

    # Idioma destino
    tgt_lang = args.tgt.strip() or "es-419"

    # Filtrar los que parecen ya traducidos al destino
    def looks_translated(p: Path) -> bool:
        return p.name.endswith(f".{tgt_lang}{p.suffix}")
    files = [f for f in files if not looks_translated(f)]
    if not files:
        console.print(f"[yellow]No hay archivos fuente por procesar en {subs_dir} (ya existen traducciones a {color_lang(tgt_lang)}).[/]")
        return

    console.print("\n[bold]Archivo(s) a procesar:[/]")
    for i, f in enumerate(files, 1):
        console.print(f"{i}. {f.name}")

    # Detectar o aceptar idioma fuente
    if args.src:
        src_lang = args.src
    else:
        import pysubs2
        sample_text = "".join([ev.plaintext for ev in pysubs2.load(files[0], encoding="utf-8") if ev.type=="Dialogue"])[:2000]
        src_guess = quick_lang_detect(sample_text)
        ans = ask(f"Idioma detectado en el primer archivo: {color_lang(src_guess)}. ¿Es correcto? (s/n o escribe el código ISO)", "s")
        if ans.lower() not in {"s","si","sí","y","yes"}:
            src_lang = ans.strip() or src_guess
        else:
            src_lang = src_guess

    # Resolver modelo/proveedor
    model_alias = (args.modelo or "gpt-oss").strip()
    provider = args.proveedor
    model_id = args.model_id.strip()

    if model_id:
        # Si se entrega id explícito, se requiere proveedor explícito o deducimos por heurística
        if provider == "auto":
            provider = "groq" if "/" in model_id else "cerebras"
    else:
        if model_alias in SUPPORTED_ALIASES:
            provider = SUPPORTED_ALIASES[model_alias]["provider"] if provider == "auto" else provider
            model_id = SUPPORTED_ALIASES[model_alias]["model"]
        else:
            # Permitir alias desconocidos como id directo
            model_id = model_alias
            if provider == "auto":
                provider = "groq" if "/" in model_id else "cerebras"

    # Construir cliente LLM
    reasoning = args.razonamiento if provider == "cerebras" else None
    llm = LLMClient(provider=provider, model=model_id, reasoning=reasoning)

    console.print(Panel.fit(f"Proceso iniciado, Modelo usado: {model_id}\nProveedor: {provider}", border_style="magenta"))

    total_t0 = start_time()
    try:
        # Preparar filtros de estilo
        only_styles_set = {s.strip() for s in (args.only_styles or "").split(",") if s.strip()} or None
        skip_styles_set = {s.strip() for s in (args.skip_styles or "").split(",") if s.strip()} or None
        # Filtro inteligente por defecto (si el usuario no especificó nada y no se desactiva):
        if not args.disable_filter and not only_styles_set and not skip_styles_set:
            skip_styles_set = {"Signs", "OP Romaji", "ED Romaji"}

        for f in files:
            t0 = start_time()
            await process_subtitle(
                f, llm, src_lang, tgt_lang,
                preview=args.preview,
                only_styles=only_styles_set,
                skip_styles=skip_styles_set,
            )
            dt = start_time() - t0
            console.print(f"[bold cyan]⏱ Tiempo {f.name}:[/] [green]{fmt_duration(dt)}[/]")
    except QuotaExceededError as e:
        if provider == "groq":
            console.print(Panel(
                "Cuota de tokens/límites de tasa en Groq excedidos (429).\n"
                "Opciones: espera el reinicio diario, reduce consumo con\n"
                "SUBLINGO_MAX_TOKENS=2048-4096, SUBLINGO_BATCH=30-60;\n"
                "o usa otra clave/modelo.",
                title="Límite de tokens alcanzado", border_style="red", expand=False
            ))
        else:
            console.print(Panel(
                "Cuota de tokens de Cerebras excedida (429).\n"
                "Opciones: espera el reinicio diario, reduce consumo con\n"
                "SUBLINGO_REASONING=none, SUBLINGO_MAX_TOKENS=2048-4096,\n"
                "SUBLINGO_BATCH=30-60; o usa otra clave/modelo.",
                title="Límite de tokens alcanzado", border_style="red", expand=False
            ))
        return

    dt_total = start_time() - total_t0
    console.print(f"[bold green]Trabajo finalizado[/]  [dim]| total:[/] [bold magenta]{fmt_duration(dt_total)}[/]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelado por el usuario.[/]")
