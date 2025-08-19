# -*- coding: utf-8 -*-
# Transcripción robusta por lotes con faster-whisper (aprovecha GPU si está disponible).
# Dependencia principal: pip install faster-whisper

from faster_whisper import WhisperModel
import pathlib, torch, gc, shutil, math, re, sys, traceback
from datetime import timedelta

# ==================== UTILIDADES ====================
# Funciones auxiliares para formateo de tiempo, orden natural de archivos
# y post-procesado de texto (limpieza y reducción de repeticiones).

def fmt_hhmmss(seconds: float) -> str:
    seconds = int(math.floor(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _split_natural(s: str):
    # Divide texto en bloques alfanuméricos para ordenar de forma "natural"
    # (p. ej. 2 < 10 aunque ambos comiencen con dígitos).
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r'(\d+)', s)]

def windows_natural_key(p: pathlib.Path):
    # Clave de orden "natural": primero prefijos numéricos, luego resto del nombre.
    name = p.name.strip()
    m = re.match(r'^(\d+)\b', name)
    if m:
        lead = int(m.group(1)); rest = name[m.end():]
        return (0, lead, _split_natural(rest))
    else:
        return (1, float('inf'), _split_natural(name))

# --- Post-procesado de texto ---
# Remueve líneas irrelevantes, colapsa repeticiones y ajusta espaciado/puntuación.
BASURA = {
    "Subtítulos realizados por la comunidad de Amara.org",
    "Subtitles by Amara.org community",
}

def limpiar_basura(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l and l not in BASURA]
    return "\n".join(lines)

def dedupe_lines(text: str, max_runs: int = 1) -> str:
    # Elimina repeticiones de líneas consecutivas, permitiendo hasta max_runs ocurrencias.
    out, last, run = [], None, 0
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if line == last:
            run += 1
            if run < max_runs:
                out.append(line)
        else:
            last, run = line, 0
            out.append(line)
    return "\n".join(out)

def colapsar_palabras_repetidas(text: str) -> str:
    # Ejemplos:
    # "Bortilla, Bortilla, Bortilla" -> "Bortilla"
    # "hola hola hola" -> "hola"
    text = re.sub(r"\b(\w+)(?:[,\s]+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    return text

def postprocesar(text: str) -> str:
    text = limpiar_basura(text)
    text = dedupe_lines(text, max_runs=1)
    text = colapsar_palabras_repetidas(text)
    # Ajustes básicos de puntuación
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip()

def srt_timestamp(t):
    # Formato SRT estándar: 00:00:00,000
    td = timedelta(seconds=max(0.0, float(t)))
    return str(td)[:-3].rjust(12, "0").replace(".", ",")

def escribir_srt(segments, path_srt):
    # Emite archivo .srt a partir de segmentos aceptados.
    with open(path_srt, "w", encoding="utf-8") as f:
        idx = 1
        for s in segments:
            start = srt_timestamp(s.start)
            end = srt_timestamp(s.end)
            text = s.text.strip().replace("\n", " ")
            if not text:
                continue
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
            idx += 1

# ==================== CONFIGURACIÓN DE RUTAS ====================
# Se procesan todos los audios en 'audios/' y se escriben resultados en 'transcripciones/'.
# La carpeta de salida se recrea en cada ejecución para garantizar un estado limpio.

AUDIO_DIR = pathlib.Path("audios")
OUT_DIR = pathlib.Path("transcripciones")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

audio_files = sorted(
    [f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in [".mp3", ".wav", ".m4a", ".opus", ".ogg"]],
    key=windows_natural_key
)

print(f"🎧 Se encontraron {len(audio_files)} audio(s) para transcribir:\n")
for i, audio in enumerate(audio_files, start=1):
    print(f"  {i}. {audio.name}")

if not audio_files:
    print("⚠️ No se encontraron archivos de audio en la carpeta 'audios/'.")
    sys.exit(0)

# ==================== SELECCIÓN DE MODELO Y DISPOSITIVO ====================
# Se selecciona automáticamente el dispositivo:
# - GPU con CUDA si está disponible (torch.cuda.is_available()).
# - CPU en caso contrario.
# La precisión se ajusta para optimizar memoria y rendimiento:
#   - GPU: float16
#   - CPU: int8
# Modelos recomendados: "large-v3" (si está disponible) o "large-v2" como alternativa.

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"  # fp16 en GPU; int8 en CPU

MODEL_NAME = "large-v3"

try:
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
except Exception as e:
    print(f"⚠️ No se pudo cargar {MODEL_NAME}, probando large-v2... ({e})")
    MODEL_NAME = "large-v2"
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)

print(f"\n✅ Modelo cargado: {MODEL_NAME} | device={device} | compute_type={compute_type}")

# ==================== TRANSCRIPCIÓN ====================
# Se recorre cada archivo de audio, se transcribe con parámetros conservadores
# (estabilidad y reducción de repeticiones), y se generan TXT + SRT por archivo.

total_seconds = 0.0
errores = []

for i, audio_file in enumerate(audio_files, start=1):
    print(f"\n📌 [{i}/{len(audio_files)}] Transcribiendo: {audio_file.name}")

    output_base = f"{i} - {audio_file.stem}"
    out_txt = OUT_DIR / f"{output_base}.txt"
    out_srt = OUT_DIR / f"{output_base}.srt"

    try:
        # Configuración orientada a robustez (radio/pódcast/entrevistas):
        segments, info = model.transcribe(
            str(audio_file),
            language="es",                          # Ajusta si el idioma es distinto
            vad_filter=True,                        # Filtra silencios/no-voz
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,       # Evita dependencia entre segmentos
            temperature=0.0,                        # Mayor estabilidad
            no_speech_threshold=0.6,                # Ignora tramos sin voz
            compression_ratio_threshold=2.4,        # Filtra texto anómalo/duplicado
            beam_size=5,                            # Decodificación más robusta que greedy
            # best_of=1,                            # (solo para sampling; no aplica con beam)
            # word_timestamps=False,
        )

        dur = info.duration or 0.0
        total_seconds += dur

        # Construye texto filtrando segmentos de muy baja confianza.
        trozos = []
        segs_buenos = []
        for s in segments:
            if getattr(s, "avg_logprob", None) is not None and s.avg_logprob < -1.0:
                continue
            text_seg = s.text.strip()
            if text_seg:
                trozos.append(text_seg)
                segs_buenos.append(s)

        texto = "\n".join(trozos)
        texto = postprocesar(texto)

        # Guardar TXT con cabecera básica (nombre/duración)
        with out_txt.open("w", encoding="utf-8") as f:
            f.write(f"{output_base}\n")
            f.write(f"Duración: {fmt_hhmmss(dur)}\n\n")
            f.write("Transcripción:\n\n")
            f.write(texto + "\n")

        # Guardar SRT usando solo los segmentos aceptados
        escribir_srt(segs_buenos, out_srt)

        print(f"✅ Transcripción: {out_txt.name} | Subtítulos: {out_srt.name}")
        print(f"⏱️ Duración del audio: {fmt_hhmmss(dur)}")
        print("\nTranscripción (primeras líneas):")
        for line in texto.splitlines()[:6]:
            print(" ", line)

        # Liberación de memoria por iteración (CPU/GPU)
        del segments, info, segs_buenos, trozos, texto
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # Registro de errores por archivo y continuidad del lote.
        errores.append((audio_file.name, str(e)))
        print(f"❌ Error transcribiendo {audio_file.name}: {e}")
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

# ==================== REPORTE FINAL ====================
# Resumen al finalizar el lote: duración total y listado de archivos con error.

print(f"\n🧮 Duración total transcrita: {fmt_hhmmss(total_seconds)}")
if errores:
    print("⚠️ Audios con error:")
    for name, msg in errores:
        print(f"  - {name}: {msg}")
else:
    print("🎉 ¡Todas las transcripciones han sido completadas con éxito!")
