# Whisper Transcriber

Transcriptor **robusto** de múltiples audios a texto y subtítulos usando [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) (aprovecha GPU si está disponible), con limpieza automática del texto y generación de `.srt`. Funciona tanto en **GPU con CUDA** como en **CPU**.

---

## 🚀 Instalación y Primer Uso

### Primera vez (configuración inicial)
1. Clona el repo:
   ```bash
   git clone git@github.com:nicolaymh/whisper-transcriber.git
   cd whisper-transcriber
   ```

2. Crea el entorno e instala dependencias:
   - Con **GPU NVIDIA**:
     ```bash
     make setup
     make install-gpu
     make install-cudnn
     ```
   - Solo **CPU**:
     ```bash
     make setup
     make install-cpu
     ```

3. Coloca tus audios en la carpeta:
   ```bash
   audios/
   ```

4. Ejecuta la transcripción:
   ```bash
   make run
   ```

### Uso diario (después de la primera vez)
```bash
cd whisper-transcriber
make run
```

Los resultados se generan en `transcripciones/` como `.txt` y `.srt`.

---

## 🚀 Características clave
- **Batch**: procesa todos los audios dentro de `audios/` y guarda resultados en `transcripciones/`.
- **Modelos Whisper**: intenta `large-v3` y, si falla, retrocede a `large-v2` automáticamente.
- **Optimización automática**: autodetecta CUDA; usa `float16` en GPU y `int8` en CPU.
- **Filtrado de ruido/no-voz**: VAD activado y umbrales para evitar repeticiones o texto espurio.
- **Post-procesado avanzado**: limpia basura, deduplica líneas y colapsa palabras repetidas.
- **Salida doble**: genera **TXT** con cabecera y **SRT** con marcas de tiempo.
- **Gestión de memoria**: liberación explícita de caché en GPU por iteración.

---

## 📦 Requisitos

Antes de usar el transcriptor asegúrate de tener instalado:

### 1) Python
- **Python 3.10+** recomendado (el Makefile creará un entorno virtual `.venv/` con todas las librerías necesarias).

### 2) FFmpeg
Necesario para soportar múltiples formatos de audio.

- **Ubuntu/Debian**
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **Windows**
  Descarga desde <https://ffmpeg.org/download.html> y agrega `ffmpeg.exe` a tu **PATH**.
- **macOS (Homebrew)**
  ```bash
  brew install ffmpeg
  ```

### 3) Dependencias Python
No es necesario instalarlas manualmente.  
El **Makefile crea el entorno virtual y gestiona todas las dependencias** (`torch`, `faster-whisper`, `cudnn`, etc.).

> ⚡ **Nota**: No necesitas ejecutar `source .venv/bin/activate`.  
> Cada comando `make` activa y usa el entorno virtual automáticamente.

---

## 🗂️ Estructura de carpetas
```
.
├── audios/                 # ⇦ coloca aquí tus .mp3, .wav, .m4a, .opus, .ogg
├── transcripciones/        # ⇦ se recrea en cada ejecución (se borra si ya existe)
├── transcribe.py           # script principal (el algoritmo provisto)
└── README.md               # este archivo
```

> ⚠️ **Ojo**: el script elimina `transcripciones/` al inicio para empezar limpio:
> ```python
> if OUT_DIR.exists():
>     shutil.rmtree(OUT_DIR)
> OUT_DIR.mkdir(parents=True, exist_ok=True)
> ```
> Haz copia si no quieres perder resultados anteriores.

---

## ▶️ Uso
1. Coloca tus audios dentro de `audios/` (se procesan en orden “natural”, p. ej. `1_intro.mp3`, `2_parte.mp3`, ...).
2. Ejecuta:
   ```bash
   python transcribe.py
   ```
3. Revisa la carpeta `transcripciones/`:
   - `N - <nombre>.txt` → texto con cabecera y duración total del audio.
   - `N - <nombre>.srt` → subtítulos con timestamps precisos.

### Ejemplo de salida `.txt`
```
1 - entrevista_audio
Duración: 00:42:17

Transcripción:

[Primeras líneas limadas por post-procesado...]
```

---

## 🧩 Opciones de modelo disponibles

`faster-whisper` soporta varios modelos de distinto tamaño y consumo.  
En el script por defecto se usa **large-v3** (máxima precisión), con retroceso a **large-v2**,  
pero puedes cambiarlos en la variable `MODEL_NAME` al inicio.

Modelos soportados:

- `tiny` → muy rápido, bajo consumo. Menor precisión.
- `base` → rápido, precisión aceptable.
- `small` → buen equilibrio entre velocidad y calidad.
- `medium` → alta precisión, requiere más memoria.
- `large-v2` → muy preciso, recomendado si tienes GPU/CPU potentes.
- `large-v3` → versión más reciente y precisa (por defecto en este script).

👉 Ajusta en `transcribe.py`:
```python
MODEL_NAME = "small"
```
para usar el modelo que mejor se adapte a tu hardware.

---

## ⚙️ Parámetros importantes del algoritmo

### Selección de dispositivo y precisión
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"  # GPU=fp16, CPU=int8
```
- **GPU**: usa `float16` (rápido y eficiente en VRAM).
- **CPU**: usa `int8` para reducir consumo de memoria.

### Modelo Whisper
```python
MODEL_NAME = "large-v3"
try:
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
except Exception:
    MODEL_NAME = "large-v2"
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
```
- Intenta `large-v3` y hace **fallback** a `large-v2` si falla (falta de RAM/VRAM o descarga interrumpida).

### Configuración de transcripción
```python
segments, info = model.transcribe(
    str(audio_file),
    language="es",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
    condition_on_previous_text=False,
    temperature=0.0,
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.4,
    beam_size=5,
)
```
- **`language="es"`**: fuerza español (ajústalo si es otro idioma).
- **`vad_filter`**: filtra silencios/música, útil para radio/pódcast.
- **`condition_on_previous_text=False`**: evita “memoria” entre segmentos (reduce arrastre de errores).
- **`temperature=0.0`**: resultados más estables.
- **`no_speech_threshold` y `compression_ratio_threshold`**: descartan no-voz y texto raro.
- **`beam_size=5`**: decodificación robusta (más lenta que greedy, pero mejor).

### Post-procesado del texto
- **`limpiar_basura`**: elimina frases comunes no deseadas (ej. Amara.org).
- **`dedupe_lines`**: quita repeticiones de líneas consecutivas.
- **`colapsar_palabras_repetidas`**: comprime “hola hola hola” → “hola”.
- Fixes ligeros de espaciado en puntuación.

### Generación de SRT
Timestamps en formato `HH:MM:SS,mmm` con precisión milisegundos:
```python
00:00:00,000 --> 00:00:04,120
Texto del segmento 1
```

### Orden natural de archivos
Se aplica un **orden natural** para que `1_intro`, `2_parte`, `10_extra` queden bien ordenados.

---

## 🔧 Personalización rápida
- **Idioma**: cambia `language="es"` por el código ISO deseado.
- **Modelo**: sustituye `MODEL_NAME` por `tiny`, `base`, `small`, `medium`, `large-v2`, etc.
- **VAD**: ajusta `min_silence_duration_ms` (p. ej. 300 para cortes más finos).
- **Carpetas**: modifica `AUDIO_DIR` y `OUT_DIR` al inicio del script.

---

## 📈 Rendimiento y VRAM
- **GPU 8–12 GB**: `large-v2` suele ir cómodo; `large-v3` puede requerir cerrar apps.
- **GPU modesta o CPU**: considera `small` o `medium` para mayor velocidad con precisión aceptable.
- Si te quedas sin VRAM: usa `large-v2` o un modelo más pequeño.
- En **CPU**, el tiempo crecerá considerablemente; `tiny`, `base` o `small` son más prácticos.

---

## 🧪 Solución de problemas

**1) “CUDA not available” o PyTorch no detecta GPU**  
- Verifica controladores NVIDIA + versión CUDA compatible con tu PyTorch (`torch.version.cuda`).
- Instala la build correcta de PyTorch para tu CUDA (ver arriba).

**2) “ffmpeg not found”**  
- Asegúrate de tener FFmpeg instalado y en el `PATH` del sistema.

**3) “Out of memory” (VRAM)**  
- Cambia a un modelo más pequeño (`medium`, `small`, etc.`).
- Cierra otros procesos que usen GPU.
- Reduce simultaneidad (no aplicable si procesas 1 archivo a la vez).

**4) Descargas interrumpidas de modelos**  
- Borra la carpeta de caché de `faster-whisper/ctranslate2` y vuelve a ejecutar para re-descargar.

**5) Texto con repeticiones**  
- Ya se aplican filtros (`no_speech_threshold`, `compression_ratio_threshold`) y post-procesado. Ajusta umbrales si persiste.

---

## 📜 Licencia
Este proyecto se distribuye bajo **MIT** (puedes cambiarlo si tu repositorio usa otra).

## 🙌 Créditos
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) por su implementación eficiente de Whisper.
- OpenAI Whisper (modelo original).

---

## 📝 Notas
- La carpeta `transcripciones/` se **recrea** en cada corrida (se borra si existía).
- El script muestra **progreso por archivo**, duración total transcrita y lista de errores si los hubiera.
