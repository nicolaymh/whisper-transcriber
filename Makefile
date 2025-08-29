# Makefile optimizado para Whisper Transcriber (GPU + cuDNN en el venv)
# Uso típico:
#   make setup            - crea .venv y actualiza pip
#   make install-gpu      - instala Torch (CUDA 12.1) + faster-whisper
#   make install-cudnn    - instala cuDNN 9 en el venv
#   make run              - ejecuta con cuDNN (LD_LIBRARY_PATH dinámico desde el venv)
#   make run-nocudnn      - ejecuta desactivando cuDNN (fallback rápido)
#   make clean            - borra transcripciones/
#   make freeze           - genera requirements.txt
#   make help             - muestra ayuda
#
# Nota: Todo queda encapsulado en el venv; no modifica ~/.zshrc ni librerías del sistema.

.PHONY: setup install-gpu install-cpu install-cudnn run run-nocudnn clean freeze help ensure-venv ensure-audios

PYTHON       := python3
VENV_DIR     := .venv
ACTIVATE     := . $(VENV_DIR)/bin/activate

help:
	@echo "Comandos disponibles:"
	@echo "  make setup          - Crea $(VENV_DIR) y actualiza pip"
	@echo "  make install-gpu    - Instala Torch (CUDA 12.1) + faster-whisper"
	@echo "  make install-cpu    - Instala Torch (CPU) + faster-whisper"
	@echo "  make install-cudnn  - Instala cuDNN 9 en el venv"
	@echo "  make run            - Ejecuta transcribe.py usando cuDNN del venv"
	@echo "  make run-nocudnn    - Ejecuta transcribe.py desactivando cuDNN"
	@echo "  make clean          - Elimina transcripciones/"
	@echo "  make freeze         - Genera requirements.txt"

# Crea el venv si no existe y actualiza pip
setup: ensure-venv
	$(ACTIVATE) && python -m pip install --upgrade pip

# Instala dependencias con soporte GPU (CUDA 12.1)
install-gpu: ensure-venv
	$(ACTIVATE) && pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
	$(ACTIVATE) && pip install faster-whisper

# Instala dependencias solo CPU
install-cpu: ensure-venv
	$(ACTIVATE) && pip install torch torchvision torchaudio
	$(ACTIVATE) && pip install faster-whisper

# Instala cuDNN 9 dentro del venv (queda en site-packages/nvidia/cudnn/lib)
install-cudnn: ensure-venv
	$(ACTIVATE) && pip install nvidia-cudnn-cu12==9.1.0.70

# Ejecuta el transcriptor usando cuDNN desde el venv.
# Añade dinámicamente la ruta .../site-packages/nvidia/cudnn/lib a LD_LIBRARY_PATH solo para este comando.
run: ensure-venv ensure-audios
	$(ACTIVATE) && export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$$($(VENV_DIR)/bin/python -c "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0] + '/nvidia/cudnn/lib')") && python transcribe.py

# Fallback sin cuDNN (sigue usando GPU con cuBLAS; un poco más lento)
run-nocudnn: ensure-venv ensure-audios
	$(ACTIVATE) && CT2_USE_CUDNN=0 python transcribe.py

# Limpia la carpeta de resultados
clean:
	rm -rf transcripciones/

# Congela dependencias a requirements.txt
freeze: ensure-venv
	$(ACTIVATE) && pip freeze > requirements.txt

# --- Helpers ---
ensure-venv:
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)

ensure-audios:
	@mkdir -p audios
