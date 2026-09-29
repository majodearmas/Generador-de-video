#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "=== Instalando el estudio local (Python 3.10-3.12) ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
if [[ "$(uname)" == "Darwin" ]]; then
  pip install torch torchvision
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
pip install -r requirements.txt
echo "Listo. Ahora ejecuta ./iniciar.sh"
