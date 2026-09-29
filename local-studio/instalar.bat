@echo off
cd /d "%~dp0"
echo === Instalando el estudio local (requiere Python 3.10-3.12 y GPU NVIDIA) ===
python -m venv venv || goto :error
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :error
pip install -r requirements.txt || goto :error
echo.
echo Listo. Ahora ejecuta iniciar.bat
pause
exit /b 0
:error
echo Hubo un error durante la instalacion.
pause
exit /b 1
