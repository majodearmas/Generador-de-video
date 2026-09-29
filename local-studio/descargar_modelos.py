"""Descarga por adelantado los modelos de Hugging Face (opcional: el panel también los descarga solo).

Uso:
    python descargar_modelos.py            # muestra la lista y pregunta
    python descargar_modelos.py 1 3        # descarga los modelos 1 y 3
"""

import sys

from diffusers import DiffusionPipeline

import modelos

TAMANOS = {
    "SG161222/RealVisXL_V5.0": "~7 GB",
    "black-forest-labs/FLUX.1-schnell": "~34 GB",
    "Lightricks/LTX-Video-0.9.5": "~25 GB",
    "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers": "~75 GB",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "~125 GB",
}


def lista():
    vistos = []
    for catalogo in (modelos.IMAGE_MODELS, modelos.VIDEO_MODELS):
        for nombre, cfg in catalogo.items():
            if "repo" in cfg:
                vistos.append((nombre, cfg["repo"]))
    return vistos


def descargar(repo):
    print(f"\n⬇️  Descargando {repo} ({TAMANOS.get(repo, '?')}) ...")
    try:
        DiffusionPipeline.download(repo, variant="fp16")
    except Exception:
        DiffusionPipeline.download(repo)
    print(f"✅ {repo} listo")


if __name__ == "__main__":
    opciones = lista()
    elegidos = sys.argv[1:]
    if not elegidos:
        for i, (nombre, repo) in enumerate(opciones, 1):
            print(f"{i}. {nombre}\n   {repo}  ({TAMANOS.get(repo, '?')})")
        elegidos = input("\nNúmeros a descargar separados por espacio (ej: 1 3): ").split()
    for n in elegidos:
        descargar(opciones[int(n) - 1][1])
