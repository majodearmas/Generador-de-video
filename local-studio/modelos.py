"""Catálogo de modelos (todos gratis, descargados de Hugging Face) y gestor de memoria.

Solo se mantiene UN modelo cargado a la vez: al cambiar de imagen a video se libera
el anterior para que quepa en la GPU.
"""

import gc

DEMO = "Demo (sin IA, solo para probar el panel)"

IMAGE_MODELS = {
    "RealVisXL 5 - fotorrealista (8 GB VRAM)": {
        "repo": "SG161222/RealVisXL_V5.0",
        "tipo": "sdxl",
        "steps": 30,
        "guidance": 6.0,
        "size": (1344, 768),
    },
    "FLUX.1 schnell - más detalle (16 GB+ VRAM)": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "tipo": "flux",
        "steps": 4,
        "guidance": 0.0,
        "size": (1360, 768),
    },
    DEMO: {"tipo": "demo", "size": (1344, 768)},
}

VIDEO_MODELS = {
    "LTX-Video 0.9.5 - rápido (12 GB VRAM)": {
        "repo": "Lightricks/LTX-Video-0.9.5",
        "tipo": "ltx",
        "fps": 24,
        "mod": 32,
        "frames_mult": 8,
        "steps": 40,
        "guidance": 3.0,
    },
    "Wan 2.1 FLF2V 14B 720p - alta calidad (24 GB VRAM)": {
        "repo": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        "tipo": "wan",
        "clip": True,
        "fps": 16,
        "mod": 16,
        "frames_mult": 4,
        "steps": 40,
        "guidance": 5.5,
    },
    "Wan 2.2 I2V A14B - máxima calidad (24 GB VRAM + 64 GB RAM)": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "tipo": "wan",
        "clip": False,
        "fps": 16,
        "mod": 16,
        "frames_mult": 4,
        "steps": 40,
        "guidance": 3.5,
    },
    DEMO: {"tipo": "demo", "fps": 24, "mod": 16, "frames_mult": 1},
}

RESOLUCIONES = {
    "480p (832x480) - recomendado para probar": (832, 480),
    "720p (1280x720) - para YouTube": (1280, 720),
}

MEMORIA = {
    "Normal (mueve partes a la RAM cuando no se usan)": "offload",
    "Mínima VRAM (muy lento, para GPUs pequeñas)": "sequential",
    "Todo en GPU (rápido, necesita mucha VRAM)": "gpu",
}

_cargado = {"clave": None, "pipe": None}


def dispositivo():
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def info_hardware():
    try:
        import torch
    except ImportError:
        return "PyTorch no está instalado: solo funciona el modo Demo. Ejecuta el instalador."
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"GPU: {p.name} · {p.total_memory / 1024**3:.1f} GB VRAM · PyTorch {torch.__version__}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return f"Apple Silicon (MPS) · PyTorch {torch.__version__} · será lento con los modelos de video"
    return "No se detectó GPU: la IA correrá en CPU y será MUY lenta (usa el modo Demo para probar)."


def liberar():
    if _cargado["pipe"] is None:
        return
    _cargado["pipe"] = None
    _cargado["clave"] = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def obtener(clave, cargador):
    """Devuelve el pipeline `clave`, cargándolo (y liberando el anterior) si hace falta."""
    if _cargado["clave"] != clave:
        liberar()
        _cargado["pipe"] = cargador()
        _cargado["clave"] = clave
    return _cargado["pipe"]


def colocar(pipe, modo_memoria):
    """Mueve el pipeline al dispositivo según el modo de memoria elegido."""
    dev, _ = dispositivo()
    if dev == "cuda" and modo_memoria == "offload":
        pipe.enable_model_cpu_offload()
    elif dev == "cuda" and modo_memoria == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(dev)
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
        except Exception:
            pass
    return pipe
