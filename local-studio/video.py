"""Paso 2: animar la transición entre cada par de imágenes y unirlo todo en un solo video."""

import os
import random
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps

import modelos
from imagenes import SALIDAS

NEGATIVO = (
    "static, frozen, blurry, low quality, distorted, deformed, jittery, flickering, "
    "morphing artifacts, text, watermark, subtitles, cartoon"
)
MOVIMIENTO_BASE = (
    "construction timelapse of a house renovation, workers repairing and rebuilding the house, "
    "progress happens gradually, static camera, photorealistic"
)


def _cargar(cfg, modo_memoria):
    import torch

    _, dtype = modelos.dispositivo()
    if cfg["tipo"] == "ltx":
        from diffusers import LTXConditionPipeline

        pipe = LTXConditionPipeline.from_pretrained(cfg["repo"], torch_dtype=dtype)
    else:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        extra = {"vae": AutoencoderKLWan.from_pretrained(cfg["repo"], subfolder="vae", torch_dtype=torch.float32)}
        if cfg.get("clip"):
            from transformers import CLIPVisionModel

            extra["image_encoder"] = CLIPVisionModel.from_pretrained(
                cfg["repo"], subfolder="image_encoder", torch_dtype=torch.float32
            )
        pipe = WanImageToVideoPipeline.from_pretrained(cfg["repo"], torch_dtype=dtype, **extra)
    return modelos.colocar(pipe, modo_memoria)


def _num_frames(segundos, cfg):
    """Frames válidos para el modelo: múltiplo de `frames_mult` + 1."""
    mult = cfg["frames_mult"]
    k = max(round((float(segundos) * cfg["fps"] - 1) / mult), 2)
    return k * mult + 1


def _clip_ia(cfg, pipe, a, b, prompt, ancho, alto, frames, seed):
    import torch

    gen = torch.Generator("cpu").manual_seed(seed)
    if cfg["tipo"] == "ltx":
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        salida = pipe(
            conditions=[LTXVideoCondition(image=a, frame_index=0), LTXVideoCondition(image=b, frame_index=frames - 1)],
            prompt=prompt,
            negative_prompt=NEGATIVO,
            width=ancho,
            height=alto,
            num_frames=frames,
            frame_rate=cfg["fps"],
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
            generator=gen,
        )
    else:
        salida = pipe(
            image=a,
            last_image=b,
            prompt=prompt,
            negative_prompt=NEGATIVO,
            width=ancho,
            height=alto,
            num_frames=frames,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
            generator=gen,
            output_type="pil",
        )
    return list(salida.frames[0])


def _clip_demo(a, b, frames):
    """Fundido simple entre las dos imágenes: sirve para probar el panel sin IA."""
    return [Image.blend(a, b, i / (frames - 1)) for i in range(frames)]


def generar_video(rutas, prompt, modelo, resolucion, segundos, pausa, modo_memoria, seed, progreso=None):
    """Genera un clip por cada par de imágenes consecutivas y los une. Devuelve la ruta del .mp4."""
    if not rutas or len(rutas) < 2:
        raise ValueError("Necesitas al menos 2 imágenes (por ejemplo: la ruina y la casa terminada).")
    cfg = modelos.VIDEO_MODELS[modelo]
    seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)

    ancho, alto = modelos.RESOLUCIONES[resolucion]
    ancho, alto = ancho // cfg["mod"] * cfg["mod"], alto // cfg["mod"] * cfg["mod"]
    imagenes = [ImageOps.fit(Image.open(r).convert("RGB"), (ancho, alto)) for r in rutas]
    frames = _num_frames(segundos, cfg)
    texto = (prompt or "").strip() or MOVIMIENTO_BASE
    pausa_frames = int(float(pausa) * cfg["fps"])

    pipe = None
    if cfg["tipo"] != "demo":
        if progreso:
            progreso(0, desc="Cargando modelo de video (la primera vez se descarga, puede tardar)")
        memoria = modelos.MEMORIA.get(modo_memoria, "offload")
        pipe = modelos.obtener(("video", cfg["repo"], memoria), lambda: _cargar(cfg, memoria))

    video = [imagenes[0]] * pausa_frames
    pares = len(imagenes) - 1
    for i in range(pares):
        if progreso:
            progreso(i / pares, desc=f"Animando transición {i + 1}/{pares} ({frames} frames)")
        a, b = imagenes[i], imagenes[i + 1]
        if pipe is None:
            clip = _clip_demo(a, b, frames)
        else:
            clip = _clip_ia(cfg, pipe, a, b, texto, ancho, alto, frames, seed + i)
        video.extend(clip if i == 0 else clip[1:])  # evita repetir el frame de unión
        video.extend([clip[-1]] * pausa_frames)

    carpeta = os.path.join(SALIDAS, "videos")
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"reconstruccion_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4")
    _guardar_mp4(video, ruta, cfg["fps"])
    return ruta, seed


def _guardar_mp4(frames, ruta, fps):
    import imageio.v2 as imageio

    with imageio.get_writer(ruta, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(np.asarray(f.convert("RGB") if isinstance(f, Image.Image) else f))
