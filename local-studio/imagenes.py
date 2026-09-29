"""Paso 1: generar las etapas de la reconstrucción (ruina → ... → casa terminada)."""

import math
import os
import random
import unicodedata
from datetime import datetime

from PIL import Image, ImageDraw, ImageOps

import modelos

ESTILO = (
    ", photorealistic, architectural photography, wide shot, same camera angle, "
    "eye level, natural light, highly detailed, 8k"
)
NEGATIVO = "cartoon, painting, illustration, blurry, low quality, distorted, text, watermark, people close-up"

ETAPAS_BASE = [
    ("Ruina", "abandoned {casa}, ruined, broken windows, damaged roof, peeling paint, "
              "overgrown weeds, debris everywhere, dirty, decay"),
    ("Limpieza y demolición", "{casa} being renovated, debris removed, scaffolding, dumpster with rubble, "
                              "exposed brick, old windows removed"),
    ("En construcción", "{casa} under construction, new roof being installed, fresh plaster on walls, "
                        "scaffolding, new window frames, construction materials"),
    ("Terminada", "beautifully renovated {casa}, brand new, fresh paint, new windows, new roof, "
                  "clean manicured garden, warm interior lights, golden hour"),
]

SALIDAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")


def preparar_etapas(descripcion, num_etapas):
    """Construye el texto editable de etapas: una por línea, 'Nombre | prompt'."""
    casa = (descripcion or "").strip() or "two-story colonial house"
    num = int(num_etapas)
    if num <= 2:
        elegidas = [ETAPAS_BASE[0], ETAPAS_BASE[-1]]
    elif num == 3:
        elegidas = [ETAPAS_BASE[0], ETAPAS_BASE[2], ETAPAS_BASE[3]]
    else:
        elegidas = ETAPAS_BASE
    return "\n".join(f"{nombre} | {p.format(casa=casa)}" for nombre, p in elegidas)


def leer_etapas(texto):
    etapas = []
    for linea in (texto or "").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        nombre, _, prompt = linea.partition("|")
        if not prompt:
            nombre, prompt = f"Etapa {len(etapas) + 1}", nombre
        etapas.append((nombre.strip(), prompt.strip()))
    return etapas


def _cargar_txt2img(cfg, modo_memoria):
    import torch
    from diffusers import AutoPipelineForText2Image

    _, dtype = modelos.dispositivo()
    if cfg["tipo"] == "sdxl" and dtype == torch.bfloat16:
        dtype = torch.float16
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(cfg["repo"], torch_dtype=dtype, variant="fp16")
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(cfg["repo"], torch_dtype=dtype)
    return modelos.colocar(pipe, modo_memoria)


def _generar_ia(cfg, modo_memoria, prompt, base, fuerza, seed):
    import torch
    from diffusers import AutoPipelineForImage2Image

    clave = ("img", cfg["repo"], modo_memoria)
    pipe = modelos.obtener(clave, lambda: _cargar_txt2img(cfg, modo_memoria))
    ancho, alto = cfg["size"]
    kwargs = dict(
        prompt=prompt + ESTILO,
        guidance_scale=cfg["guidance"],
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    if cfg["tipo"] == "sdxl":
        kwargs["negative_prompt"] = NEGATIVO
    else:
        kwargs["max_sequence_length"] = 256

    if base is None:
        return pipe(width=ancho, height=alto, num_inference_steps=cfg["steps"], **kwargs).images[0]

    # img2img reutiliza los mismos pesos ya cargados (no descarga ni ocupa más memoria).
    # Se suben los pasos para que, tras aplicar la fuerza, se ejecuten ~los mismos pasos útiles.
    i2i = AutoPipelineForImage2Image.from_pipe(pipe)
    pasos = math.ceil(cfg["steps"] / max(fuerza, 0.05))
    return i2i(image=base.resize((ancho, alto)), strength=fuerza, num_inference_steps=pasos, **kwargs).images[0]


def _generar_demo(cfg, nombre, indice, total):
    """Imagen de relleno para probar el flujo del panel sin descargar modelos."""
    ancho, alto = cfg["size"]
    t = indice / max(total - 1, 1)
    cielo = (int(90 + 100 * t), int(90 + 110 * t), int(100 + 130 * t))
    img = Image.new("RGB", (ancho, alto), cielo)
    d = ImageDraw.Draw(img)
    d.rectangle([0, alto * 0.75, ancho, alto], fill=(int(80 + 40 * t), int(70 + 90 * t), 50))
    muro = (int(110 + 130 * t), int(100 + 120 * t), int(90 + 100 * t))
    x0, y0, x1, y1 = ancho * 0.3, alto * 0.4, ancho * 0.7, alto * 0.78
    d.rectangle([x0, y0, x1, y1], fill=muro)
    techo = (int(70 + 110 * t), int(40 + 20 * t), int(30 + 20 * t))
    d.polygon([(x0 - 30, y0), (ancho / 2, alto * 0.18), (x1 + 30, y0)], fill=techo)
    for fx in (0.36, 0.58):
        d.rectangle([ancho * fx, alto * 0.5, ancho * (fx + 0.06), alto * 0.62],
                    fill=(40, 40, 40) if t < 0.5 else (250, 220, 120))
    d.text((20, 20), f"DEMO · {nombre}", fill=(255, 255, 255))
    return img


def generar_etapas(texto_etapas, modelo, modo_memoria, seed, encadenar, fuerza, foto_base, progreso=None):
    """Genera todas las etapas. Devuelve (lista de rutas, carpeta de salida)."""
    etapas = leer_etapas(texto_etapas)
    if not etapas:
        raise ValueError("Escribe al menos una etapa (usa el botón 'Preparar etapas').")
    cfg = modelos.IMAGE_MODELS[modelo]
    seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)
    carpeta = os.path.join(SALIDAS, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(carpeta, exist_ok=True)

    primera = None
    anterior = None
    rutas = []
    for i, (nombre, prompt) in enumerate(etapas):
        if progreso:
            progreso(i / len(etapas), desc=f"Generando '{nombre}' ({i + 1}/{len(etapas)})")
        if i == 0 and foto_base is not None:
            # Tu foto real es la primera etapa; las demás se construyen a partir de ella.
            img = ImageOps.fit(foto_base.convert("RGB"), cfg["size"])
        elif cfg["tipo"] == "demo":
            img = _generar_demo(cfg, nombre, i, len(etapas))
        else:
            base = anterior if encadenar else primera
            img = _generar_ia(cfg, modelos.MEMORIA.get(modo_memoria, "offload"), prompt, base, float(fuerza), seed)
        if primera is None:
            primera = img
        anterior = img
        ruta = os.path.join(carpeta, f"{i + 1:02d}_{_seguro(nombre)}.png")
        img.save(ruta)
        rutas.append(ruta)
    return rutas, carpeta, seed


def _seguro(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return "".join(c if c.isalnum() else "_" for c in texto)[:40]
