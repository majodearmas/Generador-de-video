"""Panel local para videos de reconstrucción de casas.

1) Escribes cómo es la casa → genera las imágenes de cada etapa (ruina → terminada).
2) Usas esas imágenes (o subes las tuyas) → genera el video de la transformación.

Todo corre en tu computadora con modelos gratuitos de Hugging Face. Sin APIs de pago.
"""

import os
import traceback

import gradio as gr

import imagenes
import modelos
import video


def _gpu_disponible():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


GPU = _gpu_disponible()
IMG_DEFECTO = list(modelos.IMAGE_MODELS)[0] if GPU else modelos.DEMO
VID_DEFECTO = list(modelos.VIDEO_MODELS)[0] if GPU else modelos.DEMO


def _error(e):
    traceback.print_exc()
    if "out of memory" in str(e).lower():
        return gr.Error(
            "Sin memoria de GPU. Prueba el modo 'Mínima VRAM', la resolución 480p, menos segundos "
            "o un modelo más ligero."
        )
    return gr.Error(f"{type(e).__name__}: {e}")


def accion_imagenes(texto_etapas, modelo, memoria, seed, encadenar, fuerza, foto, progreso=gr.Progress()):
    try:
        rutas, carpeta, seed = imagenes.generar_etapas(
            texto_etapas, modelo, memoria, seed, encadenar, fuerza, foto, progreso
        )
    except Exception as e:
        raise _error(e)
    nombres = [os.path.splitext(os.path.basename(r))[0] for r in rutas]
    return (
        list(zip(rutas, nombres)),
        rutas,
        f"✅ {len(rutas)} imágenes guardadas en `{carpeta}` · semilla {seed}",
    )


def ordenar(archivos):
    rutas = [a if isinstance(a, str) else a.name for a in (archivos or [])]
    return sorted(rutas, key=lambda r: os.path.basename(r).lower())


def accion_video(archivos, prompt, modelo, resolucion, segundos, pausa, memoria, seed, progreso=gr.Progress()):
    try:
        ruta, seed = video.generar_video(
            ordenar(archivos), prompt, modelo, resolucion, segundos, pausa, memoria, seed, progreso
        )
    except Exception as e:
        raise _error(e)
    return ruta, f"✅ Video guardado en `{ruta}` · semilla {seed}"


def liberar():
    modelos.liberar()
    return "Memoria liberada."


with gr.Blocks(title="Estudio local de reconstrucción") as panel:
    gr.Markdown(
        "# 🏚️➡️🏡 Estudio local de reconstrucción de casas\n"
        "Todo corre en tu computadora con modelos gratis de Hugging Face. La **primera vez** que uses "
        "un modelo se descarga (varios GB) y después queda guardado.\n\n"
        f"**Equipo:** {modelos.info_hardware()}"
    )
    memoria = gr.Radio(list(modelos.MEMORIA), value=list(modelos.MEMORIA)[0], label="Uso de memoria")
    rutas_generadas = gr.State([])

    with gr.Tab("1 · Imágenes de las etapas"):
        with gr.Row():
            with gr.Column():
                descripcion = gr.Textbox(
                    label="¿Cómo es la casa? (en inglés da mejores resultados)",
                    value="old two-story colonial house with a wooden porch in a small town",
                )
                num_etapas = gr.Slider(2, 4, value=4, step=1, label="Número de etapas")
                btn_preparar = gr.Button("① Preparar etapas")
                etapas = gr.Textbox(
                    label="Etapas (una por línea: Nombre | descripción). Puedes editarlas.",
                    lines=6,
                    value=imagenes.preparar_etapas("old two-story colonial house with a wooden porch in a small town", 4),
                )
                foto = gr.Image(type="pil", label="Opcional: foto real de la casa en ruinas (será la etapa 1)")
                with gr.Accordion("Ajustes", open=False):
                    modelo_img = gr.Dropdown(list(modelos.IMAGE_MODELS), value=IMG_DEFECTO, label="Modelo de imagen")
                    encadenar = gr.Checkbox(value=True, label="Encadenar: cada etapa parte de la anterior")
                    fuerza = gr.Slider(
                        0.3, 0.9, value=0.6, step=0.05,
                        label="Cuánto cambia cada etapa (bajo = misma casa, alto = más cambios)",
                    )
                    seed_img = gr.Number(value=-1, precision=0, label="Semilla (-1 = aleatoria)")
                btn_img = gr.Button("② Generar imágenes", variant="primary")
            with gr.Column():
                galeria = gr.Gallery(label="Etapas generadas", columns=2, height="auto")
                estado_img = gr.Markdown()
                btn_pasar = gr.Button("③ Usar estas imágenes en el video →")

    with gr.Tab("2 · Video"):
        with gr.Row():
            with gr.Column():
                archivos = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Imágenes en orden (se ordenan por nombre: 01_..., 02_...)",
                )
                prompt_video = gr.Textbox(label="Cómo es el movimiento", value=video.MOVIMIENTO_BASE, lines=3)
                modelo_vid = gr.Dropdown(list(modelos.VIDEO_MODELS), value=VID_DEFECTO, label="Modelo de video")
                resolucion = gr.Dropdown(list(modelos.RESOLUCIONES), value=list(modelos.RESOLUCIONES)[0],
                                         label="Resolución")
                segundos = gr.Slider(2, 10, value=5, step=1, label="Segundos por transición")
                pausa = gr.Slider(0, 3, value=1, step=0.5, label="Pausa en cada etapa (segundos)")
                seed_vid = gr.Number(value=-1, precision=0, label="Semilla (-1 = aleatoria)")
                btn_vid = gr.Button("Generar video", variant="primary")
            with gr.Column():
                salida_video = gr.Video(label="Video final")
                estado_vid = gr.Markdown()

    with gr.Row():
        btn_liberar = gr.Button("Liberar memoria de la GPU", size="sm")
        estado_mem = gr.Markdown()

    btn_preparar.click(imagenes.preparar_etapas, [descripcion, num_etapas], etapas)
    btn_img.click(
        accion_imagenes,
        [etapas, modelo_img, memoria, seed_img, encadenar, fuerza, foto],
        [galeria, rutas_generadas, estado_img],
    )
    btn_pasar.click(lambda r: r, rutas_generadas, archivos)
    btn_vid.click(
        accion_video,
        [archivos, prompt_video, modelo_vid, resolucion, segundos, pausa, memoria, seed_vid],
        [salida_video, estado_vid],
    )
    btn_liberar.click(liberar, None, estado_mem)


if __name__ == "__main__":
    panel.queue().launch(
        server_name="127.0.0.1",
        inbrowser=os.environ.get("NO_ABRIR_NAVEGADOR") is None,
        allowed_paths=[imagenes.SALIDAS],
    )
