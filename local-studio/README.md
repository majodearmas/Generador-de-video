# 🏚️➡️🏡 Estudio local de reconstrucción de casas

Panel sencillo que corre **en tu computadora**, con modelos **gratis de Hugging Face**. No usa APIs de pago
ni necesita internet después de descargar los modelos.

1. **Imágenes:** describes la casa → genera cada etapa (ruina → limpieza → construcción → terminada).
2. **Video:** toma esas imágenes (o las tuyas) → anima la transformación entre cada par y une todo en un `.mp4`.

## Qué necesitas

| | Mínimo para probar | Recomendado |
|---|---|---|
| GPU | NVIDIA 8–12 GB (RTX 3060) | NVIDIA 24 GB (RTX 3090/4090/5090) |
| RAM | 32 GB | 64 GB |
| Disco libre | 40 GB (RealVisXL + LTX) | 150 GB+ si bajas los modelos Wan |
| Software | Python 3.10–3.12 ([python.org](https://www.python.org/downloads/), marca *Add to PATH*) | |

Sin GPU NVIDIA se puede abrir el panel y usar el **modo Demo** para ver el flujo, pero la IA real sería
demasiado lenta en CPU.

## Instalación

**Windows:** doble clic en `instalar.bat` y luego en `iniciar.bat`.

**Linux / Mac:**
```bash
cd local-studio
./instalar.sh
./iniciar.sh
```

El panel se abre en el navegador en <http://127.0.0.1:7860> (solo accesible desde tu PC).

## Modelos (se descargan solos la primera vez que los eliges)

| Paso | Modelo | Hugging Face | Tamaño aprox. | VRAM |
|---|---|---|---|---|
| Imagen | RealVisXL 5 (fotorrealista) | `SG161222/RealVisXL_V5.0` | 7 GB | 8 GB |
| Imagen | FLUX.1 schnell | `black-forest-labs/FLUX.1-schnell` | 34 GB | 16 GB+ |
| Video | LTX-Video 0.9.5 (rápido) | `Lightricks/LTX-Video-0.9.5` | 25 GB | 12 GB |
| Video | Wan 2.1 FLF2V 14B 720p | `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` | 75 GB | 24 GB |
| Video | Wan 2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 125 GB | 24 GB + 64 GB RAM |

Todos los modelos de video usan **primer y último fotograma**: el clip empieza exactamente en la imagen
de una etapa y termina exactamente en la siguiente.

Para descargarlos antes (por ejemplo, de noche):
```bash
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
python descargar_modelos.py
```

## Cómo usarlo

1. **Pestaña 1:** escribe la casa (en inglés sale mejor, ej. `old adobe house with a red tile roof`) →
   **Preparar etapas** → revisa o edita las descripciones → **Generar imágenes**.
   - ¿Tienes una foto real de una casa en ruinas? Súbela y será la etapa 1.
   - Si las etapas no parecen la misma casa, baja *"Cuánto cambia cada etapa"* (Ajustes).
   - Si te gusta un resultado, anota la **semilla** para repetirlo.
2. **Usar estas imágenes en el video →**, y luego en la **pestaña 2** elige el modelo → **Generar video**.
   - Puedes subir tus propias imágenes: se ordenan por nombre (`01_...`, `02_...`).
   - Empieza con **480p y 5 segundos**; cuando el resultado te guste, pasa a 720p.

Todo se guarda en `local-studio/salidas/`.

## Si algo falla

- **"Sin memoria de GPU":** elige *Mínima VRAM*, 480p, menos segundos o un modelo más ligero.
- **Tarda mucho la primera vez:** está descargando el modelo; mira la consola.
- **Tiempos aproximados con una RTX 4090:** imagen SDXL ~10 s; clip LTX de 5 s ~1–2 min; clip Wan 14B
  de 5 s a 720p ~15–30 min.

## Licencias

Cada modelo tiene su propia licencia en su página de Hugging Face. FLUX.1 schnell y Wan
son Apache 2.0 (permiten uso comercial). Revisa en su página de Hugging Face la licencia de RealVisXL y de
LTX-Video antes de monetizar los videos.
