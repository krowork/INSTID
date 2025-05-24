--inbrowser    		Automatically open the url in browser, if --share is used, the public url will be automatically open instead

--server_port    	Choose a specific server port, default=7860 (example --server_port 420    so the local url will be:  http://127.0.0.1:420)

--share				Creates a public URL

--model_path		Name of the sdxl model from huggingface   (the default model example: --model_path stablediffusionapi/juggernaut-xl-v8     diffuser model you can find here: https://huggingface.co/stablediffusionapi/juggernaut-xl-v8

--medvram			Medium vram settings, uses around 13GB, max image resolution of 1024

--lowvram			Low vram settings, uses a bit less than 13GB, max image resolution of 832



---
title: InstantID
emoji: 😻
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.15.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# InstantID Web Interface

Una interfaz web para [InstantID](https://github.com/InstantX/InstantID), un método para generar imágenes que preservan la identidad de una persona en segundos.

## Requisitos

- Python 3.8+
- CUDA compatible GPU (recomendado)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/krowork/INSTID.git
cd INSTID
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar la interfaz web:
```bash
python gradio_interface.py
```

2. Abrir el navegador en la URL mostrada (generalmente `http://localhost:7860`)

3. En la interfaz web:
   - Subir una imagen con un rostro claro y visible
   - Ajustar el prompt de texto para describir la imagen deseada
   - Ajustar los parámetros según sea necesario:
     - Número de pasos (20-100)
     - Fuerza de IdentityNet (0.0-1.5)
     - Fuerza del Adapter (0.0-1.5)
   - Hacer clic en "Submit" para generar la imagen

## Parámetros

- **Número de pasos**: Controla la cantidad de pasos de inferencia. Más pasos = mejor calidad pero más tiempo.
- **Fuerza IdentityNet**: Controla cuánto se preservan los rasgos faciales. Mayor valor = más similitud facial.
- **Fuerza Adapter**: Controla la influencia del adaptador de identidad. Mayor valor = más influencia de la imagen original.

## Notas

- La primera ejecución descargará automáticamente los modelos necesarios (~5GB)
- Se recomienda usar una GPU para mejor rendimiento
- Las imágenes generadas se mostrarán en la interfaz web
