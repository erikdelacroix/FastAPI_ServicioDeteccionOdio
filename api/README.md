# API de Detección de discurso de odio

API REST (FastAPI) que expone un modelo fine-tuned (BETO) para clasificar textos en:
- `no_discriminatorio`
- `racismo_xenofobia`
- `machismo_sexismo`
- `homofobia_LGTBIQ`

## Requisitos

- **Python 3.11** (recomendado para evitar compilaciones de `pydantic-core`)
- Entorno virtual activo con dependencias instaladas:

```powershell
# Crear y activar (Windows PowerShell)
py -3.11 -m venv .venv_api
.\.venv_api\Scripts\Activate.ps1
pip install -U pip
pip install -r api\requirements_api.txt
```


## Arranque
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload --log-level debug
```

## Link
http://127.0.0.1:8000/docs#/