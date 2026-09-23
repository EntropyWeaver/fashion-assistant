@echo off
echo ⚙️ Activando entorno virtual...
call venv\Scripts\activate

echo 🚀 Iniciando backend con FastAPI...
start cmd /k "uvicorn app.api.main:app --reload"

timeout /t 2

echo 🎨 Iniciando frontend con Streamlit...
start cmd /k "venv\Scripts\activate && streamlit run streamlit_app.py"
