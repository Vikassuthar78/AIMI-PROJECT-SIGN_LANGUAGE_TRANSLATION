@echo off
cd webapp
call ..\venv\Scripts\activate
streamlit run app.py
pause
