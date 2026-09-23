@echo off
echo Введите OpenAI API key в этом окне. Он не будет сохранен в проекте.
set /p OPENAI_API_KEY=OPENAI_API_KEY: 
python -m streamlit run app.py
pause
