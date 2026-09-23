@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".streamlit" mkdir ".streamlit"
echo.
echo Вставьте НОВЫЙ OpenAI API key.
echo Он сохранится локально и сайт сейчас НЕ запустится.
echo.
set /p APIKEY=API key: 
if "%APIKEY%"=="" (
 echo ОШИБКА: ключ не введен.
 pause
 exit /b 1
)
echo OPENAI_API_KEY = "%APIKEY%" > ".streamlit\secrets.toml"
echo.
echo Ключ сохранен. Теперь запустите ЗАПУСТИТЬ.bat
pause
