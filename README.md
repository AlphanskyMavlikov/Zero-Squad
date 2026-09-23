# Contractor Matcher V2

## Новое
- Semantic matching: OpenAI `text-embedding-3-small`, а не keyword matching.
- Embeddings работают только после hard filters.
- Полная декомпозиция score.
- ИИ-агент генерирует объяснение из структурированных evidence-данных, текст не захардкожен.
- Воронка причин отсева.
- «Почему не этот подрядчик?» с конкретными причинами.

## Запуск
1. Python 3.10+
2. `pip install -r requirements.txt`
3. Задайте ключ в Windows PowerShell:
   `$env:OPENAI_API_KEY="ваш_ключ"`
4. `streamlit run app.py`

Можно также запустить `run.bat` после задания переменной среды.

## Архитектура
`CSV → город+категория → HARD FILTERS → embeddings → score → TOP 3 → AI explanation`

Hard filters: дата, бюджет, формат, опционально язык и длительность.
Embedding similarity никогда не отменяет hard filters.

## Детерминизм
Сортировка: total score DESC → semantic similarity DESC → id ASC.
При одинаковых данных и моделях порядок стабилен.

## API
API используется только для semantic embeddings и объяснений. Embeddings кэшируются локально в `embeddings_cache.json`, поэтому повторные запросы не пересчитывают уже известные описания.

## Исправление StreamlitSecretNotFoundError

Версия не обращается к `st.secrets`, если файла `secrets.toml` нет.
Ключ читается только из переменной среды `OPENAI_API_KEY`.

### Вариант 1 — PowerShell
```powershell
$env:OPENAI_API_KEY="sk-proj-ВАШ_КЛЮЧ"
streamlit run app.py
```

### Вариант 2 — SET_API_KEY.bat
Запустите `SET_API_KEY.bat`, вставьте ключ в консоль и нажмите Enter.
Ключ намеренно не записан в исходный код или ZIP.
