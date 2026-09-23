# UNEcon RAG Assistant

**Тема ВКР:** «Разработка справочного модуля с контекстным дополнением для абитуриентов».

Будущая система будет помогать абитуриентам находить ответы в официальных материалах университета и показывать источник информации.

## Текущее состояние

Проект содержит frontend, backend API и ручную загрузку утверждённых HTML-страниц СПбГЭУ в локальные нормализованные документы. Поиск по документам, RAG и языковая модель пока не реализованы.

## Требования

- Python 3.12
- Node.js 24.x и npm 11.x
- GPU не требуется

Основные команды ниже приведены для Windows PowerShell.

## Настройка окружения

Из корня репозитория создайте локальный файл настроек:

```powershell
Copy-Item .env.example .env
```

Файл `.env.example` содержит development defaults для backend и адреса API frontend. Локальный `.env` не коммитится. Для другого адреса API измените `VITE_API_BASE_URL` в локальном `.env`.

## Backend

Из корня репозитория создайте и активируйте виртуальное окружение, затем установите зависимости:

```powershell
py -3.12 -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
Set-Location backend
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Запустите сервер:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API health-check: `http://localhost:8000/api/v1/health`.

Для backend-проверок откройте отдельное окно PowerShell из корня репозитория:

```powershell
.\backend\.venv\Scripts\Activate.ps1
Set-Location backend
python -m pytest -q
ruff check .
```

### Официальные источники HTML

Файл `data/source_manifest.json` содержит утверждённый список официальных страниц для приёма 2026 года. Загрузчик проверяет весь манифест до сетевых запросов и получает только явно перечисленные адреса `unecon.ru`. Переходы на другие домены запрещены; ссылки внутри страниц не обходятся. В Task002 поддерживаются только HTML-страницы, без PDF.

Из каталога `backend/` после установки зависимостей загрузите одну страницу:

```powershell
python -m app.ingestion.cli fetch --source-id admissions-bachelor
```

Для обработки всех активных источников:

```powershell
python -m app.ingestion.cli fetch
```

Результат сохраняется в `data/processed/html/<source-id>.json`: текст, SHA-256 текста, исходный и конечный адреса, категория и год приёма. Этот каталог содержит загруженные данные и намеренно не отслеживается Git. Команда печатает результат по каждому источнику и завершает работу с ненулевым кодом при ошибке.

## Frontend

В отдельном окне PowerShell из корня репозитория установите зависимости и запустите frontend:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Откройте `http://localhost:5173`. Экран проверит backend и покажет состояние подключения.

Проверки frontend:

```powershell
npm run lint
npm run build
```

## Структура

- `backend/` — FastAPI приложение, конфигурация и focused-тесты.
- `frontend/` — React-приложение на TypeScript и Vite.
- `data/source_manifest.json` — утверждённые источники; `data/processed/html/` — локальные результаты загрузки.
- `docs/` — архитектурные принципы и целевой концептуальный поток.

Во время разработки приложения запускаются локально. Конфигурация адресов отделена от исходного кода, поэтому проект не привязан к компьютеру разработчика и допускает последующее серверное развёртывание. Конкретная production-инфраструктура здесь не задаётся.
