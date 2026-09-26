# Task025 — миграция жизненного цикла документов

Дата проверки: 2026-09-26. Исходная ветка `main`, HEAD `756b8943a0bf3dff1b187755667707609827fbe5`; исходное рабочее дерево чистое.

## Реестр и происхождение

Принята строгая схема v2. Все 12 записей относятся к кампании 2026 года; версии — 1, статусы — active. `supersedes`, `published_at`, `effective_from`, `effective_to` у всех равны `null`. Даты не выводились из имён файлов. Аудит на 2026-09-26 не обнаружил предупреждений.

| Source ID | Logical document ID | Version | Status | Table aware |
|---|---|---:|---|---|
| `admissions-bachelor` | `admissions-bachelor-page` | 1 | active | false |
| `admission-documents` | `admission-documents-page` | 1 | active | false |
| `admission-rules` | `admission-rules-page` | 1 | active | false |
| `entrance-exams` | `entrance-exams-page` | 1 | active | false |
| `admissions-faq` | `admissions-faq-page` | 1 | active | false |
| `tuition` | `tuition-page` | 1 | active | false |
| `admission-rules-pdf` | `admission-rules-document` | 1 | active | false |
| `admission-capacity-pdf` | `admission-capacity-document` | 1 | active | true |
| `admission-deadlines-pdf` | `admission-deadlines-document` | 1 | active | false |
| `entrance-exams-list-pdf` | `entrance-exams-list-document` | 1 | active | true |
| `entrance-exam-regulations-pdf` | `entrance-exam-regulations-document` | 1 | active | false |
| `tuition-order-128-pdf` | `tuition-fees-document` | 1 | active | true |

Статус и даты отделены от постоянной версии. Нормализованные документы, чанки и метаданные индексов используют v2; public API, ID чанков и архитектурный идентификатор `dual-channel-pdf-page-diversity-v1` сохранены. При изменении активной версии прежние чанки/индекс отклоняются до использования.

## Сверка официальных источников

До замены рабочих артефактов повторно получены все 12 источников через прежнюю проверку домена и каждого redirect. У шести HTML совпал content SHA-256, у шести PDF — file SHA-256. Все конечные URL совпали; нормализованные документы и старые чанки проверены строго. Ни один источник не обновлён по смыслу.

В таблице HTML сравнивается по извлечённому тексту, PDF — по точным байтам. Приведённый хеш одинаков у принятого и полученного содержимого.

| Source ID | Сравнение | SHA-256 до = после | Результат |
|---|---|---|---|
| `admissions-bachelor` | HTML content | `00c931722ad666b2446e3e9bbfe4ea121dcdc72cf1b0b27a8db616126d0f3609` | совпадает |
| `admission-documents` | HTML content | `24bc133284f651684c4d08a0d3eacca5d12684dbfeb364d0b287544ad50aac25` | совпадает |
| `admission-rules` | HTML content | `f9c68c8b2d35321568f015f7ffa6a7569c348e4361af4219c676d251da973321` | совпадает |
| `entrance-exams` | HTML content | `6fc0e9c5a6f3ab2ef6086232262983ce2f0a2d703df5aa1bbe6c902e67c8842b` | совпадает |
| `admissions-faq` | HTML content | `18b95f50f5493f2c57cd1e3a3ec4f0391e3afb2afdb4739deb7a17e5fdc52504` | совпадает |
| `tuition` | HTML content | `e4732f98ee7a5b559246eb658ac182141e5874b7421068ff9c3a0642ab0f7e02` | совпадает |
| `admission-rules-pdf` | PDF file | `c853cc2cb723e34d55db1b5544cb152f5474a698dd314e9eca1df7e57b8a73da` | совпадает |
| `admission-capacity-pdf` | PDF file | `748b164c0d1a5e7c82d7fcc4e8e417334be73f062b02c0bed5245e66ae2850c6` | совпадает |
| `admission-deadlines-pdf` | PDF file | `6afb22ce0d2c2b5f075512cf71c3c03e57883a31f4e5385afbe8aa5f0166c3d2` | совпадает |
| `entrance-exams-list-pdf` | PDF file | `656ccb00805760f0685c703587c0d5ae5f6dad9ca11ed098f72e6a2180f3d4d7` | совпадает |
| `entrance-exam-regulations-pdf` | PDF file | `a758edde3c7c25cee792cd8c7d16f4e3f5111843c70a90b9109937aaa4c80dba` | совпадает |
| `tuition-order-128-pdf` | PDF file | `455d11c699f84b20739df78aa830bd37ff091b3a13ad776eb11f466a5893544e` | совпадает |

## Снимки

Сохранены шесть HTML-снимков в `data/processed/originals/html/` и шесть PDF в `data/processed/originals/pdf/`. HTML — точная декодированная строка загрузчика в UTF-8; PDF — исходные байты. Все 12 пар проверены после переноса. Повторное извлечение HTML воспроизводит нормализованные текст и заголовок; PDF snapshot SHA равен file SHA.

| Source ID | Snapshot SHA-256 |
|---|---|
| `admissions-bachelor` | `c6a810197c26e191d5611e37bedd6deaea70130b3eecc94f9b6667a952244504` |
| `admission-documents` | `84df97f5df5cc272184cf3951f0c5fd9ccc107716f46c431797e89fbb7719641` |
| `admission-rules` | `05367072497f8e918a0a2d9b752c86b52324f50e6c7795122f64989c7d7b71e7` |
| `entrance-exams` | `a28d4536a97465ecd430009366e6ee7f5baa52bf1569b061dd896a681b110ce3` |
| `admissions-faq` | `abbbc9fdd5384041dd54f0433af3c4f0ce86491970d99cd3a803b693dbdc3c37` |
| `tuition` | `040739057c933b49dca01c808f97e14fa6dd64003f8918a953a7197785b43dd5` |
| `admission-rules-pdf` | `c853cc2cb723e34d55db1b5544cb152f5474a698dd314e9eca1df7e57b8a73da` |
| `admission-capacity-pdf` | `748b164c0d1a5e7c82d7fcc4e8e417334be73f062b02c0bed5245e66ae2850c6` |
| `admission-deadlines-pdf` | `6afb22ce0d2c2b5f075512cf71c3c03e57883a31f4e5385afbe8aa5f0166c3d2` |
| `entrance-exams-list-pdf` | `656ccb00805760f0685c703587c0d5ae5f6dad9ca11ed098f72e6a2180f3d4d7` |
| `entrance-exam-regulations-pdf` | `a758edde3c7c25cee792cd8c7d16f4e3f5111843c70a90b9109937aaa4c80dba` |
| `tuition-order-128-pdf` | `455d11c699f84b20739df78aa830bd37ff091b3a13ad776eb11f466a5893544e` |

Старых HTML-снимков не было: изменение оформления относительно исторической загрузки установить нельзя. Снимки v2 фиксируют проверенный текущий вход. Изменение байтов существующего source ID, в том числе только оформления, не разрешает его перезапись.

## Проверка миграции и поиска

Резервная копия прежних документов, чанков, обоих индексов и манифеста сохранена в игнорируемом `data/processed/task025/backup/`. Новые артефакты подготовлены отдельно в `data/processed/task025/candidate/`; рабочие файлы заменены только после полного сравнения.

- 338/338 чанков: точное совпадение ID, текста, хешей, порядка, источников и страниц.
- 250/250 строк: полное совпадение извлечённых артефактов Task014 и записей индекса. Распределение: capacity 75, entrance exams 93, tuition 82.
- Оба индекса действительно пересобраны BGE-M3, размерность 1024, batch 16. Таблицы извлечены из локальных снимков прежним `lines_strict`, PyMuPDF 1.28.2; скачивания при сборке таблиц нет.
- Настройки моделей сохранены: BGE-M3 и bge-reranker-v2-m3, reranker batch 8, max length 512, каналы 20+5, ранжирование 25, квота страницы 2, итог 5. `auto` выбрал CPU для обеих моделей. Использован локальный кеш с отключённым доступом Hugging Face.
- Байты обоих пересобранных FAISS-файлов совпали с прежними. Производственные записи также совпали; следовательно, основной top-20 при прежнем запросном эмбеддинге сохранён.
- Канонический прогон: 80/80 top-5 совпали с принятым Task018 по ID, порядку, источнику, странице и reranker score с допуском 1e-4. Замороженные метрики воспроизведены.

| Метрика | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| primary | 63/80 | 70/80 | 74/80 | 0.83625 |
| accepted | 65/80 | 71/80 | 74/80 | 0.8545833333333333 |
| page | 36/56 | 45/56 | 49/56 | 0.7273809523809524 |

Диагностические задержки одного запроса в каноническом прогоне (секунды; без порога приёмки):

- mean_seconds: 7.004349.
- median_seconds: 6.926035.
- p95_seconds: 7.593506.
- min_seconds: 5.878374.
- max_seconds: 14.904325.

SHA-256 файлов FAISS до и после совпадает:

- `index`: `926c252595aad9b881eb5a72df639a78ec97eb068df27d4178f555bd03bb454a`.
- `table_index`: `972187257fe331377666c65c2141f8360119fe3bafa491da8b89212ed3314698`.

## Заключительный freshness

Аудит выполнен после переноса: `2026-09-26T13:21:25.757677+00:00`. Код завершения 0. Все 12 active источников — `unchanged`. `presentation_changed`, `content_changed`, `file_changed`, `redirect_changed`, `missing_local_artifact`, `error` — по 0. Снимки и корпус аудитом не изменялись.

Локальные отчёты: `data/processed/task025/candidate/migration_comparison.json`, `data/processed/task025/table_comparison.json`, `data/processed/task025/evaluation/canonical_retrieval.json` и `data/processed/freshness/report.json`.

## Проверки и целостность

- Backend: 447 тестов прошли; Ruff — без ошибок.
- Frontend: 70 тестов прошли; lint и production build — без ошибок.
- `git diff --check` — без ошибок.
- 45 исторических файлов оценки и датасетов сохранили исходные хеши. Производственный промпт, API, генератор, frontend и зависимости не изменены. Исторические эксперименты получили только совместимость с полем статуса реестра.
- Снимки, документы, чанки, индексы, резервная копия и отчёты JSON остаются под игнорируемым `data/processed/`.

Замороженные датасеты:

- `retrieval_questions.json`: `52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e`.
- `generation_questions.json`: `11a284b4279f627ae787d0ec4777e0884733b4ec937d6658be3857c2dd15d0a4`.

## Ограничения и следующий этап

Генерация и Ollama не запускались: точная эквивалентность retrieval подтверждена; новых семантических оценок нет. Известные ограничения исторических оценок не пересмотрены. Начальная загрузка моделей и холодный запуск остаются прежними.

Это однократная миграция и основа проверки актуальности, а не атомарное переключение всего корпуса. Сбой между публикациями двух файлов выявляется проверкой пары; механизм управляемой публикации и отката ещё не реализован. Реестр не активирует версии по датам и не обновляется при freshness.

Proceed to Task026: controlled incoming-document staging, validation, atomic publication, and rollback.
