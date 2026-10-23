# Змістовна історія migration-bugfixes

Дата: 2026-10-23. Це реконструкція накопиченої роботи, не нове навчання моделі.

## Межі операції

Основа — `cc6797135067ade7e1ad38c328029b411003be9e`, попередня вершина
`migration-bugfixes`. Її історія збережена. Перевірений результат очищення
`3917686450dcf6659a916fe930de4759e7209f05` розкладено на тематичні коміти.
Коміт-дамп `aa9f8f5` із `migration-bugfixes-allwithtemp` і технічні backup-коміти
не є предками нової серії. `main` ця операція не змінює.

Перенесення зроблено за залежностями модулів. Це логічні етапи для review і
bisect, а не спроба видати їх за фактичну хронологію всіх експериментів.

## Серія

1. Спільний artifact I/O, потрібні конфігурації джерел і видалення obsolete entrypoints.
2. Виміряні кандидати FlyWire, походження даних і прибирання старих синтетичних входів.
3. ToF geometry, версійні контракти вчителя й навчальних даних.
4. Функціональні probes, аналіз recurrent dynamics і frozen-core continuation.
5. XY coordinator, STOP/optional exploration і перегляд точного checkpoint.
6. NumPy runtime, assembly, різночастотні сенсори й authority handover.
7. Session receiver, transport, fallback reference і автономний offline release.
8. Точний k306 checkpoint, компактні bundles і вибрані підсумкові докази.
9. Поточна навігація, пояснення рішень, cleanup та цей запис історії.

Коміти містять відповідні тести. Кожний стан перевірено в окремому checkout
через повний доступний на цьому етапі `pytest tests`. Середовище — наявний
Windows Python 3.12 venv; це не свіжа установка залежностей і не тест Raspberry Pi.
Точні SHA комітів і результати перших восьми етапів —
[history_rebuild_validation.json](history_rebuild_validation.json).

## Що не переносилося

231 файл старих діагностичних і відбіркових прогонів (93 133 824 байти Git blobs)
не додається до нової історії. Це outputs, а не вихідний код. Перелік із
Git blob IDs — [historical_artifacts_manifest.json](historical_artifacts_manifest.json).
Поточні інтеграційні summaries, обраний checkpoint із graph/report і NumPy release
залишені. Історичні markdown можуть посилатися на outputs, доступні лише локально.

Вихідний код моделі, фільтрів, контролерів і перевірок не змінювався відносно
перевіреного очищеного стану. Новими є лише організація історії та її документація.
Старі досліди з інших checkpoints потребують окремих локальних ваг і датасетів,
як і до цієї операції.

## Відновлення

Локально, поза Git checkout:

- `archive/history_rebuild_20261023/pre_series.bundle` — повна попередня Git-історія.
- `archive/history_rebuild_20261023/omitted_working_files.zip` — 231 файл у точних
  початкових working-tree bytes; `omitted_working_inventory.json` містить SHA-256.
- `archive/pre_cleanup_20261023/workspace_and_data.zip` — попередній резерв
  усього коду й даних першої серії очищення.

Git bundle перевірено через `git bundle verify`, ZIP — за CRC та SHA-256 файлів.
Архіви локальні, на тому самому диску. Вони не публікуються разом із кодом.
Зникнення назви тимчасової гілки не означає фізичного знищення відновлюваних
Git objects. Для історичного відновлення можна клонувати bundle в новий каталог.

## Подальша інтеграція

`migration-bugfixes` продовжується від старої вершини звичайним fast-forward.
Для перенесення в `main` варто зберегти цю серію через звичайний PR merge,
якщо потрібні окремі коміти; squash знову об'єднає їх в один.
Рішення про merge у `main` не входить до цієї операції.
