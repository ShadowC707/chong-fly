# Перевірка трьох зауважень до Кроку 4

Дата: 2026-10-23. Перевірено поточний код після очищення, а не лише історичний
звіт `transport_loop_v3`. Ваги, controllers, budgets і sealed пакети незмінні.

## Склад автономного пакета: зауваження вже закрите

`k306_tau025_hold_v1` — історичний bundle Кроку 3. Поточний inference bundle —
`k306_tau025_integrated_v1`, конфігурація — `configs/mvp_runtime_v3.json`.
Повний переносимий offline release — `data/releases/k306_tau025_offline_v1`.
Його `release.json` уже містить `session_receiver.py`, `transport.py` та `run.py`
з SHA-256. `edge/release.py` копіює їх і виконує ізольований self-check **до**
публікації. `edge/export_numpy.py` відповідає за нейронний/runtime bundle,
а не за весь receiver/transport application.

Поточний release повторно пройшов self-check: 40 прийнятих команд, дві сесії,
watchdog, відхилення старої сесії, без заборонених імпортів. Тест
`test_portable_release_runs_outside_repository_and_rejects_corruption` також
перевіряє відмову при пошкодженні саме `session_receiver.py`.
Старий пакет не переписуємо та не доповнюємо заднім числом.

## Ненульовий заморожений інтегратор: контракт збережено

`freeze_sync_v1` забороняє накопичення інтеграла без AUTO, але зберігає вже
накопичений стан. Це прямо перевіряє
`test_shadow_handover_tracking_and_auto_cancellation`. Його значення у SHADOW
впливає на preview, але **не означає прийняту команду**: lease відсутній.

При watchdog fault саме SessionReceiver защіпає заборону керування. Pi дізнається
про неї з наступного доставленого кадру; миттєве обнулення Pi в момент відмови
приймача без каналу повідомлення неможливе. Відновлення після такого fault
потребує MANUAL/reset і нової сесії зі свіжим FlightSystem. Інтеграл і hidden state
тоді нульові; це перевіряють фізичні сценарії повторного входу.

Тому ненульовий `shadow_integral_max_abs` у старому звіті не доводить windup
чи витік команд. Просте AUTO → SHADOW без receiver fault є іншою ситуацією:
сам FlightBridge допускає наступний admission без повного session reset.
Якщо змінювати політику збереження інтеграла для такого переходу, потрібен
окремий парний дослід переходів, а не підгонка метрики до нуля.
Зміна контракту `freeze_sync_v1` для цього аудиту не потрібна.

## Час Pi: додано явну модель

`evaluate.validate_transport_loop` приймає `--pi-compute-delay-s` (типово 0).
Один FIFO worker обслуговує отримані кадри послідовно; completion перевіряється
кожні 4 мс. Нове завдання за зайнятим worker чекає, а не обчислюється паралельно.
Control tick і перевірка віку сенсорів виконуються на completion; тільки після
цього відповідь потрапляє в downlink. Старі session IDs не замінюються новими.

Це спрощення: runtime математично обчислюється на completion, а не протягом
фізичного інтервалу. Є фіксований service time та квантування, немає моделі ОС,
serialization, CPU contention чи виміряного Pi execution time. Результат
`benchmark.py` для policy/runtime не дорівнює автоматично повному часу
`FlightSystem.tick` із acquisition, IPC і передаванням.

Траса додає час arrival/start/due для Pi; наявний `issue_global_s` — фактичний
completion. Метрики окремо показують queue wait і service elapsed. Перевірка
строгого round trip `<20 мс` враховує 1 нс чисельної похибки, щоб значення
`19.9999999999 мс` не зараховувалося як запас перед 20 мс.

До запуску зафіксовано [план](../data/diagnostics/pi_compute_v1/plan.json):
ті самі 14 сценаріїв, seeds і пороги; два service time — 0 та 2 мс.
Підсумок: [comparison.json](../data/diagnostics/pi_compute_v1/comparison.json).
Повні JSONL, графіки й report кожного запуску залишаються локальними outputs.

```powershell
.\venv\Scripts\python.exe -m evaluate.validate_transport_loop --config configs/mvp_runtime_v3.json --out-dir data/diagnostics/compute_2ms_repeat --pi-compute-delay-s 0.002
```

Код виходу 1 означає, що хоча б один критерій сценаріїв не виконано; звіт
при цьому зберігається. Це навмисно: latency assumption не повинне автоматично
послаблювати admission або вимоги до результату.

## Результат і практичний висновок

| Сценарії | 0 мс Pi | 2 мс Pi (4 мс після квантування) |
|---|---|---|
| normal | PASS, round trip 8 мс | PASS, round trip 12 мс |
| bounded_jitter | PASS, до 16 мс | FAIL, sequence rejection ще до AUTO |
| maneuver_latency, maneuver_held | PASS, 16 мс | FAIL, sequence rejection ще до AUTO |
| Інші 10 сценаріїв | PASS | PASS |
| Разом | **14/14** | **11/14** |

У трьох невдалих сценаріях `accepted_count=0`: це не успішний маневр з гіршою
висотою, а відсутність admission. Доставка на межі наступного 20-мс кадру вже
не відповідає `latest_sensor_sequence`: новий кадр оголошується **перед** drain
мережі. Це окреме обмеження від TTL lease у 40 мс. Саме захист приймача відхиляє
відповідь; виправлення float-порівняння у звіті не спричинило ці відмови.

Нульовий варіант зберігає всі фізичні рядки історичної серії v3 точно. Для
ненульового варіанта перевірки старої серії залишилися незмінними, включно
з реальним досягненням маневру. Відсутність аварій у сценарії, де AUTO не
ввімкнулось, не є доказом працездатності автономного польоту.

TDD: шість нових cases спочатку впали через відсутні compute queue/budget
helpers; після реалізації пройшли. Додано наскрізний тест фізичного runner:
arrival 4 мс → issue 8 мс, відповідний вік сенсора, відмова на простроченому
вході при service time 44 мс. Повний suite: **807 passed за 83.56 с**.

Наступний потрібний факт — виміряний бюджет цілого циклу на цільовій Pi та
каналі зв'язку, з розподілом затримок, а не лише середній inference time.
Після нього вирішується питання швидшого обслуговування/каналу або окремо
обґрунтованої зміни протоколу. Тут не збільшували TTL, не дозволяли старі
кадри й не змінювали ваги. Перебудова автономного release не потрібна:
змінений фізичний evaluator не входить у його склад.
