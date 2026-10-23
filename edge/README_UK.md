# NumPy runtime для Raspberry Pi

**Актуальний стан після структурного завершення:** config
[`mvp_runtime_v3.json`](../configs/mvp_runtime_v3.json), bundle
`data/edge/k306_tau025_integrated_v1`, автономний offline release
`data/releases/k306_tau025_offline_v1`. Повна карта запусків і залишкових
обмежень — [MODULE_INTEGRATION_UK.md](../docs/MODULE_INTEGRATION_UK.md).
Описані нижче v1/v2-пакети — незмінні історичні версії. Для поточного
repository `build_system` використовуйте v3; старі конфіги призначені для
відповідних старих bundled implementations.

```powershell
.\venv\Scripts\python.exe -I data/releases/k306_tau025_offline_v1/run.py --self-check
```

Це автономна перевірка синтетичної телеметрії, приймача і модельованого
транспорту; не hardware runner і не повтор фізичної серії.

## Історичний Крок 3: сенсорний hold і контрольований handover

Крок 3 (2026-10-23): пакет `data/edge/k306_tau025_hold_v1`, конфігурація
[mvp_runtime_v2.json](../configs/mvp_runtime_v2.json). Він явно вмикає
`sensor_mode="multi_rate_v1"` і `handover_mode="freeze_sync_v1"`.
Без цих опцій direct API зберігає legacy behavior для порівняння.

```powershell
.\venv\Scripts\python.exe data/edge/k306_tau025_hold_v1/system.py --config configs/mvp_runtime_v2.json --manifest-out resolved_hold_new.json
```

Це лише перевірка/складання і manifest, не фоновий цикл чи політ.
`SensorSnapshot` містить окремі `Measurement` для flow, ToF, range із capture
attitude та current attitude. Повторюйте ID/timestamp/value held вимірювання,
а не підставляйте час нового control tick. Invalid/stale не стають zero-fill.
У trace є `sensor_channels`, `altitude_integral`, `altitude_slew_anchor_pwm`.

Повна схема входів, результати 54 прогонів, обмеження й наступний етап:
[SENSOR_HANDOVER_UK.md](../docs/SENSOR_HANDOVER_UK.md).

Розділи нижче описують Крок 2 та попередні пакети. Після зміни repo-code їхні
pinned configs треба запускати **через system.py відповідного старого пакета**;
поточний builder навмисно відхиляє змішування коду різних версій.

## Спільне складання системи — Крок 2

Пакет `data/edge/k306_tau025_system_v1` містить `system.py` з `RunConfig`,
`build_system()` і журналом команд. Приклад конфігурації:
[configs/mvp_runtime_v1.json](../configs/mvp_runtime_v1.json): `both`,
`original_v1`, Levy вимкнений, STOP увімкнений. Потрібні Python >= 3.10 і NumPy.

```powershell
.\venv\Scripts\python.exe data/edge/k306_tau025_system_v1/system.py --config configs/mvp_runtime_v1.json --manifest-out resolved_run_new.json
```

Ця команда перевіряє пакет, складає систему та записує resolved manifest;
вона не запускає політ або фоновий цикл. Відносний `bundle_dir` рахується від
JSON-файла. SHA-256 метаданих обов'язковий; невідомі поля та непідтримуваний
transport відхиляються. Виконувані implementations також мають збігатися з
хешами пакета. Після подальших змін repo-code для старого пакета використовуйте
саме його `system.py`, а не нову реалізацію з репозиторію.

В автономному каталозі (копіювати **весь** пакет, config зберігати окремо):

```text
python /path/to/bundle/system.py --config /path/to/run.json --manifest-out /path/to/new_manifest.json
```

У перенесеному config потрібно вказати правильний `bundle_dir`; очікуваний
metadata SHA лишається тим самим. Старі пакети без `system.py` продовжують
підтримувати прямий `FlightRuntime`, але новий builder їх не підміняє поточним кодом.

Програмний інтерфейс у репозиторії:

```python
from edge.system import RunConfig, build_system

system = build_system(RunConfig.from_json("configs/mvp_runtime_v1.json"))
system.write_manifest("new_manifest.json")
# frame — чинний синхронний SensorFrame; caller забезпечує такти.
result = system.tick(frame, now_s=t, request="SHADOW", manual_pwm=manual_pwm)
record = result["trace"].to_dict()  # готовий запис для JSONL
```

`network_pwm → coordinated_pwm → supervised_pwm → issued_pwm → accepted_pwm`.
`applied_pwm` лишається alias до `supervised_pwm`. Відсутня команда — `None`,
не нейтральна заглушка. Без явного повідомлення приймача `accepted_pwm=None`
і `receiver_status="unobserved"`; `observe_receiver()` лише додає спостереження,
не викликає `CommandGate`, не передає команди й не надає повноважень.
Власний час приймача не порівнюється з годинником Pi. `session_id` тут —
кореляція журналу, не реалізація транспортних сесій чи захисту від replay.

Resolved manifest фіксує bundle hash, фактичні code paths/hashes, усі активні
контрольні параметри та поточний synchronous/SHADOW контракт. Його SHA у trace
дорівнює SHA байтів файла, записаного через `write_manifest()`; новий шлях
розгортання дає інший manifest, хоча ваги можуть бути тими самими.

Точне відтворюване порівняння старого і нового пакетів:

```powershell
.\venv\Scripts\python.exe -m edge.validate_structure --before data/edge/k306_tau025_bridge_v1 --after data/edge/k306_tau025_system_v1 --out structural_parity_new.json
```

Результати й межі перевірки: [SYSTEM_ASSEMBLY_UK.md](../docs/SYSTEM_ASSEMBLY_UK.md).

## Попередні серії

Оновлення 2026-10-22: `edge.validate_numpy` за замовчуванням виконує **код
вибраного пакета** в ізольованому subprocess. Поточний чотирирежимний replay:

```powershell
.\venv\Scripts\python.exe -m edge.validate_numpy --bundle data/edge/k306_tau025_bridge_v1 --execution bundled --out data/diagnostics/bundle_parity_new.json
```

Для окремої перевірки реалізації з робочого репозиторію використовуйте
`--execution repository`; її результат не підтверджує виконання коду старого
пакета. Старі команди/результати нижче — історія попередніх export-серій.
Нові межі й наступні зміни: [INTEGRATION_CONTRACTS_UK.md](../docs/INTEGRATION_CONTRACTS_UK.md).

Інтеграційний протокол MANUAL / SHADOW / AUTO / FAULT та receiver watchdog:
`edge/FLIGHT_BRIDGE_UK.md`. Новий пакет з обгорткою:
`data/edge/k306_tau025_bridge_v1`. Це програмна специфікація для стенда;
UART і фізичний ESP-арбітр ще потребують реалізації.

Оновлення 2026-10-21: додано opt-in пошукові yaw-імпульси та STOP/HOVER.
Новий пакет — `data/edge/k306_tau025_numpy_behaviors_v2`; початковий пакет
нижче не перезаписано, `model.npz` у двох пакетах бітово однаковий.
Детальний контракт, приклади й обмеження: `scripts/XY_BEHAVIORS_UK.md`.
Використання: `FlightRuntime(path, coordinator_mode='both', levy_explore=True,
stop_hover=True, seed=20261021)`, потім `step(..., stop_requested=True)`
для запиту гасіння швидкості. Без прапорців поведінка лишається попередньою.
Перша серія підтвердила зупинку у шести симуляціях, але не перевагу пошуку
за покриттям; у dense-сцені пошук зменшив мінімальний зазор до 0.101 м.

Готовий пакет: `data/edge/k306_tau025_numpy_v1`.
Джерело — точний `mvp_frozen_core_v1/trial_00003/after`,
`bridge_role_k306`, tau ×0,25. Ваги не навчалися й не квантизувалися.

## Що входить у пакет

- `model.npz`: ефективні масковані ваги float32, tau, A, зміщення,
  сенсорні gain/bias та маскований DN-декодер; 72 397 байтів (70,7 KiB).
- `metadata.json`: нейронний і контрольний контракти, джерело та його
  SHA-256, розміри/одиниці, пресети координатора, хеші файлів пакета.
- `metadata.sha256`: контрольна сума метаданих.
- `runtime.py`, `altitude_control.py`, `xy_coordinator.py`: автономний
  інференс і контролери на NumPy та стандартній бібліотеці.
- `benchmark.py`, `requirements.txt`: вимірювання продуктивності на цільовій машині.

Копіювати потрібно весь цей каталог, а не лише `.npz`.
`__pycache__` переносити не потрібно. PyTorch, SciPy, Isaac Gym і
решта репозиторію на Raspberry Pi не потрібні. SHA-256 виявляє зміну
файлів, але не є цифровим підписом від зловмисної підміни всього пакета.

## Експорт і перевірка на основному комп'ютері

```powershell
.\venv\Scripts\python.exe -m edge.export_numpy --out-dir data/edge/k306_tau025_numpy_v1
.\venv\Scripts\python.exe -m edge.validate_numpy --bundle data/edge/k306_tau025_numpy_v1 --out data/diagnostics/edge_numpy_v1/parity.json
.\venv\Scripts\python.exe data/edge/k306_tau025_numpy_v1/benchmark.py --bundle data/edge/k306_tau025_numpy_v1 --out data/diagnostics/edge_numpy_v1/benchmark_host.json
```

Ці файли вже створені. Для повторного експорту або вимірювання
вкажіть новий каталог/ім'я звіту: старі результати не перезаписуються.
Експортер потребує PyTorch, перевіряє вихідний research checkpoint
штатним завантажувачем і відхиляє непідтримувані архітектури.

## Запуск на Pi

Скопіюйте готовий пакет на Raspberry Pi. У його каталозі:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python benchmark.py --bundle . --out benchmark_pi.json
```

Цей запуск лише вимірює обчислення на синтетичних сенсорних входах:
мережа окремо та повний крок з XY-координатором і AltitudeHold.
Він не керує двигунами. Результат `benchmark_pi.json` потрібен для
висновку щодо швидкодії саме Pi Zero 2W / Pi 4B. Поточний звіт
`benchmark_host.json` отриманий на Windows AMD64.

Benchmark задає один BLAS-потік до імпорту NumPy. Для власного
польотного процесу також задайте `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` перед імпортом NumPy.

## Контракт входів і виходів

Викликайте крок один раз на 0,02 с. Runtime не містить планувальника,
обміну з ESP32, арбітра ручного керування чи команд армінгу.

Мережа приймає:

- `flow_xy`: два нормалізовані значення потоку за поточним контрактом
  проєкту; це не сирі відліки PMW3901 і не швидкість у м/с;
- `tof_8x8`: масив (8,8) або (64,) із дальностями, поділеними на 3,5 м,
  у межах [0;1], у тому самому порядку рядків/стовпців, що й навчання;
- `memory_ring`: необов'язкові 8 значень [0;1], за замовчуванням 1.
  У цьому чекпойнті ці входи від'єднані маскою;
- `AltitudeSample`: реальні дальність нижнього сенсора в метрах,
  roll/pitch у радіанах у прийнятій FLU-системі, timestamp і valid.
  Timestamp та now_s мають використовувати одну монотонну шкалу часу.

Приклад адаптера до ваших сенсорних даних:

```python
from runtime import FlightRuntime, AltitudeSample

controller = FlightRuntime(".", coordinator_mode="baseline")

def tick(flow_xy, tof_8x8, range_m, roll_rad, pitch_rad,
         measurement_time_s, valid, now_s):
    sample = AltitudeSample(range_m, roll_rad, pitch_rad,
                            measurement_time_s, valid)
    return controller.step(flow_xy, tof_8x8, sample, now_s=now_s)
```

Результат містить:

- `network_pwm`: сирий вихід мережі `[throttle, roll, pitch, yaw]`;
- `applied_pwm`: вихід після опціонального XY-координатора і AltitudeHold;
- `coordinator`: прапорець, причини втручання і внутрішні діагностичні значення.

**Throttle мережі — заповнювач, не команда утримання висоти.**
Для застосування всіх чотирьох каналів використовуйте FlightRuntime,
а не голий `NumpyPolicy.step_np`.

`reset()` скидає нейронний стан, контролери та зафіксований fault.
Не викликайте reset кожен кадр. При некоректній телеметрії runtime
припиняє видавати команду винятком і запам'ятовує fault. Скидання
після перевірки стану та передача повноважень належать зовнішній системі.

## Координатор

Типовий режим — `baseline`, тобто координатор вимкнений. Доступні
`anti_rollback`, `turn_latch`, `both`. Пресети:

- `original_v1`: вікно [0,02;0,06], захоплення знака від 20 PWM;
- `near_zero_v1`: вікно [-0,025;0,005], захоплення від 60 PWM.

```python
controller = FlightRuntime(".", coordinator_mode="both",
                            coordinator_preset="near_zero_v1")
```

Одноразова перевірка нового пресету в арені (40 с, той самий seed)
завершилася без зіткнення, але мінімальний запас став 15,03 см
проти 15,85 см у original_v1, а максимальний безперервний відкат —
44,44 см проти 28,23 см. Отже, новий пресет не оголошено кращим
і не встановлено за замовчуванням. Підбір параметрів на кутах
у цьому етапі завершено; графік і trace — `data/diagnostics/xy_near_zero_v1`.

## Чисельна перевірка

Для справжнього k306 виконано 2000 послідовних кроків (40 с модельного
часу): вузькі/широкі загрози з обох боків, різні дальності, потік,
чисте поле та нульовий вхід при відновленні. Порівняно сирі команди,
прихований стан і повний pipeline у baseline та обох пресетах both.

- Максимальна різниця PWM: **0,0001220703125**.
- Максимальна різниця прихованого стану: **2,980232238769531e-7**.
- Розбіжності причин втручання координатора: **0**.
- Допуск: PWM atol=0,02, rtol=0; стан atol=2e-5, rtol=2e-4.

Це еквівалентність на однакових вхідних послідовностях. Вона не
обіцяє побітово однакових довгих замкнених траєкторій на різних
процесорах. Перевірки також охоплюють reset, пошкодження файлів,
невірний контракт, invalid input і зафіксовану відмову висоти.
Автономність пакета перевірена в окремому процесі поза репозиторієм
із забороненими імпортами torch/scipy/simulation/core/configs.

## Виміряні ресурси — Windows AMD64, не Raspberry Pi

3000 кроків після 200 кроків прогріву, один BLAS-потік:

| Вимірювання | Медіана, мс | p95, мс | Максимум, мс |
|---|---:|---:|---:|
| Мережа | 0,0560 | 0,0789 | 0,3642 |
| Мережа + both original_v1 + AltitudeHold | 0,1205 | 0,2247 | 0,6468 |

Жоден виміряний крок не перевищив 20 мс. Це час обчислення,
без сенсорного I/O, передачі команд і затримок планувальника.

- Параметри NumPy: **475 520 байтів**, 464,4 KiB.
- Нейронний стан і коефіцієнт інтегратора: **2448 байтів**.
- RSS процесу: **42 053 632 байти**, приблизно 40,1 MiB.
- Піковий RSS: **42 278 912 байтів**.

RSS включає Python, NumPy, бібліотеки та кеші алокатора. Окремі
показники tracemalloc у JSON не є повним споживанням процесу;
їх вимірювали поза часовим заміром. PyTorch і SciPy у процесі
benchmark не імпортовані. Цифри не екстраполюються на Pi.

Повний набір тестів на цьому етапі: **666 passed**.
Початковий статус допуску моделі збережений у метаданих;
успішний експорт не є допуском до реального польоту.

## Підтримка коду

`edge/export_numpy.py` і `edge/validate_numpy.py` — інструменти хоста,
їм потрібен PyTorch. `edge/runtime.py` і `edge/benchmark.py` — NumPy.
У репозиторії адаптери `edge/altitude_control.py` та
`edge/xy_coordinator.py` посилаються на канонічні реалізації simulation.
При експорті в пакет копіюються **самі канонічні файли**, а не адаптери:
отриманий пакет не імпортує simulation і не потребує репозиторію.
# Наскрізна перевірка приймача (Крок 4)

Нові `edge/session_receiver.py` і `edge/transport.py` з'єднують прийнятий
FlightSystem з CommandGate у модельованому каналі. Повний фізичний запуск:

```powershell
.\venv\Scripts\python.exe -m evaluate.validate_transport_loop --config configs/mvp_runtime_v2.json --out-dir data/diagnostics/transport_loop_repeat
```

Каталог результатів має бути новим. Серія `transport_loop_v3`: 14/14 сценаріїв,
140 с симуляції, без зіткнень; повний suite 793 passed. Докладний контракт,
затримки, held altitude і повторний AUTO: [TRANSPORT_RECEIVER_UK.md](../docs/TRANSPORT_RECEIVER_UK.md).
Пакет `k306_tau025_hold_v1` не змінений; транспорт/приймач поки є окремими
модулями репозиторію, не прошивкою і не драйвером реального Wi-Fi/FC.
