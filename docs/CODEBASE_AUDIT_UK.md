# Архітектурна ревізія Chong-Fly

Дата: **2026-10-22**. Стан локальної робочої копії, включно з незакоміченими файлами. Код, тести, параметри, ваги й історичні результати під час цієї ревізії не змінено. Додано лише документи аудиту.

Це знімок стану до покращень. Наступна реалізована серія та уточнені інтерфейси:
[INTEGRATION_CONTRACTS_UK.md](INTEGRATION_CONTRACTS_UK.md).
Статус конкретних прогалин після Кроків 1–4 і структурного завершення:
[MODULE_INTEGRATION_UK.md](MODULE_INTEGRATION_UK.md). Інвентаризація нижче
залишається датованим історичним знімком, а не списком нині відкритих багів.

## 1. Висновок для команди

Інтуїція про накопичені стики правильна. Проте проблема не в тому, що мережа, координатор, висотний контролер та арбітр існують окремо. Вони мають різні відповідальності. Найбільший борг — **немає одного явно описаного й наскрізно перевіреного шляху від сенсорного кадру до команди, яку прийняв одержувач**. Поруч існують кілька частково еквівалентних композицій та кілька значень слова `applied`.

Доцільний невеликий рефакторинг меж, а не нове ядро, новий датасет чи переписування проєкту. Зберігаємо k306, рівняння, learned projections, поточні пороги та окремі state machines. Спочатку уточнюємо виконуваний код пакета, формат сенсорного зрізу, час і provenance команд; потім перевіряємо повну композицію з receiver gate. Прибирання файлів — наступний, окремий крок.

Важлива межа: завершено багато дослідницької роботи над моделлю та створено переносимий NumPy runtime. Не завершено hardware acquisition, scheduler, transport, receiver firmware, фактичне ручне перехоплення і калібрування RC/сенсорів. Це різні етапи. Поточний audit не надає льотного допуску й не скасовує отриманих результатів симуляції.

## 2. Що перевірено та чого цей аудит не доводить

- Прочитано активні inference/control/bridge/export/rollout шляхи, важливі training/data interfaces, тестові контракти та конфігурацію залежностей.
- Через AST інвентаризовано **159 Python-файлів у 9 запитаних каталогах**, додатково 6 файлів у корені й `benchmarks/`: загалом **165**. Для кожного збережено призначення, статус, символи, рядки, імпорти й SHA-256. Імпорту всіх файлів поспіль не було.
- Виконано `venv/Scripts/python.exe -m pytest tests --collect-only -q -p no:cacheprovider`: **725 cases у 63 файлах, 12.22 с**. Це collection, не новий запуск усіх 725 тестів.
- Зіставлено код трьох export bundles із поточними canonical файлами й хеші їхніх ваг. Перевірено нульові memory-стовпці exported `W_in`.
- Виконано маленький in-memory probe переходу SHADOW→AUTO: він виміряв різницю між бажаним і виданим throttle; жодних апаратних команд чи змін ваг.
- Статичний граф включає lazy/conditional imports. Його цикли не означають, що Python зараз падає з circular import. Відсутність importer не означає, що CLI не використовується.
- Не виконувались завантаження CAVE, навчання, нова серія арени, аналіз реального радіоканалу чи перевірка Raspberry Pi. Зовнішні твердження Gemini не приймались за доказ поведінки коду.

Повні додатки:

- [CODEBASE_INVENTORY_UK.md](CODEBASE_INVENTORY_UK.md): призначення, класи/функції, статус і локальні залежності кожного Python-файла; для кожного test file — фактична кількість collected cases і назви тестів.
- [codebase_audit_inventory.json](codebase_audit_inventory.json): машинна карта з SHA-256, повними docstrings, методами, імпортами, статичними циклами та всіма 725 node IDs.
- [CODEBASE_TEST_MATRIX_UK.md](CODEBASE_TEST_MATRIX_UK.md): компактний список усіх 66 файлів `tests/`, кількості cases і сфери перевірки.

| Каталог | Python-файлів | Основна відповідальність |
|---|---:|---|
| core | 3 | Рівняння, topology/routing, contract |
| configs | 2 | Host-константи й compatibility export |
| generator | 19 | Джерела FlyWire, редукція, вчителі й дані |
| optimizer | 21 | Навчання, selection, checkpoints, probes, rollout |
| simulation | 18 | Physics/sensors, host policy, canonical controllers, scenes |
| evaluate | 10 | Перегляд, фіксовані experiments і validation |
| edge | 11 | NumPy inference, export, bridge, reference receiver |
| tests | 66 | 63 файли з cases, 2 helpers, 1 порожній файл |
| scripts | 9 | Обмежені дослідницькі probes/audits |

## 3. Як трактувати статуси інвентаризації

`ACTIVE_CORE` — використовується поточним model/control/edge шляхом або завантаженням його артефактів. Це не обов'язково файл, що має бути на Pi: наприклад, host exporter теж активний.

`ACTIVE_EVAL` — актуальний runner, інструмент валідації або зібраний pytest файл. Наявність у цій групі не означає, що всі його гілки вже запускались на реальній моделі.

`DORMANT_USEFUL` — не є активною поведінкою frozen MVP, але потрібний для наступних досліджень/відтворення. Це не твердження, що його можна без перевірки ввімкнути. Зокрема, memory обчислюється в rollout, хоча відключена в k306 routing.

`ONE_OFF_HISTORY` — рецепт конкретної завершеної серії, який варто архівувати разом із plan/config/result. Деякі такі файли вже містять спільні helpers: їх не можна видалити як незалежний скрипт.

`DEAD_LEGACY` — конкретна застарілість, порожня заглушка або непридатний entrypoint, обґрунтовані нижче. Це кандидат на наступне очищення, не вже виконане видалення.

Для всіх 165 файлів: **22 ACTIVE_CORE, 88 ACTIVE_EVAL, 35 DORMANT_USEFUL, 12 ONE_OFF_HISTORY, 8 DEAD_LEGACY**. Один статус на файл неминуче спрощує картину; змішані відповідальності явно зазначені в інвентарі.

## 4. Поточна модель й артефакти, які не треба губити

Робоче джерело: `data/selection/mvp_frozen_core_v1/trial_00003/after`, кандидат `bridge_role_k306`, tau scale 0.25. Індекс trial не є достатньою ідентичністю: орієнтуємося на metadata і checkpoint hash.

Три пакети мають однаковий SHA-256 `model.npz`:

`3d9823351d745ad9e061aa87c3006682fff89999b9c6e13f7243c267a53cf8b1`

| Bundle | Зіставлення з поточними джерелами |
|---|---|
| `k306_tau025_numpy_v1` | runtime.py і xy_coordinator.py відрізняються |
| `k306_tau025_numpy_behaviors_v2` | runtime.py і xy_coordinator.py відрізняються |
| `k306_tau025_bridge_v1` | runtime, XY, altitude, bridge збігаються |

Ці копії — snapshots виконуваного середовища, а не три моделі з різними навченими вагами. Старі пакети не переписувати «для чистоти»: це знищить зв'язок між результатом і кодом. Для GitHub достатньо обраного release-пакета, маленьких manifests/reports і інструкції відновлення архівів, якщо попередні версії надійно збережені окремо.

У k306 `W_in.shape == (306, 74)`, `count_nonzero(W_in[:, 66:74]) == 0`. Отже, memory channels не впливають на цю exported мережу. Існування кільцевого буфера не додає їй пам'яті про бічні перешкоди. Натомість рекурентний hidden state мережі та state координатора реально діють.

Рівняння — graph-constrained leaky-rate model, не публікаційна CfC. `BiologicalCfCCell` та деякі docstrings лишилися compatibility назвами; сам `core/models.py` це чесно пояснює. Перейменування класу зараз створить міграційний ризик без зміни математики.

## 5. Фактична карта виконання

### 5.1. Дослідницький шлях

```text
FlyWire/CAVE + pinned annotations
  → validated source bundle → role reduction/routing → candidate registry
  → teacher demonstrations → pretrain/search/continuation
  → checkpoint + graph + report + constraints
  → load_research_run → host PyTorch policy
```

`optimizer/benchmark_candidate.py:39` володіє loader, хоча за назвою це benchmark CLI. Для frozen checkpoint він додатково викликає `RecurrentConstraints`, бо `requires_grad` не міститься в state_dict.

### 5.2. Стандартний closed-loop benchmark

```text
standalone environment observation
  → memory.update
  → policy.step_np(dt=.02)
  → optional XYCoordinator.apply(dt=.02)
  → AltitudeHold.apply(sensor timestamp, now)
  → env.step(command) × 5, physics dt=.004
  → metrics / traces
```

Саме цей шлях використовує `watch_mvp.py`, XY starts та більшість фізичних оцінок. Він **не включає FlightBridge, transport або CommandGate**. У `optimizer/rollout.py` записаний `applied_pwm` означає команду перед `env.step`.

### 5.3. Поточний deployment reference

```text
невтілений sensor/transport adapter
  → SensorFrame(normalized flow, normalized ToF, altitude sample, sequence, timestamp)
  → FlightBridge: перевірки frame/authority
      → FlightRuntime
          → NumpyPolicy, fixed dt=.02
          → optional XYCoordinator, fixed dt=.02
          → AltitudeHold, timestamps
      → ready envelope / MANUAL-SHADOW-AUTO-FAULT
      → first command = manual snapshot; далі final slew
      → CommandLease
  → невтілений transport
  → CommandGate на приймачі: sequence/authorization/lease deadline
  → невтілені receiver fallback і передача RC до G4
```

`FlightRuntime` — контейнер policy/XY/altitude, а не четвертий послідовний контролер після них. `FlightBridge` обгортає runtime. `CommandGate` має бути на незалежному приймачі; перенести його всередину Pi runtime означало б втратити його сенс при зависанні Pi.

`edge/altitude_control.py` і `edge/xy_coordinator.py` — короткі re-export adapters. Canonical implementations зараз у `simulation/`; exporter копіює їхній текст до автономного пакета. Це вже дає спільну логіку, хоча назва `simulation` вводить в оману щодо її runtime ролі.

## 6. Підтверджені проблеми й обмеження стиків

Позначення: **факт** — видно в коді/вимірюванні; **ризик** — можливий наслідок, який ще не доведено в польоті. P1 — виправити до відповідного широкого використання, P2 — узгодити під час невеликого рефакторингу. Пріоритет не означає, що результат навчання недійсний.

### A. P1: валідація bundle не гарантує виконання саме його коду

**Факт.** `edge/validate_numpy.py` імпортує `FlightRuntime` з `edge.runtime` робочої копії. `load_bundle()` перевіряє hash файлів у заданому каталозі, але не підміняє вже імпортовані класи кодом із нього. `evaluate/validate_flight_bridge.py:16` аналогічно бере старий behaviors bundle для поточного repo-runtime.

Найновіший bridge bundle зараз збігається з джерелами — конкретного розходження там не виявлено. Проте два попередні пакети відрізняються: їхній hash може бути коректним, а виконуватиметься інша implementation. Це обмежує твердження «ми запустили саме цей пакет».

**Рішення:** окремий subprocess із заданим bundle cwd/import path, який виконує bundled runtime на записаних входах; host-reference порівнює повернені outputs. У report записувати identity і source hashes реально імпортованих модулів. Поточний repo-parity залишити як окрему перевірку поточного коду. Чистий import smoke без Torch потрібний, але не замінює behavioral parity пакета.

### B. P1: повна команда не охоплена одним фізичним тестовим шляхом

**Факт.** Основний rollout оминає Bridge/Gate. `validate_flight_bridge.py` відразу подає `decision['lease'].pwm` у physics; receiver gate та затримки відсутні. Також внутрішній цикл `obs,*_=env.step(applied)` не використовує done/info як повноцінний flight acceptance. Це короткий handover diagnostic, а не новий flight benchmark.

Тому висновок «XY starts пройдено» не дорівнює «та сама система з bridge slew та lease timing пройшла ті самі старти».

**Рішення:** додати до існуючого runner режим production composition з окремими Pi/receiver clocks, CommandGate, контрольованою доставкою та явною simulation fallback policy. Повторно використати сцени/метрики, не створювати новий фізичний рушій. Спершу короткі handover/fault випадки й один контрольний trajectory; потім зафіксована матриця стартів, коли шлях узгоджено.

### C. P1 для інтеграції: кілька несформульованих часових контрактів

| Шар | Час зараз |
|---|---|
| NumpyPolicy | Приймає лише dt=.02; fraction інтегратора фіксована |
| XY у FlightRuntime | Завжди .02, зокрема timers/dwell/search |
| AltitudeHold | Derivative за measurement timestamp; slew за now |
| FlightBridge | Допускає frame і arrival delta 10–30 мс; filter/slew за arrival delta |
| CommandGate | Окремий receiver clock; deadline від локального sensor send time, TTL ≤40 мс |
| Simulation | Рівно .02 control та .004 physics, без jitter транспорту |

**Факт:** дозволені нерівномірні виклики Bridge не змінюють dt мережі. Це не помилка самої дискретизації: для scheduler із fixed-tick семантикою так можна працювати. Але scheduler, resampling/hold і політика пропущеного кадру ще не визначені. Не можна трактувати допустимий arrival gap як гарантований фізичний крок моделі.

`SensorFrame` має один timestamp; Bridge вимагає збіг altitude timestamp у межах 1e-6 с. Для синхронного simulation snapshot це нормально. Для різночастотних сенсорів потрібні окремі measurement times, freshness та спосіб утворення зрізу. Просте присвоєння всім старим вимірюванням нового timestamp приховає застарілість.

`CommandGate` правильно не відновлює вже прострочений дозвіл свіжим пакетом без reset. Це не слід «спрощувати». За cadence 20 мс і TTL 40 мс час на доставку наступного дозволу залежить від попереднього deadline; окремо `latest_sensor_sequence` відхиляє застарілий reply. Зміна Wi-Fi/UART не усуває потребу в цьому бюджеті.

**Рішення:** explicit Clock/Timing contract: measurement time, control tick time, arrival time, receiver send/receive time; fixed policy dt; допустимий вік кожного каналу; явна поведінка при пропуску й reboot/session change. Не збільшувати TTL навмання, щоб замаскувати порушення.

### D. P2: два slew limiter не знають фінального результату один одного

**Факт:** `AltitudeHold.update()` (рядки 134–150) зберігає свій `_pwm`, integral та limited flag. `FlightBridge.tick()` (рядки 132–138) додатково змінює команду за manual-entry/final slew. Нижній controller не отримує фактично виданий throttle.

In-memory probe з clear sensors, height=1 м, manual throttle=1225:

| t, с | AltitudeHold desired, PWM | Bridge issued, PWM |
|---:|---:|---:|
| 0.50 | 1213 | 1225 |
| 0.52 | 1213 | 1222 |
| 0.54 | 1213 | 1219 |
| 0.56 | 1213 | 1216 |
| 0.58 | 1213 | 1213 |

Це очікуваний плавний перехід, **не доведена нестійкість**. Але фраза «контролер знає applied throttle» тут неправильна. У SHADOW він також інтегрує помилку, хоча не керує апаратом. Власний anti-windup не бачить зовнішніх обмежень/відсутності authority.

**Рішення:** журнал стадій і чітка політика tracking/initialization при AUTO. Спершу тест на ненульову height error та manual mismatch, потім рішення: зовнішній feedback до integrator чи контрольований seed/reset при handover. Не видаляти limiter й не змінювати PI поведінку без парного порівняння. Фіксований slew може також змінити момент скидання Yaw після network neutral — це вимірюється на issued/accepted командах, а не лише network PWM.

### E. P2: одиниці й конфігурація розкидані, але подвійного спотворення ToF не доведено

**Факт:** Bridge перевіряє normalized ToF, runtime/NumPy знову перевіряє його, XY перевіряє метри. Нормалізовані значення переводяться у метри для геометричних порогів; network перетворює їх у proximity. Це різні правильні представлення, не помилкове повторне нормування.

Ризик — різні джерела констант: `configs.flight_config`, export metadata та literal values. Bridge перевіряє висоту відносно **1.0**, ToF множить на **3.5**, має hardcoded slew `[150,2000,2000,10000]`. Runtime бере altitude target із metadata. При нинішніх параметрах збіг є; після легального нового altitude config він не гарантований. `BridgeConfig.lease_s` і незалежне Gate `<=.04` також потребують спільного protocol contract.

**Рішення:** typed sensor/config snapshot з явними units, target height із altitude contract, узгоджені timing/RC bounds. Зберегти перевірки на публічних boundaries: прибирати їх усі заради «один раз перевірити» небезпечно, коли модулі запускаються окремо.

### F. P2: стан і діагностика не мають спільної схеми

**Факт:** AltitudeHold/Runtime/Bridge мають fault latch; XY має behavioral state; Gate має tripped. Це не одна і та сама машина станів. Runtime ловить ValueError, Bridge — ширший набір типів. Нейронний/XY стан встигає просунутись до відмови altitude; після ValueError runtime потребує reset, тому видачі команди з цим станом немає. Повна транзакційність не обов'язкова, якщо цей контракт збережений.

Bridge повертає preview/lease/reason, але не повний runtime result з network PWM і XY reasons. Gate.accept повертає bool, тому різні причини відмови зливаються. Термін `applied_pwm` у runtime означає до Bridge, а в trajectory — те, що подали у фізику.

**Рішення:** єдина trace schema й структуровані FaultCode/AuthorityReason з origin/time. Залишити окремі machines, описати reset table: хто скидає neural state, controller integrator, search RNG, authority і sequence/session. MANUAL має працювати навіть при несправних сенсорах. Receiver не повинен залежати від успішного Pi reset.

### G. P2: низькорівневі helpers живуть у CLI експериментів

**Факт:** 23 файли інвентарю статично імпортують `generator.prepare_flywire_candidates` (не всі лише helpers). `publication/write_json` потрібні багатьом unrelated reports, проте цей модуль також імпортує pandas, source selection і filtered candidates.

Інші приклади:

- `load_research_run` знаходиться в benchmark CLI.
- `mvp_constraints` імпортує invariant check з `probe_yaw_recovery`, а цей experiment імпортує benchmark loader.
- `validate_xy_starts` бере summarize з `compare_xy_behaviors`; plotting/helpers змішані з конкретною серією.
- `core.models` потребує `ReducedModel` із `generator.graph_reducer`, який також містить algorithms, benchmarking та CLI.

Статично знайдені цикли (включають локальні імпорти, тому не є автоматично runtime помилками):

```text
benchmark_candidate → mvp_constraints → probe_yaw_recovery → benchmark_candidate
optuna_tuner ↔ search_pipeline
prepare_flywire_candidates ↔ validate_flywire_candidates
continuous_reflex_teacher ↔ generate_reflex_dataset
```

**Рішення:** витягнути лише стабільні contracts/artifact IO/loaders/metrics у нейтральні модулі. CLI використовують бібліотеку, бібліотека не імпортує CLI. Compatibility re-exports залишити на час міграції, щоб не ламати команди й checkpoints.

### H. P1 для майбутнього generic pytest/import tooling: кореневий import-everything

`test_imports.py` рекурсивно імпортує Python-файли на верхньому рівні. `scratch_apply_gpu.py` на верхньому рівні виконує regex patch і записує source files. Поєднання створює можливість зміни коду під час звичайного discovery/import. Саме тому цей аудит використовує AST та `pytest tests`, а не `pytest` у корені.

**Наступне очищення:** вивести mutation script із importable tree або прибрати після архівації; замінити import-all на explicit entrypoint smoke; задати testpaths. Зараз ці файли не запускалися й не змінені.

### I. P2: документація й відтворення installation відстають

README починається з попередніх етапів teacher v3/v4 і benchmark v8, тоді як rollout має v9, є frozen core, NumPy runtime та bridge. Саме це може повертати команду на вже завершені етапи. Потрібен короткий current-status/index зі вказанням точного package/checkpoint/config, а історичні записи — нижче.

`requirements.txt` містить `onnx`, хоча `edge/exporter/export_onnx.py` порожній і AST не показує імпортів onnx у перелічених Python-файлах. Основний environment змішує training, tests, reduction та export. У bundle окремий NumPy requirement — це правильна межа. Не міняти зараз робоче середовище; наступним кроком розділити extras/locks і записати перевірені версії для host/Pi.

Частина active tools/code untracked, а `*.pt` ігнорується. Один коміт джерел не дасть іншому колезі працездатний MVP автоматично. Потрібні модель/release asset із hashes та явний спосіб отримати його, не приховане припущення «data вже є локально».

## 7. Що у пропозиції Gemini правильне, а що варто змінити

| Теза | Оцінка |
|---|---|
| Потрібна інвентаризація перед очищенням | Так; зроблено, з консервативними статусами |
| Немає узгодженого sensor snapshot | Частково: SensorFrame вже існує; бракує units/quality/per-sensor time та спільних derived features |
| Два flow filters означають помилкову подвійну фільтрацію | Ні: це **паралельні** фільтри з різними задачами. Cascading raw→filter→filter тут немає |
| Об'єднати всі фільтри в одне значення | Не без нового поведінкового експерименту; tau=.06 керує XY, tau=.12 — admission |
| Багато FSM — треба один великий FSM | Ні: behavior, authority і receiver lease повинні лишитися різними відповідальностями |
| Додати до perception ознаки кута/проходу | Зараз зайве: це нова поведінкова модель, не refactoring із збереженням результату |
| Бандли дублюються — можна знести | Ні: snapshots з різним кодом при тих самих вагах потрібні provenance/replay |
| Isaac/memory/generator/optimizer не потрібні inference, отже мертві | Ні: training/research — окремий життєвий цикл; drone_env/drone_interface містять активну standalone логіку |
| Сотні тестів можуть бути зайвими | Кількість не доказ. Спочатку групування/coverage, потім усунення доведених повторів |

## 8. Мінімальний цільовий інтерфейс

Це **проектна пропозиція**, а не вже додані класи. Не треба починати з великого Perception framework.

### 8.1. SensorSnapshot і PreparedInputs

Розширити існуючий SensorFrame або ввести поруч один immutable зріз: sequence/session, control timestamp, measurement timestamps/validity, normalized flow із названою шкалою й координатами, ToF у метрах із визначеною політикою invalid/no-return, altitude sample. Адаптер симуляції та адаптер ESP мають створювати один контракт; position/velocity ground truth не потрапляють у controller.

PreparedInputs один раз створює потрібні представлення: exact float32 normalized ToF для мережі, metre ranges/min для координатора/admission, named left/right summaries лише там, де вони вже використовуються. Порядок float32 conversion і арифметики перевірити parity-тестом, щоб «краще оформлення» не змінило значення на threshold.

Підготовка може бути спільною, але state filters поки лишаються окремими: `flow_for_xy` і `flow_for_admission`. Уніфікація їхнього helper без зміни alpha/reset — допустима; підміна двох сигналів одним — новий експеримент. Altitude vertical-speed filter — третій сигнал іншої фізичної величини.

### 8.2. ControlProposal і DispatchResult

Назви стадій мають бути однозначними:

```text
network_pwm
  → coordinated_pwm
  → supervised_pwm        # altitude controller підставив throttle
  → issued_pwm            # Bridge authority + final limits
  → accepted_pwm / None   # незалежний receiver gate
```

У кожного tick: timestamps, behavior state, authority state, fault/reasons, saturation/limiter flags, changed channels, sequence та bundle/config identity. Для одного control tick достатньо одного структурованого trace record; serializable представлення окремо від mutable controller objects.

### 8.3. Межі модулів

Можливе майбутнє розміщення без великої перебудови:

```text
core/                  # нейронна математика/routing
control/               # NumPy-only sensor contracts, altitude, XY, pipeline contracts
artifacts/             # integrity/publication, graph/checkpoint loaders (host частина окремо)
simulation/            # physics/sensors/scenes/adapters, без ownership deployment
edge/                  # NumPy policy, bundle/export, bridge, transport adapters згодом
evaluate/              # shared runner/metrics/plots + thin experiment entrypoints
generator/, optimizer/ # дослідження/навчання, не dependency embedded bundle
```

Це орієнтир залежностей, а не вимога створити всі папки одразу. Перші корисні переміщення — `publication/write_json`, `load_research_run`, invariant checks та canonical controllers. Зберегти re-exports і old CLI до завершення контрольованої міграції. Старі immutable bundles залишаються автономними snapshots.

## 9. Наявні механізми, які варто використати

- **control_contract, core.contracts, routing metadata:** уже є основу для опису ownership/units/dynamics; розширити її, не вигадувати незалежний четвертий manifest.
- **source_contract + publication:** перевірки integrity й атомарна публікація вже реалізовані; перенести загальні helpers, не писати новий набір hash/save функцій для кожного experiment.
- **NumpyPolicy і canonical NumPy XY/altitude:** deployment основа вже є. Відділити розміщення від симуляційної назви, а не переписати математично.
- **rollout traces + PhysicsTelemetryTracker + contact_and_clearance:** повторно використати в production-composition runner; не множити незалежні трактування clearance/STOP.
- **CommandGate timing tests:** хороша reference-spec для незалежного receiver; перенести semantics у firmware з tests, а не замінити Pi heartbeat у тому самому процесі.
- **role reduction/source quality/search/pretrain:** потрібні наступному поколінню моделей. Замороження MVP не робить дослідницьку частину непотрібною.
- **Memory / exploration teacher / zonal ToF:** зберегти як відкладені гіпотези. Поточна пам'ять із decay до clear не є картою достовірної вільної області; її ввімкнення потребує окремого sensor/routing/training контракту.
- **avionics_filter:** реально керує simulation motors. Це внутрішній симуляційний контролер, його не треба переносити на ESP як заміну G4 і не можна видалити зі standalone фізики.

## 10. Ревізія тестів

Детальна таблиця на кожен файл: [CODEBASE_TEST_MATRIX_UK.md](CODEBASE_TEST_MATRIX_UK.md). Усі поточні 725 cases лишаються в suite; групування нижче — пропозиція організації, не вже виставлені pytest markers.

| Група | Cases | Призначення |
|---|---:|---|
| CORE_CONTRACT | 199 | Math/routing, coordinates, controller ownership, edge parity, authority/lease, XY priorities |
| DATA_TRAINING_EVAL | 429 | Source integrity, teacher/dataset, training/search/checkpoints, geometry, diagnostics/reporting |
| OPTIONAL_RESEARCH | 97 | Memory, scan teacher/probe, OU/Levy variations, alternate encoding/zonal surfaces |

В `tests/` 66 Python-файлів: 63 мають cases, `conftest.py` і `synthetic_graphs.py` — потрібні helpers, `test_cfc_dynamics.py` — порожній. Отже, 0 collected для helper не означає dead code.

Статичне порівняння AST тіл test-функцій **не знайшло буквально однакових тіл**. Це не доказ відсутності логічного дублювання. Наприклад, front ToF geometry і downward raycast повторюють cylinder/box сюжети, але перевіряють різні реалізації/напрямки. Altitude, runtime, bridge fault tests також працюють на різних boundaries. Видаляти їх лише за схожою назвою не варто.

Що додати під час наступних змін, цілеспрямовано:

1. Exact-bundle execution parity в ізольованому subprocess, включно з довгим stateful replay.
2. Повний шлях snapshot→runtime→bridge→transport model→gate→physics, зі stale/drop/reorder/authority reset.
3. Jitter/freshness/per-sensor timestamp contract без прихованого переписування timestamps.
4. Handover з ненульовим height error і external throttle limiting; звірити desired/issued/accepted trace.
5. Valid non-default altitude target/config: Bridge не повинен використовувати іншу ціль.
6. Explicit rejected input/reset semantics, причина відмови й нездатність stale command поновити authority.

Це закриває прогалини композиції; не потрібно нових десятків довільних сцен або повторного навчання для організаційної ревізії.

## 11. База для майбутнього очищення

### Впевнені кандидати після окремого узгодженого cleanup

- Порожні `edge/exporter/{codegen_c,export_onnx,quantizer}.py`, `optimizer/cma_es_tuner.py`, `tests/test_cfc_dynamics.py`.
- Застарілий `evaluate/watch_champion.py`: імпортує `DroneSimulationEnv` з `optimizer.evaluate`, де такого символу немає; потім задуманий rebuild/pretrain замість exact checkpoint. Заміна вже є — `watch_mvp.py`.
- `benchmarks/hardware_benchmark.py`: proxy старої архітектури з випадковими матрицями; не використовувати його числа для Pi або поточного rate core. Заміна для реальної моделі — `edge/benchmark.py`.
- `test_imports.py` та import-time patcher: прибрати небезпечний discovery/mutation шлях, зберігши історію patcher поза import tree за потреби.

### Перенести/ізолювати, а не оголосити сміттям

- Конкретні compare/probe experiments — під recipes/experiments після extraction shared summarize/plot/IO helpers. Plan, seed, hashes та results лишаються пов'язаними.
- Isaac UI/HUD/backend — optional namespace після відділення standalone geometry/physics/sensor interfaces. Не видаляти `drone_env.py` або `drone_interface.py` цілком.
- Старі immutable bundles/diagnostics — архів із manifest; обрати latest release, а не редагувати всі копії.
- `codebase_dump.txt`, `dump_simulation.txt` — похідні ручні dumps, кандидати на локальний архів/ignore після перевірки, що у них немає єдиної копії потрібних даних.

### Зберегти

FlyWire provenance/root IDs/quality decisions, role reduction, current checkpoint/graph/report, training recipe і constraints, exact-export contract, regressions math/coordinates/routing/safety, як successful так і failed experiments. Не підміняти відтворюваність одним великим «чистим» source dump.

## 12. Рекомендований порядок наступної роботи з умовами завершення

**Крок 1 — зафіксувати поточну поведінку й виконуваний артефакт.** Додати exact-bundle replay та baseline traces; розділити repo-code parity і bundled-code parity. Умова завершення: відомо, які bytes виконувались, які weights/config використано, і replay узгоджений у заданих numerical tolerances.

**Крок 2 — впорядкувати спільні contracts/IO без зміни чисел.** Витягнути helpers із CLI; додати named sensor/time/command-stage schema й конфігураційні посилання замість 1.0/3.5 literals. Зберегти окремі фільтри, порядок операцій і public re-exports. Умова: поточні contracts/tests та baseline traces не змінились, нові non-default config перевірки проходять.

**Крок 3 — один production-composition runner.** Runtime, Bridge, receiver Gate та контрольований transport проходять той самий env/metrics шлях. Визначити reset/clock/session/fallback contracts, виміряти external slew/handover. Умова: результат містить кожну стадію command і достовірно зупиняє автономну authority при порушенні контракту; геометричні висновки отримані вже для повної композиції.

**Крок 4 — cleanup і навігація команди.** Порожні/застарілі файли, optional Isaac, experiment recipes, dependency extras, current README, release artifact location, testpaths і кілька змістовних комітів. Умова: колега може відновити середовище, отримати exact bundle і запустити задокументований smoke без випадкового training/mutation.

Для цих кроків не потрібен новий ресерч конектоми або зміна learned weights. Потрібне узгодження інженерних контрактів. Транспорт і апаратне калібрування додаються після цього, а не маскуються ще однією обгорткою.
