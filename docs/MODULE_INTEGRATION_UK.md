# Актуальні межі модулів і точки запуску

Очищення після цього етапу: [що прибрано та як відновити](REPOSITORY_CLEANUP_UK.md).

Дата: 2026-10-23. Цей документ описує завершення **конкретних прогалин
інтеграції після Кроку 4**, а не новий повний аудит усіх дослідницьких алгоритмів.
Результати: [module_boundaries_v1/report.json](../data/diagnostics/module_boundaries_v1/report.json).

## Що виправлено

### 1. Контракт конфігурації

`FlightBridge` тепер отримує `target_height_m` з runtime.altitude.config і
`tof_range_m` з runtime.metadata. Resolved manifest показує ці самі значення.
Тест із ціллю 1.2 м перевіряє як manifest, так і фактичний допуск AUTO; раніше
Bridge продовжував перевіряти близькість до 1.0 м. Це виправлення для зміненої
конфігурації; поточна модель, ціль 1 м і шкала ToF 3.5 м залишилися ті самі.
NumPy loader і далі відхиляє непідтримувані нейронні encoding/range contracts:
довільна зміна ToF scale не стала автоматично дозволеною.

Спільні `CONTROL_DT_S`, `MAX_LEASE_S` і `SLEW_PWM_PER_S` живуть у модулі
протоколу `edge/flight_bridge.py`. BridgeConfig уже не дозволяє lease 50 мс,
яку CommandGate з межею 40 мс завжди відхилив би. Маніфест використовує ті самі
slew limits. Непридатні target/range перевіряються до control tick.

### 2. Бібліотечні функції більше не залежать від CLI експериментів

| Спільна відповідальність | Новий власник | Старе ім'я, яке лишилося сумісним |
|---|---|---|
| SHA-256, JSON, staging/publication, Windows rename retries | `artifacts/io.py` (stdlib only) | `generator.source_contract`, `generator.prepare_flywire_candidates` |
| Exact checkpoint loader з перевіркою графа, ваг і frozen constraints | `artifacts/research.py` | `optimizer.benchmark_candidate.load_research_run` |
| Перевірка маски та знаків recurrent W | `core/invariants.py` | `optimizer.probe_yaw_recovery.verify_recurrent_contract` |
| Симуляційний розклад 50/30/15 Гц | `simulation/sensor_schedule.py` | `evaluate.validate_sensor_handover.SampleSchedule` |

Це перенесення реалізацій зі збереженням арифметики. Активні callers цих
функцій переведено на нових власників; старі імена — re-export тих самих
об'єктів, не другі реалізації. Loader більше не імпортує benchmark/probe/search
CLI. Залежність `loader → training constraints → invariant` не повертається
до loader через probe. Host loader потребує Torch і host graph code; він не
потрапляє до автономного inference-пакета. I/O helper окремо перевірено з
забороненими імпортами NumPy/Torch/Pandas/SciPy/generator/optimizer.

У звітах, які фіксували код переміщених loader/invariant функцій за старими
шляхами, додано шляхи їхніх нових власників. Старі звіти не переписувалися.
Windows retry тести тепер підміняють rename/sleep у `artifacts.io`, де ця
логіка реально знаходиться; самі сценарії помилок збережено.

### 3. Різні типи запуску явно розмежовано

| Команда / модуль | Що виконується | Що результат не підтверджує |
|---|---|---|
| `optimizer.benchmark_candidate`, `evaluate.watch_mvp` | PyTorch research policy → optional XY → altitude → standalone physics | Bridge, доставка, receiver acceptance відсутні |
| `edge.system --config ... --manifest-out ...` | Збирання pinned FlightSystem і перевірка конфігурації/хешів | Немає control loop чи фізики |
| `evaluate.validate_transport_loop --config configs/mvp_runtime_v3.json --out-dir <new>` | FlightSystem → link model → SessionReceiver/CommandGate → reference fallback → physics | Реального Wi-Fi/ESP/FC немає |
| `edge.validate_numpy --execution bundled ...` | Точний код bundle в ізольованому subprocess проти PyTorch | Не фізичний і не апаратний тест |
| `edge.release --config ... --out-dir <new>` | Публікація переносимого offline-пакета після успішного self-check | Не hardware deployment |
| `<release>/run.py --self-check` | NumPy core → simulated link → receiver, expiry/reset/new session | Синтетична clear telemetry, без фізики та фізичного fallback |

Research viewer/benchmark додатково записують `execution_path` та
`receiver_included=False` у нові звіти. Старий `evaluate.validate_flight_bridge`
залишається історичним діагностичним скриптом; поточна перевірка інтеграції —
`validate_transport_loop`. `watch_champion.py` — застарілий entrypoint; для
точного чекпойнта використовуйте `watch_mvp`. Немає потреби вдавати, що всі
дослідницькі тести мають запускати transport: вони відповідають на різні питання.

### 4. Переносимий offline release

Новий bundle: `data/edge/k306_tau025_integrated_v1`.
Metadata SHA-256:
`5032c78eac785fbecda1948da160b6548f0716e35ea9e49234f8a522f8cc010c`.
Поточна repository configuration: `configs/mvp_runtime_v3.json`.

Повний offline release: `data/releases/k306_tau025_offline_v1`:

```text
run.py                 # explicit --self-check, no hardware I/O
session_receiver.py    # imports the bundled CommandGate
transport.py           # deterministic simulated datagrams
run_config.json        # relative path to bundle, pinned metadata SHA-256
release.json           # hashes of the shipped files
release.sha256
README.txt
bundle/                # exact immutable NumPy bundle and requirements.txt
```

`edge.release` перевіряє checksum джерела, копії та запускає скопійований
release через `python -I` до публікації каталогу. Наявний output не
перезаписується. Пошкодження release-файла відхиляється до його імпорту.
Хеші встановлюють цілісність, не авторство пакета; self-check виконує
довірений локальний Python, це не sandbox для невідомого коду.

Перевірка з каталогу поза репозиторієм пройшла: 40 прийнятих команд, дві
сесії, watchdog після drop, відхилення старої сесії без нового trip.
Імпорти Torch/Pandas/SciPy і каталогів репозиторію заборонені та відсутні.
Це перевірка ізоляції імпортів на поточному Windows Python/NumPy 2.5.2,
**не** установка на чистому Raspberry Pi/Linux. У package requirements
залишається підтримуваний діапазон NumPy; решта host dependencies для цього
запуску не потрібні. Встановлений робочий training environment не змінювали.

### 5. Безпечні legacy-входи та документація

`test_imports.py` більше не обходить усе дерево і не імпортує довільні
experiment/mutation scripts. Він має короткий явний список runtime modules
і запускається тільки через main. `scratch_apply_gpu.py` став інертним при
імпорті; прямий запуск відхиляється з поясненням, що історичний patcher
виведений з експлуатації. Його старий рецепт збережено в окремій функції для
історії, а не виконано над поточними файлами.

Кореневий README та edge README тепер починаються з актуальних config/release
і посилання на цю карту. Старі тексти/експерименти позначені історичними;
immutable bundles і попередні результати залишилися на місці.

## Чисельна перевірка

Початкові п'ять нових tests були RED: неправильна ціль висоти в manifest,
дозволений несумісний lease, відсутні нові бібліотечні модулі. Після змін
пройшли перевірки actual AUTO на 1.2 м, import boundaries і compatibility
identity. Додано offline-release test із запуском поза репозиторієм і
навмисним пошкодженням скопійованого receiver-файла, а також перевірки
інертних legacy imports.

- Чотири isolated runtime режими × 2000 ticks: **бітовий збіг** network,
  supervised PWM, hidden state; coordinator reasons однакові.
- Повторні **14 × 10 с** фізичних сценаріїв Кроку 4: **точний чисельний збіг**
  усіх physics rows, traces, receiver events і link events з transport_loop_v3.
  У порівнянні trace виключено лише `manifest_sha256`: код змінений, його
  ідентичність правильно відрізняється. Всі командні стадії та час порівнювалися.
- Немає перенавчання. SHA-256 model.npz незмінний:
  `3d9823351d745ad9e061aa87c3006682fff89999b9c6e13f7243c267a53cf8b1`.
- Повний suite після основних змін: **801 passed in 69.57 s**.
- Після додавання явних execution-path labels у research reports:
  цільові viewer/navigation/boundary тести — **18 passed**.

Відтворення (нові каталоги, без overwrite):

```powershell
.\venv\Scripts\python.exe -m evaluate.validate_module_refactor --out-dir data/diagnostics/module_boundaries_repeat
.\venv\Scripts\python.exe -m edge.release --config configs/mvp_runtime_v3.json --out-dir data/releases/offline_repeat
.\venv\Scripts\python.exe -I data/releases/offline_repeat/run.py --self-check
```

Для isolated subprocess tests у sandbox був потрібний дозволений зовнішній
tempfile.tempdir. Production environment settings і користувацькі змінні не
змінено. Старий `edge.validate_structure` перевіряє також незмінність bytes
самого Bridge, тому для цієї серії застосовано окремий refactor validator:
Bridge дійсно змінився, хоча поточні чисельні результати залишилися тими самими.

## Що залишилося і чому

Ці п'ять конкретних прогалин активного MVP-шляху закрито в описаних межах.
Це не твердження, що всі залежності 165+ файлів дослідницького репозиторію
тепер ідеальні. Зокрема:

- Canonical XY/AltitudeHold досі лежать в `simulation/`, а `edge/` реекспортує
  їх. У release це точні самодостатні копії. Перенесення до `control/` —
  можливий окремий організаційний крок, не новий контролер і не вимога запуску.
- Деякі research helpers і тип ReducedModel ще спільні з дослідницькими
  CLI/graph code. Вони не входять у NumPy release; їх варто виділяти при
  розвитку відповідного research-напрямку, не переносити все заради папок.
- Історичні configurations не сумісні з довільно новим repo-кодом за hashes.
  Це навмисний захист. Їхній exact replay робиться кодом відповідного bundle;
  для current repository використовується v3.
- Остаточне видалення legacy, архівування великих результатів, host dependency
  locks та GitHub release assets — окрема підготовка репозиторію до публікації.
- Hardware acquisition, справжній scheduler, clock synchronization, wire
  encoding/handshake, Wi-Fi/FC drivers, firmware fallback і фізичне ручне
  перехоплення ще не реалізовані цими змінами. Offline release їх не підміняє.

Немає підстав додавати великий центральний manager або нові state machines.
Коли з'явиться конкретний апаратний адаптер, він має виконати вже визначені
контракти sensor snapshot, authority і receiver lease. Лише нова чітка
відповідальність є підставою для ще одного модуля.
