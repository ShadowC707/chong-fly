# Крок 3: різночастотні сенсори та SHADOW → AUTO

Дата: 2026-10-23. Ваги k306, XYConfig `original_v1`, PI gains та пороги допуску
AUTO не переналаштовувалися. Реалізовано нові явні режими; legacy API залишається.

## Зміни в коді

- `edge/sensors.py`: `Measurement`, `SensorSnapshot`, `SensorBudgets`,
  `SensorTracker`. Окремі identity/timestamp/validity для flow, ToF, altitude
  та поточного attitude. Стани `new`/`held` визначаються за історією, `invalid`
  з причиною фіксується при відхиленні. Payload копіюється; перевірка всіх
  каналів завершується перед комітом нового snapshot.
- `simulation/altitude_control.py`: opt-in `allow_hold`; derivative і velocity
  filter оновлюються тільки на новому вимірюванні, за measurement delta.
  На held ticks height/velocity зберігаються, а PI інтеграція та slew працюють
  за control time. `integrate=False` заморожує інтеграл. `synchronize_output`
  змінює лише anchor вихідного обмежувача, не історію estimator.
- `edge/runtime.py`: передавання цих явних режимів контролеру. Neural math,
  solver, encoding та dt не змінено.
- `edge/flight_bridge.py`: опційний multi-rate tracker перед викликом мережі;
  `freeze_sync_v1` заморожує інтеграл без AUTO і синхронізує внутрішній
  throttle limiter з ручною командою на вході в AUTO. Перша issued команда
  як і раніше дорівнює manual snapshot. Наступні обмежені двома узгодженими
  slew limiters. MANUAL priority та latched FAULT з явним reset збережено.
- `edge/system.py`: нові поля RunConfig, resolved manifest v2; журнал
  sensor states/ages/IDs, інтеграла і slew anchor поряд із п'ятьма PWM stages.
- Export/replay: новий `sensors.py` входить у перевірений автономний пакет;
  виконувані хеші цього модуля теж включені в isolated execution evidence.
- `evaluate/validate_sensor_handover.py`: зафіксований протокол, telemetry grid,
  окрема фізична перевірка, JSONL traces і графіки. Тести —
  `tests/test_sensor_hold.py`, `tests/test_handover_hold.py`.

## Контракт вимірювань

`SensorSnapshot.sequence/timestamp_s` описують створений транспортний snapshot,
а не свіжість усіх його каналів. Для кожного каналу `Measurement` містить
`sample_id`, `timestamp_s`, `value`, `valid`. Часи вже мають бути приведені
адаптером до monotonic clock Pi; тут не реалізована синхронізація годинників.

| Канал | Payload | Max age у цьому експерименті |
|---|---|---:|
| flow | 2 нормалізовані значення | 0.04 с |
| tof | 64 / 8×8 значень у [0,1] | 0.10 с |
| altitude | `(range_m, capture_roll_rad, capture_pitch_rad)` | 0.06 с |
| attitude | `(current_roll_rad, current_pitch_rad)` | 0.04 с |

Той самий ID дозволений тільки з тим самим timestamp і payload. Новий ID
вимагає більшого capture timestamp; регресія, future/stale, invalid flag,
неправильна форма чи змінений payload під старим ID відхиляються.
Свіжий транспортний sequence не омолоджує старе вимірювання.

Новий current attitude використовується для admission tilt check. Він **не
переписує** кути в held altitude payload: оцінка висоти й поточна формула
компенсації вертикальної тяги використовують capture-aligned кути range sample.
Отже, компенсація тяги теж має затримку в межах altitude age budget. Це не
екстраполяція руху або нова високочастотна оцінка висоти. Producer зобов'язаний
сформувати узгоджену range/attitude пару; validator не може виявити фізично
старе значення, якщо producer помилково видав йому новий ID і timestamp.

Budget дальності не можна підняти понад controller `max_age_s` через RunConfig.
200-мс режим тут не підтриманий. Бюджети — параметри цієї програмної перевірки,
не виміряна гарантія конкретного апаратного профілю.

Strict `SensorFrame` підтриманий: у multi-rate режимі він адаптується до всіх
нових синхронних каналів, але лише якщо його altitude timestamp узгоджений
із frame timestamp. Для реальних held values потрібен `SensorSnapshot`.

## Семантика переходу керування

1. SHADOW прогріває мережу та estimator, обчислює proposal, але не накопичує
   altitude integral. Після свіжого reset інтеграл нульовий; після AUTO→SHADOW
   він утримується на попередньому значенні, а не обов'язково обнуляється.
2. Tick входу AUTO ще обчислюється без інтегрування: повноважень до допуску
   не було. Після успішного допуску limiter anchor стає рівним manual throttle.
3. Supervised proposal цього tick лишається записаним як proposal; issued
   дорівнює manual. На наступному tick внутрішній limiter відштовхується від
   того самого значення, що й Bridge. За dt=0.02 межа становить 3 PWM.
4. В AUTO інтеграція дозволена з існуючим anti-windup. При скасуванні AUTO
   заморожується знову. Fault до входу не видає lease; reset потребує MANUAL.

Для нового mode manual throttle на вході також має лежати в допустимому
діапазоні самого AltitudeHold (1100–1400), інакше admission відхиляється.
Це запобігає seed внутрішнього limiter поза його робочими bounds.
Часові перевірки Bridge, dwell і пороги швидкості/висоти не розширені.

## Зафіксований експеримент

Висоти 0.92/1.00/1.08 м; PWM offsets 0/−15/+15; три варіанти:
legacy/sync, freeze_sync_v1/sync, freeze_sync_v1/multi.
Пошук Levy вимкнений, STOP запитується для гасіння горизонтального руху.

**Telemetry grid: 27 прогонів по 8 с.** Висота задається сталим вимірюванням,
5 с SHADOW, далі AUTO. Offset визначений точно від поточного supervised
throttle, передбаченого копією контролера. Ця копія не змінює основну систему.
Це перевірка команд і станів, а не фізичний політ.

| Показник | Legacy | Freeze/sync | Freeze/multi |
|---|---:|---:|---:|
| AUTO на 5.0 с | 9/9 | 9/9 | 9/9 |
| Максимум abs integral перед AUTO | 0.3984 | 0 | 0 |
| Перша issued мінус manual | 0 PWM | 0 PWM | 0 PWM |
| Наступний supervised мінус перша issued, max | 15.0144 PWM | 3 PWM | 3 PWM |
| Максимальна зміна послідовних issued | 3 PWM | 3 PWM | 3 PWM |
| FAULT | 0 | 0 | 0 |

Старий зовнішній Bridge вже стримував реальну issued-команду. Новий режим
узгоджує з нею внутрішній anchor і припиняє накопичення інтеграла без
повноважень. Показник 15.0144 PWM у таблиці стосується supervised proposal.

**Physical grid: 27 прогонів по 9 с.** До AUTO реально застосовується ручний
симуляційний altitude controller із target=initial height. З t=4.5 до 8 с
до його команди додається trim 0/±15; при відмові AUTO ця ручна команда
продовжує діяти. AUTO запитується у [5,8), з 8 с повернення MANUAL без trim.
Тут trim відноситься до ручного controller output, а не до supervised Pi;
це інший, фізично здійснений вплив, який не можна ототожнювати з telemetry grid.

В усіх трьох варіантах AUTO отримали **3/9** прогонів (усі з нульовим trim).
У шести ±15 випадках на кожен варіант AUTO лишилося недопущеним; усі криві там
описують ручний контур із trim. Це не провал утримання висоти автономною моделлю.
Не було FAULT або раннього завершення simulation у жодному з 27 прогонів.
Це не загальний тест уникнення зіткнень: сцена порожня, фізичний Gate відсутній.

Перерегулювання відносно 1 м **лише під час виданого AUTO** (до повернення MANUAL):

| Початкова висота | Legacy/sync | Freeze/sync | Freeze/multi |
|---|---:|---:|---:|
| 0.92 м: максимум вище 1 м | 7.23 см | 0.27 см | 0.36 см |
| 1.08 м: максимум нижче 1 м | 7.71 см | 1.71 см | 1.53 см |

При 1.00 м новий режим не покращив похибку: максимум відхилення вниз за AUTO
був 4.43 мм legacy, 9.57 мм freeze/sync, 8.12 мм freeze/multi. Відсутність
попередньо накопиченої поправки може прибрати й корисну компенсацію невеликого
зміщення. Не підбирали gains або integrator preload для приховування цього ефекту.

Multi-rate профіль: flow/attitude 50 Hz, ToF номінально 15 Hz, altitude 30 Hz.
Захоплення на першому control tick після сенсорного deadline, без backdating:
ToF має інтервали 60/80 мс, altitude — 20/40 мс. Це детермінований тест
різних частот без мережевих затримок, пакетних втрат чи апаратного timing noise.

## Валідація та артефакти

- Повний pytest: **771 passed in 67.38 s** (739 попередніх + 32 нових cases).
- Старий immutable `system_v1` проти нового hold-capable шляху на свіжих
  синхронних вимірюваннях: **8000 тактів, бітовий збіг** network/supervised PWM,
  hidden state та XY reasons. Інтеграція тут увімкнена: це перевірка сумісності,
  а не вимога однакової поведінки SHADOW між режимами.
- Isolated PyTorch parity: max PWM error 0.0001220703125, hidden 2.980232238769531e-7.
- Нові режими окремо виконані в flat bundle з `python -I`, без Torch/репозиторію,
  з held samples і реальним переходом Bridge в AUTO.
- Тести охоплюють new→held→new, stale/invalid/changed payload, regression,
  channel age budgets, float-roundoff boundary, fresh attitude при held range,
  AUTO cancellation, FAULT на вході AUTO до оновлення мережі та явний reset.

Новий пакет: `data/edge/k306_tau025_hold_v1`.
Metadata SHA-256: `685e67d78a34e1ca7a0ddc965c5c58f3bd2d01e5dadcc4150d0f00964bc71899`.
Незмінний model SHA-256: `3d9823351d745ad9e061aa87c3006682fff89999b9c6e13f7243c267a53cf8b1`.
Config: `configs/mvp_runtime_v2.json`. Попередні configs/bundles не переписані.

У `data/diagnostics/sensor_handover_v1/`: `protocol.json`, `report.json`,
54 JSONL traces, `synchronous_parity.json`, `torch_parity.json`,
`resolved_manifest.json`, `handover_commands.png`, `physical_heights.png`.

```powershell
.\venv\Scripts\python.exe -m edge.system --config configs/mvp_runtime_v2.json --manifest-out resolved_hold_new.json
.\venv\Scripts\python.exe -m evaluate.validate_sensor_handover --bundle data/edge/k306_tau025_hold_v1 --out-dir data/diagnostics/sensor_handover_repeat
```

Крок 4 залишається окремим: receiver `CommandGate`, модель транспорту,
затримки/втрати/сесії та фізичне повернення керування. Тут issued lease
безпосередньо застосовується в симуляції; accepted stage без receiver report
лишається None. Не перевірено deadline/performance на Pi та реальні сенсори.
