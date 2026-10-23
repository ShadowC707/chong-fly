# Інтеграційні контракти: рішення та порядок реалізації

**Актуальне структурне завершення після Кроку 4:**
[MODULE_INTEGRATION_UK.md](MODULE_INTEGRATION_UK.md). Конфігурація v3,
спільні бібліотечні функції, автономний offline release та parity з Кроком 4
описані там. Нижче збережено початкову постановку і послідовність її виконання.

Дата: 2026-10-22. Продовження [аудиту](CODEBASE_AUDIT_UK.md). Це специфікація наступних змін; позначені нижче майбутні інтерфейси ще не реалізовані. Поточна серія реалізує isolated bundle replay і обмежує default pytest discovery. Вона не змінює ваги, рівняння, sensor timing, SHADOW integral або controller thresholds.

## Оцінка уточнень Gemini

Три основні зауваження прийняті: різночастотність сенсорів, стан altitude controller без authority і незалежність exported runtime від репозиторію.

Важливі уточнення:

1. VL53L5CX у режимі 8×8 підтримує максимум 15 Гц; це підтверджує необхідність різночастотного контракту. Джерело: [ST UM2884, розділ 4.4](https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf).
2. Для VL53L0X **high accuracy — 200 мс timing budget**, а не 30–33 мс. 33 мс наведено для інших профілів. Бюджет вимірювання також не тотожний гарантованій частоті доставки кадрів через firmware/transport. Джерело: [ST VL53L0X datasheet, таблиця ranging profiles](https://www.st.com/resource/en/datasheet/vl53l0x.pdf).
3. Sample-and-hold не робить канал свіжим: його вік продовжує зростати. Не можна просто збільшити max_age або видалити duplicate timestamp check; треба розрізнити новий control tick, повтор тієї самої measurement і replay транспортного кадру.
4. Зміна SHADOW integral або частотної семантики — поведінкова зміна, навіть при незмінних вагах. Вона не включається приховано до «рефакторингу без зміни чисел».
5. Самодостатність пакета обов'язкова; плоский каталог — поточний зручний формат, а не математична необхідність. У цій серії формат bundle не змінюється.
6. issued PWM на Pi ще не означає accepted PWM на ESP. Actuator tracking має явно позначати, який із цих сигналів відомий; без receiver acknowledgment не можна заявляти, що Pi знає фактично прийняту команду.

## Власники п'яти відповідальностей

| Відповідальність | Майбутній інтерфейс | Власник стану | Заборонена прихована дія |
|---|---|---|---|
| Складання системи | `build_system(run_config)` | Створює об'єкти, не зберігає control history | Навчання, fallback на іншу модель/версію |
| Конфігурація | Версійний RunConfig + resolved manifest | Незмінний після старту | Override dt/routing/weights поза контрактом пакета |
| Сенсорний зріз | SensorSnapshot → PreparedInputs | Adapter володіє last measurements; фільтри — своїм станом | Новий timestamp для старого вимірювання, invalid→clear |
| Крок керування | Shared control pipeline | Policy hidden state, XY state, altitude estimator/controller | Доступ controller до simulation ground truth |
| Виконання/діагностика | Tick runner → dispatch → receiver | Bridge authority, receiver lease окремо | Самостійне відновлення AUTO після lease expiry |

`FlightRuntime` — основа shared pipeline; `SensorFrame` — основа snapshot. Створюємо нові модулі лише для виділених відповідальностей. Compatibility adapters та старі immutable bundles зберігаються.

## Час і різночастотність: майбутній контракт

Окремо зберігати control tick time, measurement timestamp/identity/validity кожного каналу та транспортні sequence/session/arrival times. Receiver watchdog використовує власний monotonic clock. Вимірювання висоти має мати узгоджений з ним attitude: новий кут нахилу разом зі старим downrange не стає новим висотним вимірюванням автоматично.

Для altitude estimator визначаємо три різні події:

- **Нове валідне вимірювання:** оновлення оцінки height/vertical speed за measurement delta; перевірки jump/range/tilt залишаються.
- **Утримане валідне вимірювання:** ідентичні identity/timestamp/payload, новий control tick, вік у межах канального бюджету; estimator не обчислює derivative повторно, значення утримуються.
- **Неприпустимий кадр:** регресія часу, змінений payload під тією самою measurement identity, invalid flag, прострочення або replay транспортної sequence; явна відмова.

Оновлення controller integrator/slew визначається control dt і authority, не числом повторних надходжень одного сенсорного значення. До реалізації треба зафіксувати конкретний режим rangefinder та перевірити придатність його затримки; 200 мс не можна автоматично оголосити допустимими для поточного altitude loop. Сенсорний max_age і command lease TTL — різні бюджети.

Поточна strict synchronous семантика лишається чинною до окремої версії цього контракту. Потрібні тести hold→new sample, hold→stale, changed payload/same timestamp, nonmonotonic time, кілька sensor rates і незмінний результат на старому synchronous replay.

## SHADOW → AUTO і tracking: майбутній контракт

Поточна поведінка вже відома: controller у SHADOW оновлюється, а Bridge при вході AUTO спочатку видає manual snapshot і потім обмежує зміну команд.

Для наступного контрольованого експерименту пропонується:

1. У SHADOW прогрівати estimator; не накопичувати control integral без authority. Це нова явна семантика, не вже реалізоване виправлення.
2. На вході AUTO синхронізувати стан вихідного limiter із валідною manual-командою; не скидати механічно estimator history.
3. Визначити tracking зовнішніх обмежень: desired/supervised, issued і accepted мають різні значення. Пріоритет MANUAL та latch faults зберігається.
4. Парні сценарії: висоти 0.92/1.00/1.08 м, різниця manual throttle від supervised 0/±15 PWM, довге SHADOW, скасування AUTO, fault під час handover. Вимірювати integral, desired/issued throttle, висоту/швидкість, slew і час відновлення.

Вибір freeze/seed/back-calculation робимо за цим порівнянням. Відсутність стрибка першої RC-команди сама по собі не доводить відсутності подальшого фізичного перехідного процесу.

## Конфігурація та журнал: майбутній контракт

RunConfig задає source bundle + expected metadata hash, coordinator mode/preset/flags/seed, bridge bounds, environment/transport adapter mode. Нейронні solver, dt, encoding і action order беруться з immutable bundle. Resolved manifest містить фактично використані значення та hashes виконуваних модулів. Hardware transport не підміняється симуляцією при помилці конфігурації.

Стадії команди:

`network_pwm → coordinated_pwm → supervised_pwm → issued_pwm → accepted_pwm / None`.

Кожний запис має sequence/session, час, authority/behavior state, structured reason/fault origin, втручання за каналами і identity конфігурації. Проміжні стадії без результату — відсутні/None; їх не треба заповнювати вигаданими neutral PWM. Runtime `applied_pwm` тимчасово залишається compatibility назвою supervised stage. У новому replay output вона вже явно називається `supervised_pwm`.

## Виконано в першій серії

- `edge/bundle_validation.py`: parent preflight hashes, точна копія дозволених bundle files у тимчасовий каталог, запуск `python -I`, replay arrays, перевірки результатів та незмінності пакета.
- `edge/_bundle_replay_worker.py`: завантажує runtime/контролери з копії bundle, блокує Torch/SciPy/Pandas і repo dependencies, повертає hidden state, network/supervised PWM, XY reasons і paths/hashes виконаних модулів.
- `edge/validate_numpy.py`: default `--execution bundled`; старий repo-code comparison доступний як явно позначений `--execution repository`. Чотири режими, 2000 тактів кожний, та сама детермінована послідовність стимулів.
- `tests/test_bundle_execution.py`: numerical replay, виконання зміненої тестової bundle implementation, виявлення drift публічним валідатором, hash corruption і forbidden import.
- `pytest.ini`: default `testpaths = tests`. Це обмежує звичайний discovery; прямий ручний запуск `test_imports.py` все ще небезпечний. Сам legacy patcher не змінений і не видалений.

Ізольований subprocess — перевірка imports/dependencies для довіреного локального коду, **не sandbox для запуску довільного недовіреного Python**. SHA-256 не встановлює авторство пакета. Тимчасові execution paths у звіті після завершення вже видалені; відтворення робиться за source bundle та hashes.

Повний чотирирежимний валідатор потребує bundle з behavior-capable FlightRuntime API (поточний `k306_tau025_bridge_v1`). Старий пакет з іншим API може бути відхилений; він не підміняється поточним repo-runtime для отримання штучного PASS.

## Результати першої серії

Точне джерело: `data/selection/mvp_frozen_core_v1/trial_00003/after`.
Пакет: `data/edge/k306_tau025_bridge_v1`. Обидві перевірки: чотири режими
(baseline, both/original, both/near-zero, both із Levy+STOP), по 2000 тактів.

| Виконуваний код | Максимальна похибка PWM | Максимальна похибка hidden state | Розбіжності XY reasons |
|---|---:|---:|---:|
| Repository | 0.0001220703125 | 2.980232238769531e-7 | 0 |
| Isolated bundle | 0.0001220703125 | 2.980232238769531e-7 | 0 |

Звіти: [repository.json](../data/diagnostics/bundle_execution_v1/repository.json),
[bundled.json](../data/diagnostics/bundle_execution_v1/bundled.json).
У bundled report записані module paths/hashes, версії Python/NumPy та відсутність
заборонених imports. Робочий каталог child був поза репозиторієм.

Ваги незмінні: SHA-256 `model.npz` —
`3d9823351d745ad9e061aa87c3006682fff89999b9c6e13f7243c267a53cf8b1`.

TDD: спочатку 4 нових replay тести впали через відсутній модуль; після реалізації
вони пройшли. Додано ще 2 parametrized cases для public validator/default/drift.
Цільовий suite: **21 passed**. Повний default pytest discovery, вже з `pytest.ini`:
**731 passed in 67.20 s** (725 попередніх + 6 нових cases).

У sandbox був WinError 5 для системного temporary directory. Для перевірок
`tempfile.tempdir` задавався зовнішнім дозволеним scratch-каталогом Codex;
production code і користувацькі environment settings для цього не змінювалися.
Єдиний змінений Python-файл із попереднього 165-файлового audit snapshot —
`edge/validate_numpy.py`; replay runner/worker/tests — нові файли. Neural/runtime/
XY/altitude/bridge implementations та старі export bundles не змінювалися.

## Порядок продовження

Оновлення 2026-10-23: Крок 3 реалізовано як явні opt-in режими
`multi_rate_v1` / `freeze_sync_v1`, config `mvp_runtime_v2.json` і окремий
пакет `k306_tau025_hold_v1`. Повний опис контракту, 27 telemetry + 27 physical
прогонів і межі висновків — [SENSOR_HANDOVER_UK.md](SENSOR_HANDOVER_UK.md).
Розділи вище з історичним формулюванням «майбутній контракт» збережені як
постановка задачі; актуальна реалізація описана у новому документі.
Крок 4 реалізовано окремим receiver/transport runner без зміни прийнятого ядра.
Сесії, watchdog, фізичний fallback, повторний AUTO та похибка held altitude:
[TRANSPORT_RECEIVER_UK.md](TRANSPORT_RECEIVER_UK.md). Авторитетна серія —
`transport_loop_v3`: 14 сценаріїв, 140 с симуляції. Це ще не апаратний транспорт.

Оновлення Кроку 2: реалізовано RunConfig/build_system/resolved manifest та
п'ятистадійний журнал із legacy alias. Новий пакет `k306_tau025_system_v1`
має точний byte parity зі старим на 4 × 2000 тактів. Деталі, артефакти й
межі: [SYSTEM_ASSEMBLY_UK.md](SYSTEM_ASSEMBLY_UK.md). Розділи вище з позначкою
«майбутній контракт» для sensor hold і SHADOW залишаються планом Кроку 3.

1. Використати isolated replay як зафіксований чисельний baseline.
2. Виділити shared composition/config/trace з compatibility re-exports, без зміни filter/PI/clock semantics; порівняти baseline.
3. Окремими змінами з парними тестами реалізувати sensor hold і handover tracking.
4. З'єднати runner з receiver gate і transport model; далі фізичні сценарії з done/collision/fallback метриками.

Точна isolated parity не закриває кроки 2–4 і не є перевіркою польоту.
