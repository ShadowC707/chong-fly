# Крок 2: структурна уніфікація — 2026-10-22

Мета — зробити складання, фактичні параметри і походження команд явними,
зберігши поточну чисельну поведінку. Ваги не навчалися; XY/Altitude/Bridge
математика, часові перевірки, reset та правила надання AUTO не змінювалися.

## Що реалізовано

1. `edge/system.py`: immutable `RunConfig`, читання JSON, `build_system`,
   `FlightSystem.tick`, `CommandTrace`, додавання явного receiver observation.
   Builder збирає існуючі `FlightRuntime → FlightBridge`. Він не створює
   паралельний контрольний контур і не переносить receiver watchdog на Pi.
2. Config задає pinned bundle hash, mode/preset, feature flags, seed,
   BridgeConfig, trace session та єдиний підтримуваний transport `offline`.
   Непідтримуваний hardware mode відхиляється без тихого fallback.
   Параметри нейромоделі та висотного контролера беруться з пакета.
3. Builder перевіряє фактично завантажені `system/runtime/bridge/XY/altitude`
   файли проти manifest пакета. У repository canonical XY/Altitude реалізації
   залишаються в `simulation/`; в автономному пакеті це їх точні копії.
4. `FlightRuntime.step` повертає network, coordinated, supervised stages;
   старий `applied_pwm` — той самий об'єкт, що `supervised_pwm`.
   Snapshot кожної завершеної стадії дозволяє журналювати часткову відмову.
   Перед MANUAL/fault tick журнал очищає попередній snapshot без зміни
   нейронного стану або стану контролерів.
5. `FlightSystem.tick` зберігає всі старі поля результату Bridge і додає trace:
   sequence/session, час, manifest identity, authority/reason/fault,
   стадії команди, XY reasons, behavior state. JSON-представлення також
   містить змінені канали між суміжними обчисленими стадіями.
6. `issued_pwm` береться лише з реально створеної lease. `accepted_pwm` не
   вгадується. `observe_receiver` перевіряє session/sequence/PWM відповідність,
   повертає новий immutable record та розрізняє rejection і unobserved.
   Це інтерфейс журналювання, не доставка, підтвердження від заліза або
   оцінка чинності lease за двома різними годинниками.
7. `edge/export_numpy.py` пакує новий модуль. Loader та isolated replay
   підтримують точні старі й новий набори файлів; довільні файли не допускаються.
8. `edge/validate_structure.py` порівнює старий прямий runtime з runtime,
   створеним новим builder, у **двох ізольованих процесах**. Додатковий
   `assembly=True` у replay перевіряє автономне виконання builder і його manifest.

Довільні фізичні runners не переводилися на інший механізм часу. Старі direct
API залишено як compatibility входи; нові інтеграційні callers мають починатися
з `build_system`. Історичний `evaluate/validate_flight_bridge.py` ще не є новим
наскрізним runner з транспортом і receiver: це предмет Кроку 4.

## Пакет і результати

- Source: `data/selection/mvp_frozen_core_v1/trial_00003/after`.
- Новий пакет: `data/edge/k306_tau025_system_v1`.
- Config: `configs/mvp_runtime_v1.json` — `both/original_v1`, STOP on, Levy off.
- Metadata SHA-256: `da88656f5925fbc4d017324951adf27280d150358f8fce58b5af33edef0bc3d4`.
- Незмінний `model.npz`: `3d9823351d745ad9e061aa87c3006682fff89999b9c6e13f7243c267a53cf8b1`.

| Перевірка | Результат |
|---|---|
| Старий `bridge_v1` → новий `system_v1`, 4 × 2000 тактів | Бітовий збіг network/supervised PWM та hidden state |
| XY intervention reasons | 0 розбіжностей |
| Ваги та altitude/XY/Bridge code bytes | Незмінні |
| Новий isolated NumPy → PyTorch, 4 × 2000 тактів | max PWM error 0.0001220703125; hidden 2.980232238769531e-7 |
| Цільові runtime/assembly/bridge/replay/gate тести | 47 passed |
| Повний default pytest discovery | **739 passed in 54.95 s** |

Артефакти в `data/diagnostics/system_assembly_v1/`:

- `structural_parity.json`: per-case byte equality, shapes/dtypes/input hashes,
  обидві execution identities, resolved manifests виконаного builder.
- `torch_parity.json`: окрема tolerance parity з PyTorch; вона не означає
  бітової тотожності PyTorch і NumPy.
- `resolved_manifest.json`: фактичне repository складання затвердженого config.
- `command_trace.jsonl`: 100 синтетичних synchronous ticks із реальним k306.

Протокол trace: dt=0.02; flow=(0,0), ToF=1 по всіх 64 зонах, downrange=0.982 м,
roll/pitch=0; manual=(1213,1500,1500,1500). Ticks 0–29 SHADOW, 30–64 AUTO,
на 60 altitude.valid=False, 65 MANUAL і явний reset_fault, 66–99 SHADOW.
Отримано 30 issued leases, FAULT на 60, MANUAL на 65, SHADOW на 66.
У всіх рядках accepted=None: приймач у цьому прикладі не запускався.
Це перевірка складання/журналу, **не фізична симуляція** руху чи handover.

Нові тести спочатку впали через відсутній `edge.system`. Окремий тест
виявив різницю SHA manifest через Windows CRLF; `write_manifest` тепер явно
зберігає LF. Тест із відхиленням лише 0.001 PWM довів, що структурний валідатор
відхиляє навіть drift, який міг би пройти ширший PyTorch tolerance.

## Межі й наступний крок

Бітовий збіг встановлено на цій машині/NumPy/BLAS і цих входах. Це не гарантія
бітового збігу Windows і Raspberry Pi, не вимірювання затримки Pi та не допуск
до польоту. Архівні пакети не змінені. SHA захищає від непоміченої зміни файлів,
а не від переписування всього пакета атакувальником.

Крок 3 лишається поведінковою зміною: SensorSnapshot/new/held/invalid,
канальні age budgets, узгоджений downrange+attitude; потім парний
SHADOW→AUTO експеримент на висотах 0.92/1.00/1.08 м і manual delta 0/±15 PWM.
Зараз altitude integral усе ще оновлюється у SHADOW; synchronous перевірки
та bridge slew за arrival delta не змінені. Наскрізний Gate/transport/physics
runner та сесії на дроті — Крок 4.
