# ESP32 ↔ Pi: стендовий контракт CFHW v1

Дата: 2026-10-26. База `487421fa076790a60f4dc51ee1ba25abb2570ddb`.
Це перший host-прототип апаратної інтеграції, **не готова прошивка чи допуск до AUTO**.
Інвентар: [hardware/INVENTORY_UK.md](hardware/INVENTORY_UK.md).
Наступний UDP host-крок: [ESP32_UDP_HOST_UK.md](ESP32_UDP_HOST_UK.md).

## Межі реалізації

```text
fake sensors → SENSOR bytes → PiHardwareAdapter → pinned FlightSystem
             → COMMAND bytes → compiled C++ Receiver → host status
             → Python RECEIPT codec → observe_receiver → trace
```

`edge/hardware_wire.py` — Python codecs усіх трьох повідомлень.
`edge/pi_hardware_adapter.py` — вузька обгортка чинного `build_system`/`tick`.
`firmware/esp32/receiver_core.hpp` — незалежне від SDK C++ ядро приймача та
декодер COMMAND; пам'ять приймача стала, немає контейнера історії сесій.
`firmware/esp32/host/receiver_host.cpp` — host harness, його текстовий stdin/stdout
**не є** дротовим протоколом. SENSOR encoder і RECEIPT encoder на ESP ще не портовані:
у mock їх виконує Python. `edge/hardware_mock.py` — детермінований стенд без фізики.

Залишаються: ESP build/SDK entrypoint, реальний scheduler і watchdog task,
сенсорні драйвери/калібрування, оцінювач clock mapping, транспорт і pairing,
джерело manual, FC adapter, локальний fallback і незалежне фізичне відкликання.
Чинні shared core, config v3, модель і sealed artifacts не редагуються.

## Спільні байти

Little endian для всіх цілих і IEEE-754 binary64. Структури C/C++ через memcpy
не передаються. CRC32/ISO-HDLC: reflected polynomial `0xEDB88320`, init/xorout
`0xffffffff`; охоплює всі байти до CRC. Це виявлення пошкоджень, не автентифікація.
Пакет самодостатній; максимум 1024 байти, без фрагментації на рівні цього контракту.

| Offset | Поле | Тип |
|---:|---|---|
| 0 | magic `CFHW` | 4 bytes |
| 4 | version = 1 | u8 |
| 5 | type: SENSOR=1, COMMAND=2, RECEIPT=3 | u8 |
| 6 | повна довжина включно з CRC | u16 |
| 8 | session: boot_epoch, generation | 2 × u64, обидва ненульові |
| 24 | sequence сенсорного кадру | u32 |
| 28 | payload | fixed per type |
| length−4 | CRC32 | u32 |

Session у Python — 32 lowercase hex символи **дротових байтів** (не числовий UUID).
Приклад: epoch=1, generation=1 → `01000000000000000100000000000000`.
Невідома версія/тип, неправильна довжина, truncation, trailing bytes, CRC,
невідомі flags/enums, NaN/Inf, некоректні діапазони — помилка decode.
Вкладені довільні довжини й виділення пам'яті за довжиною з пакета відсутні.

## SENSOR: 733 байти

Payload починається з `created_us:u64`, `local_mode:u8`, `stop_requested:u8`,
`receiver_tripped:u8`, `selected_source:u8`, `manual_valid:u8`,
`manual_rc:4×f64`, `selected_rc:4×f64` (77 байтів).
Mode: MANUAL=0, SHADOW=1, AUTO=2. Source: none=0, manual=1, auto=2, fallback=3.
Булеві flags строго 0/1. Відсутні RC кодуються всіма нульовими байтами;
це **відсутність значення**, не фізична команда. Наявні RC у [1000,2000].

Далі чотири measurement, у порядку flow, tof, altitude, attitude:

```text
sample_id:u32 | capture_us:u64 | valid:u8 | quality:u8 | values:N×f64
```

| Канал | N | Значення на вході цього v1 |
|---|---:|---|
| flow | 2 | нормалізований потік у тих самих осях/знаках, що й модель |
| tof | 64 | row-major 8×8, нормалізована **відстань** [0,1], scale 3.5 м |
| altitude | 3 | downrange м, capture roll рад, capture pitch рад |
| attitude | 2 | окремі актуальні roll/pitch рад |

`capture_us <= created_us`. Усі числа скінченні навіть при valid=0.
valid=0 залишається invalid до SensorTracker; підміна ToF «вільним простором» відсутня.
quality 0..255 — лише діагностика producer, без спільного вигаданого порогу.
Held sample зберігає ID, capture time, validity, quality і values. Новий snapshot
не оновлює capture time. При новій сесії історія measurement IDs скидається.

Це normalized-профіль для host tests, не raw driver API. Власник майбутньої
raw→physical→normalized конверсії — сенсорний адаптер до формування SENSOR.
Поки жодної апаратної конверсії немає. Policy сама виконує proximity encoding;
SENSOR його не повторює. Власник уточнив нижній монтаж PMW3901; осі/масштаб
і компенсація обертання ще потребують перевірки.

## COMMAND: 69 байтів

Payload: `has_lease:u8 | ttl_us:u32 | rc:4×f64` (37 байтів).
has_lease=0 вимагає **весь payload нульовий**. has_lease=1:
`1 <= ttl_us <= 40000`, канали строго throttle, roll, pitch, yaw, [1000,2000].
Sequence lease один раз заданий у заголовку; Python adapter перевіряє збіг
CommandPacket.sequence і CommandLease.sequence перед encode.

Binary64 зберігає поточний issued_pwm без округлення: немає нової похибки RC
чи зміни trace/observe_receiver. Реальна FC-quantization і mapping ще не визначені.
При їх додаванні потрібні окремі правила округлення, тести меж і журнал значень
після конверсії. ARM/AUX не входять у цей пакет.

## RECEIPT: 130 байтів

Заголовок відсилається до **вхідної** команди. Payload:

```text
receiver_us:u64 | reason:u8 | tripped:u8 | mode:u8 | accepted_valid:u8
source:u8 | fc_stage:u8 | selected_session:16 bytes | selected_sequence:u32
accepted_rc:4×f64 | selected_rc:4×f64
```

accepted_rc — історична подія прийняття цієї команди. selected_rc — поточний
вибір приймача. Для source=auto обов'язкові окремі selected session/sequence;
в інших source ці поля нульові. При source=none selected_rc нульові й означають
відсутність вибраного виходу. Manual/fallback не створюються з нейтральних значень.
fc_stage: not_attempted=0, write_attempted=1, confirmed=2, unknown=3.
У mock завжди not_attempted. UART write і FC execution не прирівнюються.

Reason enum, індекси 0..15:
`awaiting_command`, `accepted`, `session_mismatch`, `sequence_rejected`,
`sender_revoked`, `malformed_lease`, `lease_rejected`, `watchdog_expired`,
`authority_revoked`, `invalid_receiver_clock`, `invalid_sensor_sequence`,
`malformed_packet`, `reset_required`, `reset_requires_manual`,
`session_exhausted`, `invalid_mode`.

Статус не є ACK для retransmit. Accepted не подовжує lease. Пізній receipt
не означає, що команда ще активна. Відхилений старий session не змінює поточний
стан/причину приймача; reason цієї події залишається session_mismatch.

## Локальні повноваження, час і життєвий цикл

Порядок: локальне відкликання → watchdog → оголошення sensor frame → receive → output.
C++ poll додатково викликається на вході receive/announce/local actions: новий пакет
не приховує вже пропущений deadline. На реальному ESP окремий watchdog task
ще потрібен незалежно від blocking I/O. Самі host polls цього не доводять.

Deadline = **ESP recorded sensor send time + ttl**, не час приходу відповіді.
Приймається лише відповідь на останній оголошений sequence. На межі 20 мс
новий кадр оголошується першим: попередню відповідь відхилено навіть з ttl=40 мс.
Duplicate/reorder поточної сесії, включно з no-lease, защіпають fault.
No-lease до першого lease — awaiting_command; після активного — sender_revoked.
AUTO→MANUAL/SHADOW відкликає команду й защіпає fault. Відновлення тільки MANUAL/reset.

Reset не є мережевим повідомленням v1. Лише локальний trusted caller у MANUAL
створює новий generation. Pi створює **новий** PiHardwareAdapter/FlightSystem
для цього session та явно отриманого ClockMapping; перший matching SENSOR —
підтвердження обміну. Пакет з іншою сесією не запускає автоматичний rebind/reset.
Перезапуск Pi вимагає локального MANUAL/reset, нової сесії та нового стану ядра.
Відмова reset у AUTO зберігає чинний стан; watchdog усе одно перевірено першим.

Firmware precondition: до роботи з командами платформа має стійко зарезервувати
новий монотонний boot_epoch у NVS; при помилці сховища/вичерпанні — відсутність
допуску AUTO. **NVS backend ще не реалізовано**; host явно задає epoch=1.
Generation:u64 не обертається, переповнення → session_exhausted. Sequence:u32
не обертається: після UINT32_MAX потрібен локальний MANUAL/reset. Те саме правило
для sample_id producer. ESP clock:u64 — microseconds monotonic без регресії;
додавання deadline перевіряється на overflow. Lifecycle clock при reset не скидається.
Новий boot змінює epoch, тому старі session bytes не належать новому приймачу.

Це локальний bench handshake. Віддалене pairing, автентифікація відправника,
керування довірою до session/clock mapping і recovery на реальному транспорті
залишаються незакритими; v1 не відкриває сокет і не приймає мережевих reset.

ClockMapping у Python містить ESP/Pi anchor, похибку, drift bound і строк дії.
Timestamp консервативно зсувається до найстарішого допустимого часу, щоб не
занижувати age. Немає автоматичної синхронізації: у mock точне відображення
задане автором сценарію. Зміна mapping усередині сесії не підтримана, оскільки
вона змінювала б timestamp held sample. Прострочений mapping піднімає помилку;
caller припиняє issue, а receiver deadline не подовжується.

## Черги й часовий бюджет

Mock має одну активну Pi job, inbox capacity=1 та до 16 link events.
Переповнення відкидає **новий** елемент і рахується окремо; старим кадрам не
присвоюється новий sequence. Це явна bounded policy нового адаптера, а не зміна
історичної FIFO-моделі `evaluate.validate_transport_loop.PiComputeQueue`.
Сталий producer overload не лікується збільшенням черг, TTL або пропуском перевірок.

`hardware_mock` працює на синтетичній сітці 1 мс; service time — заданий параметр.
Це не властивість firmware scheduler. Host wall time не записується як Pi performance.
Відомі 14/14 при service=0 та 11/14 при доданих 2 мс у старому фізичному
evaluator залишаються [незакритим обмеженням](PI_COMPUTE_AUDIT_UK.md).
Новий mock не повторює ті 14 сцен і не закриває цей аудит.

До hardware AUTO потрібні реальні p50/p95/p99/max від capture через Pi й receiver,
окремий час FC output, deadline misses, ages/clock error, queue/loss/CPU/load.
Pi 4B та Zero 2W оцінюються окремо. Шлях receipt і FC output також потребує вимірювань.

## Відтворення й перші результати

Windows host із MSVC x64 + Windows SDK; Python із NumPy, pytest та залежностями
поточного suite. Компільований host core не потребує ESP SDK.

```powershell
& firmware/esp32/build_host.ps1 -OutDir .test-tmp-esp32-build-repeat
& venv/Scripts/python.exe -m pytest tests/test_hardware_wire.py tests/test_esp32_host.py tests/test_hardware_mock.py -q --basetemp .test-tmp-esp32-tests-repeat
& venv/Scripts/python.exe -m edge.hardware_mock --receiver-exe .test-tmp-esp32-build-repeat/receiver_host.exe --out data/diagnostics/esp32_host_repeat/normal.json
& venv/Scripts/python.exe -m edge.hardware_mock --receiver-exe .test-tmp-esp32-build-repeat/receiver_host.exe --out data/diagnostics/esp32_host_repeat/boundary.json --uplink-us 12000 --service-us 4000 --downlink-us 4000
```

Звіти створюються виключно в новому файлі, без overwrite.
`tests/fixtures/esp32_wire_v1.json` містить зафіксовані golden bytes; Python
перевіряє всі три типи, C++ — COMMAND, включно з fractional RC. Негативні cases:
truncation/CRC/version/length/flags/NaN/range/ttl, duplicate/no-lease/reorder,
old session/reset/reboot, uint32/uint64 overflow, watchdog без пакетів,
clock regression, authority revoke, held/invalid та відсутність manual input.

TDD: спочатку відсутні Python modules і C++ build script дали RED; перший GREEN
35 тестів. Другий RED виявив змішування event reason із current reason, занадто
сувору реакцію на заборонений reset і відсутню selected identity у receipt.
Виправлення ізольовано в новому адаптері/порті; Python reference не редагувався.

Локальні traces: `data/diagnostics/esp32_host_v1/` (ігноруються Git).

| Синтетичний цикл | Accepted | p50/p95/p99/max RTT | Deadline misses | FC attempts |
|---|---:|---|---:|---:|
| 4+2+4 мс | 50 | усі 10 мс | 0 | 0 |
| 12+4+4 мс | 0 | усі 20 мс | 80 | 0 |

Перший випадок завершується watchdog_expired після припинення обміну.
Другий отримує sequence_rejected ще до AUTO. Це коректні host-відмови,
не показник часу Pi або придатності Wi-Fi.

Фінальна перевірка: **857 passed in 85.77 s**, із них 50 нових host cases.
MSVC компілює core з `/W4 /WX`. Повний suite виконувався з process-local
`tempfile.tempdir` у `.test-tmp-esp32-runtime-v1`, оскільки початковий цільовий
запуск мав 117 PASS і одну помилку доступу до системного temp при isolated
bundle replay. Core і production environment для цього не змінювалися.
Немає прошивки/запуску на ESP32, реального вимірювання Pi, UART-команд до FC
або польотної перевірки.
