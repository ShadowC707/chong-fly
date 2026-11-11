# Основа ESP32: Pi з моделлю та ноутбук із геймпадом

Дата: 2026-10-27. Це збірна стендова основа C++/ESP-IDF, а не допуск до польоту.
Pi з моделлю і ноутбук із GPW70 — **два окремі наземні пристрої**. На борту —
сенсори, ESP32 та G4. [Узгоджена схема](hardware/GROUND_LINK_ARCHITECTURE_UK.md).
USB Host/паяння перемички не потрібні цьому ручному каналу.

Актуальна передача колезі й перевірка одним запуском без обладнання:
[HARDWARE_COLLEAGUE_UK.md](HARDWARE_COLLEAGUE_UK.md) (2026-10-29).

## Що реалізовано

| Частина | Реалізація | Межа поточної перевірки |
|---|---|---|
| ESP-IDF application | `firmware/esp32/app/src/main.cpp` | зібрано для ESP32-S3; ця app ще не прошита |
| Локальний стан/арбітр | `foundation/runtime.hpp` | host C++ tests; MANUAL/SHADOW/AUTO, пріоритет manual, fault/reset |
| AUTO receiver | наявний `receiver_core.hpp`, без зміни його семантики | попередні parity tests + нова app-інтеграція |
| Pi telemetry | `foundation/wire.hpp` | C++ SENSOR/RECEIPT читаються наявним Python codec/FlightSystem |
| Notebook protocol | `foundation/manual.hpp`, `edge/manual_wire.py` | CFMI v1 challenge/input, cross-language host tests |
| Сенсорний cache | `SampleSlot`, `SensorCache` | IDs, capture times, held, invalid; поточні кути не переписують capture angles |
| Фізичні сенсори | `PendingSensors` у `foundation/adapters.hpp` | усі дані invalid; SPI/I2C не ініціалізуються |
| UDP / Wi-Fi | два nonblocking sockets, STA або AP | target збірка, host loopback; реальна Wi-Fi робота не перевірена |
| Boot/session | NVS counter з commit/readback, локальний MANUAL reset | target збірка; збої NVS на апараті не перевірені |
| FC | `DisabledFc` | єдиний stage `not_attempted`; UART до G4 відсутній |

Реальні PMW3901/VL53L0X/matrix drivers та FC adapter не маскуються порожнім
«успіхом». Увімкнення `enable_sensor_hardware` або `enable_fc_output` повертає
помилку конфігурації, доки відповідний адаптер не реалізовано й не перевірено.
Стендові нулі/clear-space measurements у звичайній app не генеруються.

## Збірка та запуск тестів

Використовуємо вже встановлені ізольовані інструменти: `.venv-esp32-build`,
PlatformIO 6.2.0, `espressif32@6.12.0`, ESP-IDF 5.5.0, Xtensa 14.2.0.
Профіль N16R8-compatible: 16 MB Flash, DIO 40 MHz; PSRAM не використовується.

```powershell
& .venv-esp32-build/Scripts/python.exe firmware/esp32/pio_local.py run --project-dir firmware/esp32/app --jobs 2
& firmware/esp32/build_host.ps1 -Foundation -OutDir .test-tmp-foundation-host
& venv/Scripts/python.exe -m pytest tests/test_esp32_foundation.py tests/test_foundation_loop.py -q --basetemp .test-tmp-foundation-check
& venv/Scripts/python.exe -m edge.foundation_mock --foundation-exe .test-tmp-foundation-host/foundation_host.exe --out data/diagnostics/foundation_manual_run/normal.json
```

Збірка не завантажує firmware на плату. Наявна USB diagnostic firmware і
перевірена повна backup Flash залишаються окремими результатами. Не виконували
нові flash/eFuse/G4 операції. Main environment моделі й sealed artifacts незмінні.

## Конфігурація й точки уточнення

Default `AppConfig`: Wi-Fi disabled, sensors pending/invalid, FC disabled,
MANUAL, bench AUTO disabled. Запуск app із defaults не потребує відомого pinout.
Для майбутнього мережевого стенду скопіювати
`firmware/esp32/app/local_config.example.hpp` у сусідній `local_config.hpp`.
Цей файл Git-ignored. SSID/password не логуються; Wi-Fi settings зберігаються
в RAM. Мережа не обирається автоматично за першим отриманим пакетом.

Конфігурація містить:

- STA/AP role, SSID/password, дві fixed peer addresses/source ports;
- ESP local ports: Pi 47001, notebook 47002 (можна уточнити до мережевого стенду);
- explicit `trusted_link`, окремий `allow_bench_auto`;
- bench manual timeout, default 100 000 us; діапазон 1 000..100 000 us;
- PinPlan з `-1` для всіх невизначених контактів;
- вимкнені physical-sensor та FC features, які поки не можна ввімкнути.

Перевіряються конфлікти port/peer/pin, reserved pins, межі timeout, відсутні
credentials і непідтримані adapters. GPIO19/20 зарезервовані під native USB,
26..37 не віддаються периферії через memory profile, GPIO43/44 зайняті COM.
Це консервативний початковий PinPlan, не остаточна схема всіх датчиків.

Адреси й pairing мають бути встановлені до мережевого стенду. `trusted_link=true`
не створює автентифікацію: це явне твердження конфігурації про окремо
підготовлений trusted link. IP/port, CRC і session ID самі не автентифікують peer.
Не передаємо мережеві passwords у source, звіти або журнал.

## Розклад і обмеження пам'яті

| Задача | Priority | Період / робота |
|---|---|---|
| safety | 10 | 1 ms: poll watchdog, link/lock faults, sensor cache snapshot, DisabledFc |
| UDP | 5 | nonblocking; sensor/challenge кожні 20 ms, до 4 datagrams з кожного socket за ітерацію |
| sensors | 4 | 5 ms: poll backend, останній cache у static queue довжини 1 |
| console | 3 | bounded UART0 input, status кожні 2 s |

`CONFIG_FREERTOS_HZ=1000` — розклад foundation, не 4-ms grid симулятора.
Sensor backend має бути asynchronous/nonblocking; жодне очікування conversion
не повинно переноситися в safety task. Задача сенсорів не виконує мережевий I/O.

Runtime захищений коротким mutex. Socket I/O, encode SENSOR/RECEIPT та logging
виконуються **поза ним**. При очікуванні mutex понад 2 ms реєструється fault,
який відкликає повноваження при наступному доступі. Це не доведена worst-case
межа RTOS або захист від зависання самого ESP; окреме відкликання на FC потрібне.

Commands не мають app retry queue і не coalesce за принципом latest-only.
Кожний прочитаний datagram проходить gate. OS receive buffers обмежені 4096 bytes
на socket; розмір datagram перевіряється через додатковий 1025-й byte.
Foreign/oversized/untrusted packets споживають scan budget. Вичерпання budget,
socket/send errors, пропущені frame slots і lock timeouts мають лічильники.
Усі kernel/radio drops цими лічильниками не спостерігаються.

Latest-only queue використовується лише для сенсорного cache. Held measurement
зберігає sample_id/capture_us/values; помилка нового вимірювання робить канал
invalid, а не вигадує нове значення. Capture angles altitude — частина того
самого вимірювання, незалежна від поточного attitude channel.

## Пакети й manual channel

Pi CFHW v1 незмінний: SENSOR 733, COMMAND 69, RECEIPT 130 bytes.
Новий CFMI v1 має окремий magic і не декодується як AUTO COMMAND.
Усі числа little-endian, RC binary64 у порядку throttle/roll/pitch/yaw,
CRC32/ISO-HDLC. Сирі C structs у мережу не копіюються.

Спільний CFMI header — 28 bytes:

```text
magic "CFMI":4 | version=1:u8 | type:u8 | total_length:u16
session:16 | sequence:u32
```

Session — boot epoch u64 + local generation u64, обидва ненульові.

| Повідомлення | Напрямок | Body | Total із CRC32 |
|---|---|---|---|
| CHALLENGE type=1 | ESP → notebook | issued_esp_us:u64 | 40 bytes |
| INPUT type=2 | notebook → ESP | valid:u8, takeover:u8, stop_requested:u8, RC:4×f64 | 67 bytes |

Один challenge відповідає одному sensor frame, з тією самою session/sequence.
Notebook відповідає на поточний challenge один раз. Відповідь приймається до
20 ms від його початку на ESP; old session, duplicate, reorder або запізнілий
challenge не продовжують deadline. Invalid INPUT має valid=0 та всі RC bytes=0;
це відсутність ручного вводу, не PWM 1500.

Прийнятий manual sample діє до `challenge_start + manual_timeout_us`.
Це timeout утримання ручного значення для bench, окремий від AUTO lease.
Watchdog перевіряється **до** receive. Після expiry/malformed/явного invalid
INPUT manual fault защіпається; свіжий пакет його не приховує. Потрібні local
MANUAL/reset та нова session. Ноутбуковий clock у gate не порівнюється з ESP.

`takeover=true` відкликає AUTO й переводить ESP у MANUAL. Надсилання звичайних
manual samples без takeover підтримує ручне джерело під час bench AUTO.
`stop_requested` передається Pi як наявний STOP/HOVER request; це **не ARM,
не motor kill і не негайний DISARM**. Жодних button/axis mappings не призначено.
Read-only raw input adapter/stub уже доданий:
[GPW70_CALIBRATION_UK.md](hardware/GPW70_CALIBRATION_UK.md).
Calibrated GPW70/XInput sender, deadman та RF freshness ще потрібні.
XInput packet_number не використовуємо як годинник радіопакетів.

## AUTO, session і локальна авторизація

UART0 через COM 115200 приймає локальні однобайтові дії:
`M` MANUAL, `S` SHADOW, `A` bench AUTO, `R` reset у MANUAL.
Мережевого AUTO grant, ARM або reset немає.

`A` потребує explicit bench config, trusted link, свіжого manual та відсутності
защіпнутих fault. Ручний ввід пріоритетний, але Pi не може надати собі AUTO.
При manual fault або link loss AUTO відкликається; network reconnect не
відновлює повноваження автоматично. Reset створює нову generation та потребує
також нового стану FlightSystem на Pi, а не rebinding на довільний UDP packet.

Boot epoch записується в NVS namespace `chong_hw` із commit/readback **до**
першої session. При помилці storage або overflow session не створюється;
NVS не стирається автоматично. Це не crypto identity/anti-rollback: після
erase/відновлення старої Flash потрібне нове встановлення довіри/link setup.

AUTO gate reference зберігає latest sequence і TTL до 40 ms від ESP frame start.
Додано окремий app guard: якщо frame publisher затримався, відповідь для того
самого frame у 20 ms чи пізніше все одно відхиляється (`sequence_rejected`),
app fault защіпається. Попередня активна lease не подовжується. Local MANUAL
дозволяє свіжий ручний ввід; повернення AUTO потребує reset. Запізнілий пакет
іншої session не створює цього fault. Reference Receiver і model code не змінені.
Це консервативний захист від порушення розкладу, а не послаблення контракту.

## Що уточнюємо далі

| Залежність | Де підключається / що потрібно |
|---|---|
| PMW3901 вниз | SensorBackend: breakout, SPI pins, signs/scale, derotation, quality |
| VL53L0X вниз | I2C pins/address, неблокувальний ranging, validity/range, монтаж |
| Forward matrix | точна модель, 8×8 mode, address/bus, normalize/invalid zones |
| Attitude G4 | підтверджений telemetry protocol, capture alignment, latency bound |
| Clock mapping Pi↔ESP | реальний estimator з uncertainty/drift/expiry; mock offset непридатний |
| GPW70 у ноутбуці | raw XInput polling/mock готові; фактичний wired capture, calibration/mapping і controller-off test ще потрібні |
| Wi-Fi | topology, fixed addresses, trusted pairing, два peer й реальні затримки |
| FC control/revocation | узгоджений protocol, T/R/P/Y→AETR, AUX/ARM, перевірена реакція при відмові ESP |

Драйвер має передавати Measurement з capture time та власною identity, а не
часом відправлення мережевого пакета. Placeholder contacts та невідомі angles
не замінюються нулями з valid=true. Raw PWM не інтерпретується як кут нахилу
без налаштованого FC mode/rates/calibration.

## Докази й часовий бюджет

TDD: спочатку тести падали на відсутніх manual codec/app target та mock loop.
Окремий тест знайшов повторну публікацію challenge з тим самим sequence;
виправлено. Тест із затриманим publisher показав приймання на межі 20 ms;
після окремого app guard він відхиляється. Перевірено old-session isolation,
invalid/malformed bytes, expiry-before-receive, local reset, configuration
errors, held sensors/capture angles, manual priority, AUTO expiry та link loss.

Наскрізний mock: реальні host UDP sockets, fake normalized sensors/manual,
справжній pinned FlightSystem, C++ app core. 40 ticks (30 SHADOW + 10 AUTO),
10 прийнятих issued leases; manual takeover і disconnect окремим запуском;
0 спроб FC output. Control time і clock mapping у цьому тесті **штучні**.
Host wall RTT записаний окремо й не є ESP/Pi/Wi-Fi latency qualification.

Попередній аудит [PI_COMPUTE_AUDIT_UK.md](PI_COMPUTE_AUDIT_UK.md) чинний:
14/14 при нульовому service time, 11/14 при доданих 2 ms на 4-ms grid.
Потрібні реальні acquisition ages, ESP↔Pi round trip p50/p95/p99/max,
deadline misses та навантаження від двох peer. TTL/age budgets, ваги й sealed
artifacts заради проходження цих перевірок не змінювалися.

Перевірка основи від 2026-10-27: **923 passed за 85.47 s**, з них 36 foundation cases.
Target build успішний: firmware.bin 791 568 bytes, static RAM 36 804 bytes,
Flash 16 MB / DIO / 40 MHz. Остаточна incremental збірка — 30.17 s.
Image checksum/hash перевірені esptool без підключення до плати.
Ця foundation app **не прошивалася й не запускалася на ESP**; фізичні
Wi-Fi/timing/NVS/sensors/FC тести ще не виконані. На платі раніше запускалась
лише окрема USB diagnostic firmware.

Підсумкові hashes/результати: [esp32_foundation_v1.json](hardware/esp32_foundation_v1.json).
