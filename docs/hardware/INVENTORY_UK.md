# Апаратна інвентаризація Chong-Fly

Дата: 2026-10-26. Джерела: відповіді власника в чаті,
[початковий CLI export](BTFL_cli_20261026_113946_BETAFPVG473_V3.txt) та
[повний dump all](BTFL_cli_20261026_115707_BETAFPVG473_V3.txt).
Копії незмінні; SHA-256 початкового export:
`62bd35fa9b86ab4f4529e73b6431f720b49ca1410fb838ca24581c63cad075ea`.
SHA-256 повного dump:
`97bb91b7d0b08d821cc4bd176695359415fad010ba651c47d77c45da58090b99`.
Текст `defaults nosave`/`save` у файлі — вивід backup, не виконувані нами команди.

**Поточне розміщення, уточнення власника 2026-10-27:** на борту сенсори,
ESP32 та G4; поза дроном Pi з моделлю і окремий ноутбук з GPW70 USB receiver.
Вони спілкуються з ESP як два окремі peer. Це замінює попередній вибір direct
USB Host/manual без ноутбука. [Поточна архітектура](GROUND_LINK_ARCHITECTURE_UK.md).

| Вузол | Підтверджено | Ще невідомо / не перевірено |
|---|---|---|
| FC | export: BETAFPVG473_V3, BEFH, STM32G474; Betaflight 2025.12.2, commit 79065c96b, config e378347, MSP API 1.47 | фактичне приймання RC від ESP, вихід і failsafe на стенді |
| FC UART | UART1 MSP 115200 (TX PA9/RX PA10); UART2/3 function=0; UART4 function=131073; VCP MSP | куди припаяно другий кінець UART1, фізична доступність pads/рівні; UART4 не вважати вільним |
| FC sensors | status виявив ICM42605 gyro/ACC та DPS310 barometer | доступність/частота/latency attitude telemetry і capture alignment |
| FC motors | власник налаштував; export DSHOT300, bidirectional ON, 12 poles, yaw reversed | апаратний mapping/стенд/ручний політ не підтверджені цим файлом |
| RC / modes | RX_SERIAL вимкнений; provider CRSF; map AETR1234; MSP override mask=0; ARM на AUX1, ANGLE/HORIZON на AUX2 із перекриттям | сумісний активний RC source, фактичний режим, незалежний ARM/kill, ручний канал |
| FC failsafe | DROP, failsafe_delay=15 (Stage 1 period 1.5 с), throttle=1000, recovery_delay=5; flight rxfail=a, AUX=h | фактична реакція при зникненні обраного RC/ESP/Pi; це не 40-мс receiver lease |
| FC geometry | align_board_roll/pitch=0, align_board_yaw=-45; profile 0 angle_limit=60 | реальний монтаж FC/сенсорів та відповідність команд кутам, осі й знаки |
| ESP32 | ESP32-S3 rev.0.2, Flash 16 MB, reported PSRAM 8 MB; USB diagnostic реально запускалась; app foundation зібрана на ESP-IDF 5.5 | foundation boot/NVS/Wi-Fi/tasks на апараті, реальні сенсорні drivers та FC output ще не перевірені |
| Pi | Pi 4B 8 GB; поза дроном; саме тут залишається модель | ОС, доступ, реальний service time; наземний Pi↔ESP timing |
| PMW3901 | власник уточнив: дивитиметься **вниз**, фільтрація не завершена | точний breakout, осі/масштаб, derotation, якість, pinout |
| ToF matrix | дивиться вперед | точне маркування, 8×8 режим, зональна орієнтація, шина/адреси |
| VL53L0X V2 | дивиться вниз | точний модуль, mounting offset, error/range/validity, шина/контакти |
| GL5528 | лише історичний список | монтаж/канал відсутні; до memory моделі не під'єднуються |
| Pi↔ESP | CFHW v1, C++ encoders/receiver, host mock із pinned model; ESP Wi-Fi/UDP foundation зібрана | фізичний Wi-Fi/два peer, pairing/clock sync і latency budget не підтверджені |
| Ручне керування | GPW70 → USB receiver → ноутбук → Wi-Fi/UDP → ESP; Windows XInput перевірено; CFMI challenge/input і manual gate пройшли host mock | реальний notebook sender/calibration, Wi-Fi, radio-loss поведінка, mapping і незалежний kill |

## Виявлені передумови інтеграції

Фото ESP тепер отримано й збережено. Маркування, кандидат USB-підключення та
перша перевірка мультиметром: [ESP32_BOARD_USB_UK.md](ESP32_BOARD_USB_UK.md).
На USB-OTG pads повідомлено 4.84/0.15 V; прямий receiver VBUS не перевірений.
Для чинної схеми notebook USB receiver ця перемичка не потрібна.
Поточна firmware foundation: [ESP32_FOUNDATION_UK.md](../ESP32_FOUNDATION_UK.md).

Поточний `simulation/pmw3901_emulator.py` формує дві компоненти швидкості тіла
відносно поверхні під дроном, нормовані на height та max_flow_rate.
Власник виправив попереднє повідомлення: PMW3901 буде **знизу**. Попередню
передумову про передній flow знято. Геометрія тепер відповідає задуму моделі,
але raw counts ще потребують калібрування, осей/знаків, масштабу й компенсації
обертання. Сам факт нижнього монтажу не доводить числової еквівалентності.

AltitudeHold потребує roll/pitch на час capture downrange. Поточні кути окремо
потрібні для admission. Джерело telemetry, годинник, точність вирівнювання і
допустима похибка ще не встановлені. Виявлений ACC сам по собі цього не закриває.

UART1 MSP — кандидат для конфігурації/телеметрії; сам факт MSP не доводить
дозволений спосіб RC input у цій збірці. При виборі MSP override для 2025.12.x
офіційний опис попереджає про утримання останніх MSP RC після зупинки відправника.
Тому відмова ESP output не може автоматично трактуватися як release на G4.
Потрібен перевірений механізм відкликання, ручний канал і props-off tests.
All-neutral або throttle 1500 як fallback не визначені.

Повний dump отримано й прочитано. Налаштування FC не змінювалися. Наявний
CRSF provider не означає, що приймач CRSF підключено або що джойстик сумісний.
MSP на UART1 не дорівнює налаштованому MSP RC input. У dump немає активного
MSP OVERRIDE mode (permanent ID=50), mask=0; вмикати його автоматично не можна.

Наш RC order T/R/P/Y потрібно явно відображати у поточний FC AETR як R/P/T/Y:
індекси (1,2,0,3). Це mapping для зафіксованого dump, не вибір RC протоколу.
ARM на AUX1 має діапазон [1700,2100), ANGLE на AUX2 [900,2100), HORIZON
на AUX2 [1300,1700). При такому перекритті фактичний стан потрібно читати/перевіряти,
а не вважати ANGLE гарантованим. AUX-входи не належать чотирьом нейронним каналам.

Поточний ручний шлях: GPW70 із USB-приймачем у ноутбуці → Wi-Fi/UDP → ESP.
Pi з моделлю працює поза дроном як другий peer. CFMI manual packet уже
реалізований у foundation; ще потрібні calibrated notebook sender,
network pairing, clock mapping та перевірений шлях відкликання при відмові ESP.
Hardware AUTO до закриття цих передумов не реалізоване. Новий UDP
host-крок і обмеження описано в [ESP32_UDP_HOST_UK.md](../ESP32_UDP_HOST_UK.md).

Уточнена власником модель — **GPW70 Sports Black**; попереднє позначення PN-01B
не використовуємо для вибору драйвера. Комплектний USB-приймач є у власника.
[Сторінка виробника](https://www.bloody.com/en/product.php?id=356&pid=50) та
[офіційна інструкція](https://www.bloody.com/en/manuals/pdf/gpw70.pdf) описують
2.4G wireless через USB receiver і USB wired; Bluetooth у цих способах не заявлений.
PC XInput підтверджено документацією, але це не доводить Xbox Wireless/BLE
або готову сумісність із ESP. Direct USB Host був попереднім кандидатом;
поточний вибір — приймач у ноутбуці та окремий manual network channel.
Деталі та апаратні невизначеності: [GPW70_INTEGRATION_UK.md](GPW70_INTEGRATION_UK.md).
Окрему C++/ESP-IDF USB-діагностику тепер зібрано, записано на ESP32 після
повної перевіреної резервної копії та перевірено її запуск: `USB_HOST_READY`,
п'ять heartbeat без panic/reset. На момент першого журналу receiver ще не був
підключений (уточнення власника), тож `device=0` очікуваний. Це підтверджує
startup USB Host library, а не enumeration чи керування з GPW70.
Матеріали: [USB_PROBE_FIRMWARE_UK.md](USB_PROBE_FIRMWARE_UK.md),
[esp32_usb_probe_v1.json](esp32_usb_probe_v1.json).
Read-only Windows probe тепер підтвердив XUSB-compatible режим 045E:028E та
читання стану XInput; [фактичний запис](gpw70_windows_v1.json). Це звужує вибір
ESP-драйвера до USB XUSB/XInput для поточного режиму. Сумісність на ESP ще не
підтверджена, raw USB configuration/input не захоплені, FC-команди не передавалися.

Джерела для тлумачення, не підтвердження стендової сумісності:
[Betaflight CLI](https://betaflight.com/docs/wiki/guides/current/Cli),
[Serial](https://betaflight.com/docs/wiki/guides/current/Serial),
[MSP companion control](https://betaflight.com/docs/wiki/guides/current/MSP-Companion-Computer-Control).
Також перевірено source саме commit `79065c96b`:
[MSP function bit=1](https://github.com/betaflight/betaflight/blob/79065c96b/src/main/io/serial.h),
[MSP override без перевірки віку в read](https://github.com/betaflight/betaflight/blob/79065c96b/src/main/rx/msp_override.c),
[збереження MSP RC frame](https://github.com/betaflight/betaflight/blob/79065c96b/src/main/rx/msp.c).
Тлумачення AUX IDs/діапазонів і failsafe period звірене з
[msp_box.c](https://github.com/betaflight/betaflight/blob/79065c96b/src/main/msp/msp_box.c),
[rc_modes.c](https://github.com/betaflight/betaflight/blob/79065c96b/src/main/fc/rc_modes.c),
[failsafe.c](https://github.com/betaflight/betaflight/blob/79065c96b/src/main/flight/failsafe.c).
Версію FC встановлено саме з export; новіші налаштування з current docs
не переносяться у 2025.12.2 автоматично.
