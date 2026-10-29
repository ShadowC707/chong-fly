# USB-діагностика ESP32-S3 / GPW70

Ціль `firmware/esp32/usb_probe` — окрема стендова прошивка C++/ESP-IDF.
Вона піднімає native USB Host, друкує device/configuration descriptors і читає
interrupt IN лише для поточного Windows VID:PID `045e:028e` з інтерфейсом
`FF/5D/01`, alternate 0. Інші пристрої лише описуються в журналі.
За неоднозначних або некоректних дескрипторів endpoint не обирається.

Вона не має FC UART, RC mapping, ARM/DISARM, AUTO, Wi-Fi, сенсорних драйверів
або виклику нейромоделі. Receiver core до цього target не підключений.
USB OUT команд геймпаду не надсилаємо; стандартна USB enumeration присутня.
Якщо конкретний GPW70 потребує vendor initialization, це стане окремим
результатом перевірки, а не припущенням про сумісність.

## Відтворення

Tooling: окреме `.venv-esp32-build`, `platformio==6.2.0`, platform
`espressif32@6.12.0`, ESP-IDF 5.5.0, Xtensa 14.2.0. Профіль плати —
DevKitC-compatible ESP32-S3, 16 MB Flash, UART0 console 115200. PSRAM у цій
діагностиці не використовується. Це не підтверджує точну ревізію clone board.

```powershell
& .venv-esp32-build/Scripts/python.exe firmware/esp32/pio_local.py run --project-dir firmware/esp32/usb_probe --jobs 2
& venv/Scripts/python.exe -m pytest tests/test_usb_gamepad_probe.py tests/test_esp_flash_backup.py tests/test_esp32_host.py -q --basetemp .test-tmp-usb-probe-check
```

Спочатку написані тести: без реалізації не було target `-Probe` та модуля
резервного копіювання. Після реалізації — 43 passed (16 USB, 3 backup,
24 receiver), включно з перевіркою MSVC `/W4 /WX`. Виправлено помилку
Windows fsync: синхронізація виконується на відкритому файлі для запису.

USB fixtures штучні. Перевіряються межі дескрипторів, short/overflow,
невідповідність довжини, OUT/bulk замість interrupt IN, великий packet,
дубльований IN, інші USB класи/протоколи, signed sticks і неправильні input
reports. Кандидат input layout: 20 bytes, header `00 14`, little-endian
buttons/triggers/sticks. Це ще не захоплення raw USB GPW70.

У журналі input зберігається лише останній для друку до 10 Hz; лічильник
перезаписів видимий. Ця політика стосується лише діагностичного журналу,
не змінює заборону прихованих черг/повторів команд та не визначає control rate.
`usb_received_us` — час отримання USB на ESP, не час радіопакета геймпада.
`rf_health=unknown` залишається навіть при безперервних USB reports.

## Резервна копія і стан стенду

Перед першим записом читаємо всі 16 MB Flash через COM14. `backup_flash.py`
читає обмеженими фрагментами 64 KiB; esptool перевіряє MD5 кожного успішного
фрагмента, невдалий повторюється після перепідключення. Повний файл отримує
SHA256 та публікується лише після завершення всіх фрагментів і fsync.
Незавершені спроби залишаються `.part` або логами, не готовими backup.
Копії можуть містити старі облікові дані, тому `data/hardware_backups/`
ігнорується Git і не поширюється.

Повна копія: `data/hardware_backups/esp32_20261026_v1/flash_verified.bin`,
16 777 216 bytes, SHA256
`3a3c9c078b2777a11339b34019af7ca2d9a7e5ed5f71be0a435a9d20a7906071`.
Окрема команда esptool `verify-flash 0` підтвердила digest усіх 16 MB на платі.
Читання фрагментами потребувало 11 повторів; старий уже запущений процес після
100% читання натрапив на помилку read-only fsync Windows. Завершені байти та
log 100% перевірено, fsync виконано через `r+b`, після чого файл опубліковано
і незалежно звірено з ESP. Інструмент виправлено, тести пройшли.
Read-only `get-security-info` показав Secure Boot disabled та Flash Encryption
disabled; eFuse не змінювалися. Логи цих операцій збережені поряд із backup.

Для повернення **саме цієї плати** до стану до діагностики є повна копія.
Команда нижче — інструкція відновлення, зараз не виконується. COM port потрібно
звірити перед майбутнім використанням; не переносити копію на іншу плату.

```powershell
& .venv-esp32/Scripts/python.exe -m esptool --chip esp32s3 --port COM14 --baud 115200 write-flash 0 data/hardware_backups/esp32_20261026_v1/flash_verified.bin
& .venv-esp32/Scripts/python.exe -m esptool --chip esp32s3 --port COM14 --baud 115200 verify-flash 0 data/hardware_backups/esp32_20261026_v1/flash_verified.bin
```

Повний host regression: **885 passed**, 125.86 s. Це включає модельне ядро,
UDP loopback, C++ receiver, 16 USB і 3 backup tests.

**Фактично виконано на ESP32, 2026-10-26:** чиста збірка з двома jobs успішна;
остаточна збірка 188.23 s. App 246 032 bytes; linker RAM 12 748 bytes.
Заголовки bootloader/app перевірені: ESP32-S3, DIO, 40 MHz, 16 MB.
Записані лише образи bootloader `0x0`, partitions `0x8000`, app `0x10000`,
без erase-all/eFuse операцій. esptool підтвердив hash кожного записаного образу.
Після перезапуску через COM14 отримано `CHONG_USB_PROBE_V1`,
`USB_HOST_READY` і п'ять heartbeat до 10 259 ms без panic/reset.
Boot log: [esp32_usb_probe_boot_v1.txt](esp32_usb_probe_boot_v1.txt).
Source/build/backup hashes: [esp32_usb_probe_v1.json](esp32_usb_probe_v1.json).
При першому записі журналу приймач фізично **ще не був підключений** — це
уточнив власник. `device=0 reports=0` у цьому журналі очікувані й не доводять
несправності живлення або несумісності. Raw GPW70 input ще не підтверджений.

Живлення VBUS і стан USB-OTG jumper залишаються невідомими; прошивка не
керує ними. Звичайний стендовий крок: COM від ноутбука лишається єдиним
джерелом живлення, у native USB вставляється пасивний USB receiver через
перехідник. Якщо enumeration відсутня після фактичного підключення, потрібна
перевірка VBUS/перемички. Відсутність пристрою до підключення не є такою
підставою. Деталі плати: [ESP32_BOARD_USB_UK.md](ESP32_BOARD_USB_UK.md).
Не закорочувати перемичку навмання. GPIO19/20 зайняті native USB.

## Межі інтеграції

2026-10-27 власник повідомив про підключення receiver. Два фізичні UART
captures по 12 s (поточний стан і reset ESP із підключеним за повідомленням
receiver) показали USB_HOST_READY, відповідно 6 і 5 heartbeat,
`device=0 reports=0`, без device descriptor/input або panic.
Запис: [esp32_usb_attached_20261027_v1.json](esp32_usb_attached_20261027_v1.json),
[журнал](esp32_usb_attached_20261027_v1.txt).
Це не доказ несумісності GPW70 й не вимірювання VBUS. Поточний порт/adapter
ще не звірені фото, живлення USB не виміряне. Наступний крок — фото
підключення й мультиметра, потім конкретна перевірка живлення/USB routing.
Прошивку й FC під час цього спостереження не змінювали.

Окремо потрібні: raw GPW70 enumeration/input на ESP, вимкнення/повторне
ввімкнення геймпада при незмінно вставленому dongle, USB detach/reconnect,
калібрування осей, явна ручна авторизація та відкликання при відмовах.
Актуальні FC налаштування не змінюються. Повна інтеграція сенсорів/UDP/RC
залишається наступними кроками.

Часовий бюджет з `PI_COMPUTE_AUDIT_UK.md` не закритий: ця діагностика не
вимірює Pi inference, sensor age, clock mapping, Wi-Fi jitter або FC latency.
Host loopback не є апаратним timing evidence; TTL 40 ms/latest-sequence
та узгоджена поведінка ядра не послаблюються.
