# Передача апаратної частини колезі

Стан на 2026-10-29. Цей документ — поточний маршрут роботи; попередні
датовані inventories та JSON-звіти збережені як історичні докази.

## Схема й готовність

На дроні: сенсори → ESP32 → G4. На землі **два пристрої**:
Raspberry Pi 4B 8 GB виконує модель, ноутбук читає GPW70. Кожний має окремий
UDP-канал до ESP. ESP перевіряє команди й обирає MANUAL/SHADOW/AUTO.
Pi не надає собі AUTO, не керує ARM і не очищає fault.

Готові wire codecs, C++ receiver/арбітр, sensor cache, Pi adapter, bounded UDP,
ESP-IDF app foundation, raw XInput/mock/replay. Модель, конфігурація та sealed
bundle не змінені. Сенсорні драйвери й FC output поки відсутні: звичайна app
публікує invalid sensors, FC stage завжди `not_attempted`. За замовчуванням
Wi-Fi та bench AUTO вимкнені. Foundation зібрана, але ще **не прошита**.

Перевірено 2026-10-29: **969 tests passed** у повному suite та **162 passed**
в окремому hardware subset; обидва mock-сценарії й ESP target build успішні.
Після перевірок hashes моделі, конфігурації та захищених модулів незмінні.

| Перевірка | Що встановлено |
|---|---|
| Windows host | Python/C++ parity, пакети, fault/reset, watchdog, manual priority, mock із pinned FlightSystem |
| ESP target build | Код компілюється для ESP32-S3; це не перевірка запуску, Wi-Fi чи NVS |
| Попередній фізичний ESP test | Лише запуск окремої USB diagnostic firmware; GPW70 enumeration не отримано |
| GPW70 Windows | Раніше працював через receiver/XInput; wired capture та RF-loss test не завершені |
| Raspberry Pi | Реального повного циклу й часового бюджету ще не виміряно |
| G4 | Dump прочитано; RC input, AUX/ARM, telemetry alignment та відкликання ще не перевірено |

## Перша перевірка без обладнання

Команди нижче запускати з кореня репозиторію на Windows. Потрібні Python
3.10+ (тут перевіряли 3.12), MSVC x64 Build Tools і Windows SDK. Якщо вже є
робочий `venv`, повторно створювати його не потрібно. Для окремого легкого
середовища апаратних тестів достатньо NumPy та pytest:

```powershell
py -3.12 -m venv .venv-hardware
& .venv-hardware/Scripts/python.exe -m pip install -r firmware/esp32/requirements-validation.txt
& .venv-hardware/Scripts/python.exe -m scripts.check_hardware_host --out-dir data/diagnostics/colleague_host_v1
```

Перевірка збирає native C++ harness, запускає hardware tests, два mock-сценарії
(нормальний та manual takeover/disconnect), запис і replay штучного вводу.
Реальні UDP sockets відкриваються лише на loopback з тимчасовими портами.
COM, XInput, обладнання, зовнішні peers та прошивання не використовуються.
Новий output directory обов'язковий: наявні докази не перезаписуються.

Результат — `report.json`, `tests.xml` та окремі журнали кроків у вибраному
каталозі. Ненульовий exit code означає невдалу перевірку. Зміна захищених
файлів між початком і кінцем теж робить результат невдалим. `success=true`
означає успіх **host-перевірки**; усі hardware qualification flags лишаються false.

`--full-suite` запускає весь suite замість hardware subset і потребує повного
дослідницького середовища репозиторію. Це опція для наявного основного venv,
не вимога для легкого `.venv-hardware`.

## Збірка ESP без плати

SDK ізольований від Python моделі:

```powershell
py -3.12 -m venv .venv-esp32-build
& .venv-esp32-build/Scripts/python.exe -m pip install -r firmware/esp32/requirements-build.txt
& .venv-hardware/Scripts/python.exe -m scripts.check_hardware_host --build-firmware --out-dir data/diagnostics/colleague_build_v1
```

Перший build завантажує SDK у Git-ignored `.pio-esp32/` і потребує мережі.
`--build-python` дозволяє вказати інший Python із PlatformIO.
Ціль — `firmware/esp32/app`, а не історичний `usb_probe`.
Команди тільки збирають firmware й записують розміри/хеші: upload відсутній.

Поки `app/local_config.hpp` не створений, використовується безмережевий default.
Якщо локальний профіль уже є, build включає саме його; він Git-ignored і може
містити секрети. Для базової перевірки використати чистий checkout без такого
файлу. Не публікувати firmware binaries із мережевими паролями.

## Перші дії колеги з обладнанням

1. **Геймпад на ноутбуці.** Виконати [raw capture інструкцію](hardware/GPW70_CALIBRATION_UK.md):
   визначити XInput slot, записати neutral і sweep, потім окремо receiver-present /
   gamepad-off. Дротовий режим спершу підтвердити фактичним читанням. Збережені
   файли можна передати для програмної перевірки без контролера:

   ```powershell
   & .venv-hardware/Scripts/python.exe -m edge.gamepad_probe --source replay --replay data/diagnostics/gamepad_probe/neutral.json --out data/diagnostics/gamepad_probe/neutral_replay.json
   ```

   Replay зберігає значення, старі timestamps і disconnect; не створює новий
   фізичний ввід, калібрування чи RC-команди. Він відхиляє пошкоджені/суперечливі
   файли й не довіряє записаному summary. Файл обмежений 8 MiB / 12 000 samples.
   Газ, ARM, deadman і button mapping ще не призначені.

2. **Інвентар сенсорів і проводів.** Зафіксувати точні breakout modules,
   живлення, SPI/I2C/UART контакти, адреси й осі. PMW3901 та VL53L0X дивляться
   вниз, матриця — вперед. Модель матриці й остаточний pinout ще невідомі.
   Для altitude потрібні roll/pitch на час захоплення range; поточні кути
   не можна дописувати до старого range як новий sample.

3. **G4 без пропелерів.** Звірити фактичний dump із [inventory](hardware/INVENTORY_UK.md).
   UART1 MSP уже є, але це не доводить робочий RC input. Потрібні окремо
   підтверджений спосіб подавання RC та реакція FC на зупинку/відмову ESP.
   Зупинка надсилання MSP сама по собі не підтверджує відкликання. Канали
   T/R/P/Y мають явний mapping у AETR; AUX/ARM — окремий контракт.

4. **ESP та мережа у SHADOW.** Після узгодження підключень і профілю — окремий
   запуск foundation на платі, два реальні peers, сенсорні capture ages,
   clock mapping та вимірювання циклу. Зараз немає готового live Pi runner
   з clock estimator і calibrated notebook sender; host mock їх не замінює.

## Незакритий часовий бюджет

[PI_COMPUTE_AUDIT_UK.md](PI_COMPUTE_AUDIT_UK.md) залишається чинним:
14/14 при нульовому service time, 11/14 при доданих 2 ms на 4-ms simulation grid.
Передача Pi на землю та малі UDP packets не закривають ці відмови.

Поточний контракт потребує відповіді для останнього кадру **строго до 20 ms**
від його початку на ESP; lease TTL максимум 40 ms рахується від того самого
початку. Повільний ToF не збільшує допустимий round trip. TTL та пороги заради
проходження стенду не послаблюються.

Для наступного апаратного звіту потрібні raw записи session/sequence,
ESP frame-start та command-receive times на одному ESP clock; Pi tick start/end
на Pi clock; capture times/ages/validity кожного сенсора; причини відхилення,
deadline misses, send/receive/drop/queue counters і навантаження обох peers.
Потім рахувати p50/p95/p99/max та втрати, окремо для Pi 4B. Вимірювати
подальший FC output окремо від receiver acceptance. Без clock mapping не
віднімати timestamp Pi від timestamp ESP. Поточний console STATUS не містить
усієї цієї траси: інструментування ще треба додати перед вимірюваннями.

Рішення про апаратний AUTO потребує фактичних вимірювань і перевіреного
відкликання на G4. До того продовжувати host/mock та SHADOW.

## Навігація й докази

- [Основа прошивки й CFMI](ESP32_FOUNDATION_UK.md).
- [CFHW v1](ESP32_WIRE_V1_UK.md), [UDP adapters](ESP32_UDP_HOST_UK.md).
- [Розміщення двох наземних пристроїв](hardware/GROUND_LINK_ARCHITECTURE_UK.md).
- Поточний перевірений результат без обладнання:
  [hardware_host_check_20261029_v1.json](hardware/hardware_host_check_20261029_v1.json).

Локальні журнали, captures, SDK, приватний config і повний Flash backup у Git
не включаються. Історичні JSON reports описують джерела на зазначену в них дату;
їхні source hashes не слід трактувати як hashes майбутніх редагувань.
