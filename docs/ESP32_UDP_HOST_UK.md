# UDP host-етап і уточнення ролі ESP32

Дата: 2026-10-26. Це наступний крок після
[CFHW v1](ESP32_WIRE_V1_UK.md); байтовий формат не змінений.
Повний [FC dump](hardware/BTFL_cli_20261026_115707_BETAFPVG473_V3.txt)
збережений без змін; актуальні факти — [inventory](hardware/INVENTORY_UK.md).

## Узгоджена відповідальність

```text
Сенсори → ESP: acquisition, validity, filtering, capture time, normalization
        → SENSOR / UDP → Pi: FlightSystem → issued lease
        → COMMAND / UDP → ESP: local authority + session/sequence/deadline gate
        → FC adapter із явним RC mapping → G4: стабілізація
```

ESP є локальним арбітром MANUAL/SHADOW/AUTO. Pi не керує ARM/AUX і не може
надати собі AUTO, очистити fault чи подовжити deadline часом надходження пакета.
При втраті команди приймач відкликає AUTO; **дія на виході FC** залежить від
ще не реалізованого ручного/fallback шляху. Просто припинити MSP output на
Betaflight 2025.12.2 не є доведеним відкликанням. Нового fallback тут немає.

Під «очищенням сенсорів» розуміємо перевірку status/quality, відсікання невдалих
вимірювань і узгоджений фільтр із відомою затримкою. У новому UDP-коді фізичних
фільтрів немає: формули, коефіцієнти та пороги не вигадані. Filter completion
не має омолоджувати capture time; held sample зберігає identity і capture angles.
PMW3901 власник уточнив як **нижній**; matrix залишається передньою, range нижнім.
Потрібні осі/масштаб/derotation і часовий зв'язок downrange із кутами FC.

«Недопустимий поворот» не можна визначити лише за числом RC без режиму/калібрування
FC: у різних режимах змінюється зміст команди. Чинні shape/range/identity/authority/
deadline перевірки збережені. Нового clipping, angle envelope або slew limit
на ESP немає; це зміни спільного control contract і потребують парних тестів.
Admission tilt envelope Bridge не є глобальною межею нахилу AUTO-маневрів.

## Чому UDP і чому явне кодування

UDP — практичний кандидат з малим транспортним обрамленням, але це не гарантія
затримки, доставки, порядку чи відсутності дублів. Кожний CFHW frame є однією
UDP datagram: SENSOR=733, COMMAND=69, RECEIPT=130 bytes. Socket send успіх не
означає receipt, receiver acceptance чи виконання FC. Старі команди не повторюємо.

C++ structs доречні як локальні типи. Надсилати `sizeof(struct)` байтів без
специфікації не потрібно: padding/alignment/endian залежать від платформи.
CFHW уже визначає порядок байтів і числа; native decoder читає їх явно.
Немає потреби додавати JSON/брокер/новий framework для цього обміну.

## Що реалізовано

- `edge/hardware_udp.py`: nonblocking UDP endpoint, точний налаштований IPv4
  peer, обмежений scan budget (типово 8), bounded OS receive buffer, max frame
  1024 bytes. Oversize відхиляється, у тому числі Winsock WSAEMSGSIZE; не
  обрізається у нібито валідний frame. Немає автоматичного peer discovery.
- `PiUdpPump`: той самий pinned `PiHardwareAdapter`/FlightSystem, один send на
  команду, обмежений pending/receipt/observation storage. Невалідний пакет не
  генерує замінну RC-команду. Старий session не викликає rebind. Receipt лише
  записує спостереження; не дає повноважень і не спричиняє retransmit.
- `edge/hardware_udp_probe.py`: реальні сокети **127.0.0.1** і native C++ core
  через тестовий pipe. Усі три повідомлення проходять UDP; RC acceptance
  залишається рішенням C++ receiver. SENSOR/RECEIPT codec ще виконується в Python.
- `tests/test_hardware_udp.py`: byte identity, чужа адреса, bounded flood scan,
  oversize, duplicate preservation, malformed/current vs old session,
  прострочений clock mapping, bounded receipts та наскрізний loopback.

Exact-peer filter не є автентифікацією. Session/clock binding задає локальний
trusted caller до I/O. Для справжнього Wi-Fi ще потрібні pairing/access policy,
clock estimator, restart lifecycle і конфігурація peer; паролі в чат не передаються.
За замовчуванням endpoint слухає лише 127.0.0.1. Probe не відкриває зовнішній порт.

Черга ОС може відкидати datagrams; Python counters не бачать усі kernel drops.
Scan exhaustion означає лише використання ліміту, не виміряну глибину черги.
Queued кадри обробляються в порядку надходження без relabel/latest-wins;
порушення контрольного часу передається чинному Bridge, а не приховується.
Receipts і pending issue history типово 64 записи; eviction рахується.

## TDD та результат

Початковий RED — відсутній UDP-модуль. Після реалізації локальна перевірка не
отримувала datagrams; окремий мінімальний тест із двома звичайними UDP sockets
теж завершився timeout. Повтор поза мережевим обмеженням середовища виконав
8/8 tests; додано перевірку класифікації пошкодженого повнорозмірного кадру.
Регресія UDP + попередніх host-contract tests: **59 passed in 5.91 s**.
Повний suite після цього: **866 passed in 69.94 s**, без пропущених тестів.
Process-local temp і pytest basetemp розташовані у `.test-tmp-udp-*` всередині
проєкту; системні налаштування та правила мережевого доступу не змінювалися.
Підсумок із source hashes: [udp_host_validation_v1.json](hardware/udp_host_validation_v1.json).

Probe: 40 SENSOR → 40 COMMAND → 40 RECEIPT, SHADOW перші 30 кадрів,
10 прийнятих leases, watchdog_expired після завершення обміну, FC attempts=0.
Результат: `data/diagnostics/esp32_udp_host_v1/report.json` (локальний output).

Часи приймача/моделі у probe **синтетичні**, з періодом 20 мс. Окремо
виміряний host wall RTT від початку кодування SENSOR до відповіді C++ harness:
p50=661.5 мкс, p95=808.435 мкс, p99=1454.436 мкс, max=1831.8 мкс.
Це Windows loopback + NumPy + IPC, без receipt return trip, не ESP/Pi/Wi-Fi.
Успішний loopback не доводить виконання hardware deadline. Production scheduling
і валідація ages відносно фактичного completion ще потребують окремого рішення.

Незакритий [PI_COMPUTE_AUDIT_UK.md](PI_COMPUTE_AUDIT_UK.md) залишається чинним:
попередні 11/14 із доданими 2 мс не «виправлені» цим transport-кодом. Не змінено
TTL, latest-sequence semantics, ваги, controllers, config v3 чи sealed artifacts.

## Відтворення

```powershell
& firmware/esp32/build_host.ps1 -OutDir .test-tmp-udp-build-repeat
& venv/Scripts/python.exe -m pytest tests/test_hardware_udp.py tests/test_hardware_wire.py tests/test_hardware_mock.py tests/test_esp32_host.py -q --basetemp .test-tmp-udp-tests-repeat
& venv/Scripts/python.exe -m edge.hardware_udp_probe --receiver-exe .test-tmp-udp-build-repeat/receiver_host.exe --out data/diagnostics/udp_host_repeat/report.json
```

Потрібне середовище, що дозволяє локальне UDP. Будь-які reports — у новий файл.
ESP build, Wi-Fi driver, sensor filtering, telemetry alignment та FC adapter
не заявлені як реалізовані або випробувані на обладнанні.

Джерела API: [Python 3.12 socket](https://docs.python.org/3.12/library/socket.html),
[ESP-IDF BSD sockets](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/lwip.html).
Поточний ESP-IDF guide використано лише для підтвердження наявності socket API;
версію SDK для firmware ще не вибрано.
