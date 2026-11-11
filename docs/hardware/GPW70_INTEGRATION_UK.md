# GPW70: документально підтверджене та наступна перевірка

Дата: 2026-10-26. Власник уточнив **A4Tech Bloody GPW70 Sports Black** та
підтвердив наявність комплектного USB-приймача. Старе PN-01B не є основою драйвера.

## Відомо з первинних джерел

- Підключення: власний 2.4G wireless канал через комплектний USB receiver або
  дротовий USB. Bluetooth серед описаних способів відсутній.
- Є PC/XInput і PC/DInput режими. Це підтверджує PC API режим, а не тотожність
  усіх USB/радіопротоколів контролерам Microsoft.
- Інструкція описує LED-2 для PC/XInput і LED-1+LED-2 для PC/DInput.
- Для дротового режиму треба під'єднати кабель і натиснути Home; одночасно
  вставлений USB receiver має пріоритет, і кабель тоді може лише заряджати.
- Заявлений виробником радіус — до 10 м, не перевірена дальність RC-каналу.

Джерела: [Bloody product](https://www.bloody.com/en/product.php?id=356&pid=50),
[GPW70 Quick Start Guide](https://www.bloody.com/en/manuals/pdf/gpw70.pdf).
В інструкції є розбіжність часу утримання Home для зміни режиму (таблиця й текст);
не фіксуємо цю тривалість у нашому контракті. Режим перевіряємо за LED і USB даними.

## Обрана схема ручного каналу

**Оновлення 2026-10-27 — чинний вибір власника:**

```text
GPW70 → USB receiver → ноутбук ── manual Wi-Fi/UDP ──→ ESP32 → G4
Окрема Raspberry Pi з моделлю ←── сенсори/статус ──── ESP32
                            └── AUTO lease ────────→ ESP32
```

**Pi поза дроном, модель лишається на Pi; ноутбук передає тільки геймпад.**
Це два peer, не один процес на ноутбуці. Деталі й незакритий timing budget:
[GROUND_LINK_ARCHITECTURE_UK.md](GROUND_LINK_ARCHITECTURE_UK.md).
USB Host/перемичка на ESP тепер не потрібні ручному каналу; їх доведення відкладено.

Нижче збережено попередній вибір до цього уточнення, а не поточну ціль:

```text
GPW70 → комплектний 2.4G USB receiver → USB Host на ESP32-S3
                                            → local MANUAL/AUTO arbiter → G4
```

Власник підтвердив роботу **без ноутбука**: комплектний USB-приймач має бути
під'єднаний до ESP32-S3, а не залишатися у наземному ПК. Наземний Wi-Fi bridge
з ноутбука не входить в обрану схему. Це рішення про архітектуру,
**не підтверджена сумісність** USB driver/живлення. Власний радіоканал GPW70
закінчується на USB receiver; його не замінюємо Wi-Fi/Bluetooth радіо ESP.
Ручний ввід безпосередньо на ESP не потребує роботи нейромережі або Pi-процесу.
Це все ще спільна точка відмови ESP; незалежне припинення керування FC залишається
окремою вимогою. Жодного ARM/mode mapping із кнопок поки не призначено.

Ручний USB-ввід і команди Pi — окремі джерела для локального арбітра ESP.
При MANUAL Pi lease не має права потрапити до FC; при AUTO він усе одно проходить
session/sequence/deadline gate. Перезапуск/зависання Pi не повинен блокувати
обслуговування ручного джерела або watchdog ESP. Фізичну поведінку при відмові
самого ESP та втраті радіозв'язку ще треба визначити й перевірити.

Espressif надає [USB Host](https://docs.espressif.com/projects/esp-usb/en/latest/esp32s3/usb_host.html)
і [usb_host_hid 1.2.1](https://components.espressif.com/components/espressif/usb_host_hid/versions/1.2.1/readme).
HID driver підтримує interfaces class 0x03, raw input reports та читання report
descriptor. Спочатку треба встановити, що саме GPW70 видає в кожному режимі:
XInput у Windows не доводить наявності потрібного HID interface. Не встановлюємо
generic Xbox/Bluetooth parser за аналогією з назвою.

Фото конкретної плати отримано: ESP32-S3-N16R8, COM/USB, майданчики USB-OTG;
[розбір і вимірювання](ESP32_BOARD_USB_UK.md). USB розведення/живлення ще не перевірені.
Наявний USB-UART programming connector сам по собі не доводить можливість
під'єднати receiver як USB Host. Pinout і VBUS wiring до перевірки плати не задані.

## Поточне спостереження на Windows

Власник надіслав screenshot `joy.cpl`: **Controller (XBOX 360 For Windows), OK**
і повідомив, що керування працює. Додатковий read-only probe цього комп'ютера
знайшов USB VID=045E/PID=028E, revision=0101, driver `xusb22`/`xusb22.inf` і
compatible IDs `MS_COMP_XUSB10`, class FF/subclass 5D/protocol 01.
XInputGetState успішно прочитав slot 0 (code=0); інші три slots не підключені.
Поточний snapshot стіків/тригерів збережено в
[gpw70_windows_v1.json](gpw70_windows_v1.json).

Це уточнює попередній HID-кандидат: у **цьому режимі** плануємо USB XUSB/XInput
host driver, а не автоматично `usb_host_hid`. Windows також створює HID child
з `input.inf`, але це не доказ USB interface class=03 на приймачі. Потрібні
повний configuration descriptor/endpoints і фактичні input transfers.
Linux [xpad.c](https://github.com/torvalds/linux/blob/master/drivers/input/joystick/xpad.c)
відносить 045E:028E до XTYPE_XBOX360, окремо від XTYPE_XBOX360W receiver.
Це джерело для вибору кандидата, не перевірка GPW70 на Linux або ESP.

Поточні дані отримано через Windows-драйвер, не raw USB capture. Не створюємо
з них вигаданий 20-byte packet fixture. Snapshot не є калібруванням neutral:
right trigger під час читання мав значення 136. XInput packet_number теж не
можна трактувати як capture timestamp чи доказ свіжого радіопакета.

## Наступний стендовий крок

Для поточної схеми receiver у ноутбуці додано raw XInput/stub adapter і
дротовий calibration probe: [GPW70_CALIBRATION_UK.md](GPW70_CALIBRATION_UK.md).
Wired data mode підтверджений інструкцією, фактичний capture нашого cable
ще очікується. USB Host/decoder steps нижче зберігають історичну попередню схему.

1. Під'єднати комплектний receiver до Windows PC й увімкнути GPW70 кнопкою Home.
2. Перевірити осі/кнопки через `Win+R → joy.cpl → Properties`.
3. Зчитати VID/PID, USB interfaces і, якщо є HID, report descriptor та raw reports.
   За потреби окремо повторити для іншого PC mode; дані різних режимів не змішувати.
4. На host створити fixtures з **фактичних** reports і TDD parser → ManualSample.
   Тестувати center/range/sign, кнопки, invalid reports і timeout, а не вигаданий
   Xbox layout. Throttle=1500 не призначається як універсальний ручний neutral.
5. Окремо перевірити втрату радіозв'язку/вимкнення геймпада та unplug receiver.
   USB reports можуть продовжувати приходити від receiver; їх надходження саме
   по собі не доводить свіжість радіоканалу або присутність оператора.

Перші два пункти виконано на Windows: є повідомлення власника та read-only
PnP/XInput probe. Тепер окрема діагностична C++/ESP-IDF прошивка фізично
стартувала на ESP32-S3: `USB_HOST_READY`, п'ять heartbeat без panic/reset.
Перший журнал знято без підключеного receiver, за уточненням власника.
Деталі: [USB_PROBE_FIRMWARE_UK.md](USB_PROBE_FIRMWARE_UK.md).
Raw-report candidate decoder має 16 synthetic host tests; фактичні raw GPW70
reports, radio-loss behavior та вихід FC ще не перевірялися. Змін FC немає.
Попередні 866 tests не є свідченням сумісності GPW70 з ESP. Поточний нейронний/
UDP контракт незмінний. Наступний факт — enumeration/input після підключення
receiver до native USB ESP. Якщо їх немає, перевіряємо USB routing/VBUS
конкретної clone board; живлення receiver не вважаємо вже підтвердженим.
