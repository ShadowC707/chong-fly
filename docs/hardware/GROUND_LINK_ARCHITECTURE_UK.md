# Два наземні джерела та бортова ESP32

Рішення власника від 2026-10-27, після явного уточнення: **модель працює на
Raspberry Pi поза дроном; ноутбук передає тільки ручний ввід геймпада**.
Модель не переноситься на ноутбук. Початкове розміщення Pi на борту й вибір
USB receiver безпосередньо на ESP більше не є поточною ціллю.

```text
Поза дроном:
  GPW70 → його USB receiver → ноутбук ── ручний канал ──┐
                                                       ↓
  Raspberry Pi з наявною моделлю ←── сенсори ──────── ESP32 → G4
             └──────────────── команди моделі ─────────↑
На борту: сенсори, ESP32, G4, живлення та силова частина.
```

Ноутбук і Pi — два незалежні peer. Обидва спілкуються з ESP через обрану
Wi-Fi мережу; Wi-Fi topology/AP, credentials, pairing та адреси ще не налаштовані.
USB receiver і довгий USB cable/adapter прибираються з борту. USB-OTG jumper
для обраного ручного каналу не потрібний; перевірку/паяння зупинено, bridge не
виконувався. Стара USB diagnostic firmware збережена як стендовий результат;
нова Wi-Fi foundation app уже написана й зібрана, але на ESP ще не записана.
Інструкція: [ESP32_FOUNDATION_UK.md](../ESP32_FOUNDATION_UK.md).

## Ролі й допуск

- ESP збирає сенсори, фільтрує/калібрує, зберігає capture timestamps/validity
  і складає SensorSnapshot для Pi. Ручний snapshot та фактично вибраний вихід
  повертаються Pi через чинний sensor contract.
- Pi виконує наявний FlightSystem та повертає тільки наданий ним lease.
  Ваги, контролери, sealed bundle й development моделі залишаються в іншому чаті.
- Ноутбук читає GPW70 через Windows XInput. Він не виконує модель і не надсилає
  її COMMAND як заміну ручному вводу. Поточний Windows режим уже перевірений.
- ESP розрізняє peer/типи повідомлень і вирішує, яке джерело має повноваження.
  Черговість приходу пакетів не визначає джерело. Ручний пріоритет, AUTO
  authorization, STOP/ARM/reset мають явні правила й не надаються самим фактом
  отримання UDP. Відмова Pi не блокує обслуговування ручного каналу/watchdog ESP.

Pi channel зберігає CFHW v1: SENSOR 733 bytes, COMMAND 69 bytes,
RECEIPT 130 bytes. Foundation реалізує окремий CFMI v1 manual channel:
ESP CHALLENGE 40 bytes, notebook INPUT 67 bytes, session/sequence, validity,
manual takeover/STOP request та RC; CRC й перевірка свіжості на ESP clock.
Bench manual timeout 100 ms окремий від AUTO lease та не кваліфікований для
польоту. Профіль осей/газу/кнопок ще не визначений.
Не підставляємо throttle=1500 або неявний ARM.
Запит зміни режиму не дорівнює локальному наданню AUTO/reset на ESP.

На host використовується окремий fixed-peer UdpEndpoint для кожної ролі;
прийом/пам'ять обмежені. Такі ж межі потрібні на ESP. IP/port filtering,
session ID і CRC не є автентифікацією. Узгоджений trusted link/pairing ще
потрібний для апаратної реалізації.

XInput packet_number змінюється при зміні стану, а не при кожному свіжому
радіопакеті. Незмінні стіки не означають втрату зв'язку; успішне читання
USB receiver також не доводить RF freshness. Потрібен окремий тест вимкнення
GPW70 при receiver, що залишається в ноутбуці.

## Час і перевірки

Малі пакети зменшують обсяг даних, але не встановлюють верхню межу затримки.
За 50 sensor frames/s поточний SENSOR має 36 650 bytes/s payload; із COMMAND
та RECEIPT на кожний кадр разом 46 600 bytes/s. Це розрахунок для wire v1,
без Wi-Fi/UDP/IP overhead, втрат, retries радіорівня й manual channel.

Для наявного latest-sequence gate відповідь Pi має прибути **до наступного
20-ms sensor frame**. TTL до 40 ms не дозволяє команду для попереднього кадру.
Час збору конкретного сенсора й цей round trip — різні обмеження: повільний
ToF не дає автоматично більше часу для відповіді на control frame.

[PI_COMPUTE_AUDIT_UK.md](../PI_COMPUTE_AUDIT_UK.md) залишається чинним:
14/14 при нульовому service time, 11/14 при доданих 2 ms (4-ms grid симуляції).
Переміщення Pi на землю не закриває цей бюджет і не змінює inference hardware.
Висновок про energy/масу потребує окремих вимірювань; прибрані вузли й кабелі
відомі, числовий виграш споживання/аеродинаміки ще не вимірювався тут.

Спочатку перевіряємо два peer у host/mock, межі черг і розділення ролей.
Виконано окрему перевірку наявного UdpEndpoint з двома парами host loopback
socket: однакові payload від переплутаних source ports відхилено, правильні
peer збережено; backlog manual channel не споживає receive budget Pi channel.
Разом із попередніми UDP tests — **11 passed за 2.72 s**. Це перевірка
транспортного розділення за налаштованими адресами, не authentication,
manual packet semantics або ESP scheduler/Wi-Fi/FC output.
Далі ESP Wi-Fi SHADOW з двома реальними наземними пристроями: capture ages,
mapping похибка годинників, повний round trip p50/p95/p99/max, deadline misses,
мережеві втрати та відмови кожного peer окремо й разом. FC output залишається
вимкненим до перевірки арбітра й механізму відкликання на фактичній G4.
Поточні FC MSP settings ще не гарантують втрату старої команди після відмови ESP.

Джерела Windows API: [Microsoft XInput](https://learn.microsoft.com/en-us/windows/win32/xinput/getting-started-with-xinput).
