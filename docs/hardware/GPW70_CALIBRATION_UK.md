# GPW70: заглушка вводу й дротова перевірка на ноутбуці

Дата: 2026-10-27. Поточна архітектура незмінна: Pi виконує модель,
ноутбук обслуговує геймпад, ESP збирає сенсори й обирає джерело керування.
Для цього етапу геймпад підключається до **ноутбука**, не до USB Host ESP.

Найкоротший наступний крок — raw input adapter і калібрування. Контроль G4
потребує додаткової перевірки активного RC protocol та відкликання останньої
команди. У наданому dump UART1 MSP не дорівнює налаштованому RC input;
припинення MSP override messages не доводить відкликання на цій версії FC.

## Що додано

`edge/gamepad_input.py` визначає один `InputSource.poll()` для:

- `MockGamepadSource`: обмежений сценарій штучних raw input samples;
- `XInputSource`: читання одного явно вибраного Windows slot через готовий
  XInputGetState, без автоматичного переходу до іншого контролера.

Обидва повертають той самий GamepadSample. У ньому notebook monotonic read
completion time, sample ID, API status та raw axes/buttons. При від'єднанні
`state=None`: попередні стіки не повертаються як свіжий ввід. Після вичерпання
mock сценарію теж повертається відсутність вводу, а не neutral чи повтори.
Відсутній DLL/драйвер не вмикає mock автоматично.

`edge/gamepad_probe.py` записує bounded raw capture: до 60 s, 1..200 Hz,
максимум 12 000 samples. Файл створюється виключно новий, не перезаписується.
Summary містить min/max/median шести осей і спостережені button masks.
`neutral`/`sweep` — мітки фази від оператора, не автоматично перевірений профіль.
Ці модулі не мають UDP/ARM/FC output; не змінюють калібрування Windows/GPW70.

Незмінний XInput packet_number означає незмінний стан, а не втрату зв'язку.
API poll timestamp не є часом фізичного руху стіка або RF packet. Точний
пристрій і фізичний transport не встановлюються за одним XInput slot.

## Дротовий режим

Офіційна інструкція GPW70 підтримує USB wired data. Порядок:

1. Вийняти комплектний 2.4G receiver.
2. З'єднати сам геймпад із ноутбуком кабелем із data lines.
3. Натиснути Home: без цього підключення може лишатися заряджанням.

Якщо receiver залишається підключеним, GPW70 надає пріоритет 2.4G і кабель
може тільки заряджати. Джерело: [GPW70 Quick Start Guide](https://www.bloody.com/en/manuals/pdf/gpw70.pdf).
Не припускаємо, що будь-який USB-C cable передає дані або wired mode вже
підтверджений для наявного кабелю. Сумісність перевіряється фактичним input.

## Запуск

```powershell
& venv/Scripts/python.exe -m edge.gamepad_probe --list
& venv/Scripts/python.exe -m edge.gamepad_probe --source mock --seconds 1 --out data/diagnostics/gamepad_probe/mock.json
```

`--list` лише читає connection status чотирьох Windows slots. Вибрати slot
потрібно явно; приклад нижче використовує 0 тільки якщо його показало connected.

```powershell
# Не торкатися стіків/тригерів під час цієї фази:
& venv/Scripts/python.exe -m edge.gamepad_probe --source xinput --slot 0 --phase neutral --seconds 5 --out data/diagnostics/gamepad_probe/neutral.json
# Для наступної фази — повний хід обох стіків, тригерів і кнопок:
& venv/Scripts/python.exe -m edge.gamepad_probe --source xinput --slot 0 --phase sweep --seconds 15 --out data/diagnostics/gamepad_probe/sweep.json
```

Межі стіків XInput -32768..32767, тригерів 0..255 — межі формату API,
не виміряне калібрування конкретного GPW70. Після captures ще визначаємо
реальні centers/ranges, dead zones, знаки та розклад T/R/P/Y. Жоден пружинний
стік або trigger не призначається газом без явного профілю.

## Що ці перевірки доводять

1. Mock — працездатність програмного читача та обробки від'єднання.
2. Фактичний wired capture — дані осей/кнопок конкретного підключення.
3. Наявний foundation mock — CFMI/CFHW packets, manual takeover та lease gate
   у C++ з pinned FlightSystem; час і сенсори там штучні.
4. Фактичну реакцію G4 доведе тільки окремий стенд ESP→FC з перевіркою
   channel mapping, режиму й відкликання. Пункти 1–3 її не підтверджують.

Дротовий тест не кваліфікує 2.4G RF-loss behavior або Wi-Fi timing. Після
нього можна окремо підмінити wired source на wireless receiver на ноутбуці,
зберігаючи формат raw samples та майбутній calibrated mapper.

## Фактичний стан перевірки

TDD: нові tests спочатку впали через відсутній input module; після реалізації
**19 input cases пройшли**, разом із foundation core — **53 passed за 2.68 s**.
Перевірено Windows 16-byte ABI, signed extremes, fixed slot, disconnect без
cached RC, незмінний packet number, bounds, mock exhaustion і bounded capture.

Read-only Windows query `xinput1_4.dll` зараз повернув 1167 для slots 0..3:
контролер у момент перевірки не виявлено. Це не висновок про несумісність
дротового режиму; його підключення ще не підтверджене. Mock capture CLI успішний.
ESP/G4 firmware/configuration не змінювали, команд не надсилали.

Оновлення 2026-10-29: додано offline `--source replay --replay <capture.json>`.
Воно читає всі записані samples без затримок, зберігає старі timestamps,
перевіряє schema/bounds/identity/disconnect і не відкриває XInput або UDP.
Replay — програмне відтворення, не підтвердження calibration/RF freshness.
Команда й актуальний маршрут колеги: [HARDWARE_COLLEAGUE_UK.md](../HARDWARE_COLLEAGUE_UK.md).

Первинні API джерела: [XInputGetState](https://learn.microsoft.com/en-us/windows/win32/api/xinput/nf-xinput-xinputgetstate),
[XINPUT_STATE](https://learn.microsoft.com/en-us/windows/win32/api/xinput/ns-xinput-xinput_state),
[XINPUT_GAMEPAD](https://learn.microsoft.com/en-us/windows/win32/api/xinput/ns-xinput-xinput_gamepad).
