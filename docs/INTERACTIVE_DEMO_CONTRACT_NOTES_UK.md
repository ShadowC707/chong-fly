# Питання контракту, виявлене під час інтеграції demo

03.11.2026. Ядро та sealed bundle не змінювалися.

04.11.2026: зафіксовано також невідповідність сферичного індикатора контакту
консервативним axis-expanded боксам benchmark detector. Користувацький випадок
має реальний зазор 0,076378 м при collision=box. Повний MRE та результати:
`demo/probe_contact.py`, `output/interactive_demo/contact_reproduction.json`,
секція «Контакт біля кутів» у `docs/INTERACTIVE_DEMO_UK.md`.
Нові live-прогони уточнюють контакт у зовнішньому demo-адаптері
`demo/contacts.py` (sphere-sweep-v1). Ядро benchmark, ваги й контролери
не змінені; старі записи зберігають expanded-primitives-v1.

## Назва/одиниці drag та фактична формула

Після запиту користувача про довге затухання швидкості перевірено
`simulation/drone_env.py`: `DroneDynamicsParams.drag_linear` підписаний як
linear drag coefficient `[N/(m/s)]`, але `QuadcopterDynamics.step` обчислює
`F_drag = -p.drag_linear * self.vel * norm(self.vel)` — квадратичний опір.
Для цієї формули одиниці коефіцієнта мають відповідати N/(m/s)², а не вказаним
у коментарі. З цього не випливає автоматично, що слід змінювати формулу чи
коефіцієнт: калібрування і намір моделі має перевірити основний чат.

Відтворення: `.\venv\Scripts\python.exe -m demo.probe_drift`.
Протокол: W одну simulation second, відпустити, запитати AUTO; далі читати
швидкість і штатні admission reasons. Приблизні спостереження на поточній
машині: t=1 с — 0,650 м/с, t=2 — 0,531, t=6 — 0,154, t=16 — 0,055;
AUTO з’являється між 16 і 17 с. M/запит MANUAL відкликає керування одразу,
незалежно від залишкової швидкості. Enter запитує AUTO, а не MANUAL.

Артефакт: `output/interactive_demo/drift_admission_probe.json`.
Жодного додаткового гальмування, зміни drag чи послаблення admission у demo
не додано. Підпис очікування AUTO тепер явно відрізняється від запиту MANUAL.

## Звичайне відкликання та блокування приймача

У handoff (`docs/INTERACTIVE_DEMO_HANDOFF_UK.md`, розділ 6) сказано:
«Звичайне відкликання без fault і аварійний reset не змішувати».
Сам `FlightBridge` у MANUAL повертає звичайний `manual_override`; його
`fault` залишається `None`. Але `SessionReceiver.set_mode('MANUAL')` після
AUTO викликає `_trip('authority_revoked')` і защіпає `CommandGate.tripped`.
Наступний `set_mode('AUTO')` цього блокування не знімає.

Мінімальний відтворюваний приклад, без GUI, фізики або апаратного зв’язку:

```powershell
.\venv\Scripts\python.exe -m demo.probe_receiver_revocation
```

Очікування від формулювання handoff: звичайне відкликання потрібно відрізняти
від аварійного fault; можливість повторного admission без session reset потребує
явного уточнення. Це не твердження, що приймач обов’язково має дозволяти такий
повторний вхід: конкретна політика відновлення залишається за основним чатом.

Фактичний результат:

```json
{
  "first_accepted": true,
  "after_manual": {"reason": "authority_revoked", "tripped": true, "pwm": null},
  "reentry_accepted": false,
  "reentry_reason": "lease_rejected",
  "reentry_tripped": true
}
```

Код: `edge/session_receiver.py` (`set_mode`, `_trip`),
`edge/flight_bridge.py` (`CommandGate.accept`),
`demo/probe_receiver_revocation.py`.
Локальний артефакт: `output/interactive_demo/receiver_revocation_probe.json`.
Тест інтеграції: `tests/test_interactive_demo.py::test_auto_requires_dwell_acceptance_and_takeover_revokes_old_packets`.

Рішення demo: не скидати gate приховано й не обходити приймач. MANUAL працює
одразу; повторний AUTO після такого відкликання вимагає явної нової сцени / R
з новою сесією. UI називає `authority_revoked` звичайним відкликанням із
блокуванням чинного приймача, а інші причини — аварійним блокуванням. Зміни
математики, API приймача або політики повторного входу тут не внесено.
