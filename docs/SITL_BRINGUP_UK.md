# Betaflight SITL: запуск і межа готовності

Дата: 2026-11-07. Продовження [PyBullet feasibility spike](READY_SIM_SPIKE_UK.md).

Оновлення: наступний обмежений етап завершено — див.
[Lockstep: 20 прогонів і таблиця газу](SITL_LOCKSTEP_UK.md).
Нижче збережено результати початкового асинхронного стенду без переписування історії.

## Рішення

**Справжній Betaflight SITL зібрано й запущено. Зв'язок із PyBullet працює,
але весь стенд ще не допущений до порівняння нейромереж.**

Перевірено передавання орієнтації, кутової швидкості, RC-команд і виходів
моторів. Один повний стендовий прогін пройшов ARM → ANGLE → моторні проби →
disarm. У замкнених фізичних спробах завершилися 5 із 8 прогонів; інші
зупинилися через запуск/обмін. Два завершені Pitch-прогони також помітно
відрізняються. Це означає, що **відтворюваність ще не доведена**, а не що
«муха стала гіршою»: нейромережа взагалі не завантажувалася.

На цій межі припинено розширення сценаріїв. Наступна задача — перевірка
часової моделі/синхронізації самого SITL та статичного перетворення
RC throttle → виходи моторів → RPM. Підбирати ваги чи PID за цими трасами
зараз було б методологічно неправильно.

## Середовище й артефакти

- Запущено вже встановлений Docker Desktop, Linux engine; нову WSL-дистрибуцію
  не встановлювали. Docker Desktop залишився запущеним, тестових контейнерів
  після завершення немає.
- Betaflight: `79065c96ba0bb5cdc675e67d7093e05dab8b330e`, версія збереженого
  G4 dump — 2025.12.2. Config submodule:
  `e37834762542f9b7087e917431a0141cbb04a4da`.
- Linux binary SHA-256:
  `1a5d7ba34de102b78d5c485ab71576f5ea0131a17b9b8851157dccb579997e22`.
- Image: `chong-fly-sitl:79065c96b-v1`; image ID:
  `sha256:ea9aefdef6f74684f3fa5580ff5b6911a452a48cfd9d26b38ce7a7a1808e5ae0`.
- Ubuntu 24.04, native GCC 13, Python 3.12, Linux PyBullet 3.2.7,
  NumPy 2.2.6. Це окремий Linux-стенд; його версії **не тотожні** попередньому
  Windows PyBullet 3.25. Парне порівняння цих двох версій не виконувалося.
- [Dockerfile](../scripts/sitl/Dockerfile),
  [image metadata](../output/sitl_bringup_v1/image-inspect.json),
  [pip freeze](../output/sitl_bringup_v1/pip-freeze.txt).

Виконання тестів: `--network none`, без published ports, без devices, без
privileged mode. Репозиторій монтувався read-only; запис дозволено лише в
папку результатів. Усі TCP/UDP — localhost **усередині контейнера**.
Віртуальне arming не є arming справжнього G4.

Dockerfile відтворює спосіб збірки, але Ubuntu/APT не заморожені на snapshot;
повторна майбутня збірка може мати інший binary hash. Для точного повторення
наявної збірки використовується зафіксований локальний image ID.
Загальна перевірка Makefile вимагала ARM compiler навіть для SITL; це обійдено
явними make-параметрами native GCC. Вихідний код керування Betaflight не
патчився. Для PyBullet знадобилися Python development headers.

## Код і віртуальна конфігурація

| Файл | Роль |
|---|---|
| [sitl_contract.py](../simulation/sitl_contract.py) | Перетворення FDM, декодування motor packet, MSP checksum і часткових TCP-відповідей |
| [sitl_probe.py](../evaluate/sitl_probe.py) | Власний процес firmware, CLI setup, сенсорні/моторні проби, коротка фізична петля |
| [sitl_virtual_v1.cli](../configs/sitl_virtual_v1.cli) | Окрема конфігурація ANGLE/AETR/rates/mixer; **не файл відновлення G4** |
| [report_sitl_bringup.py](../evaluate/report_sitl_bringup.py) | Таблиця й графіки з усіма спробами, включно з невдалими |
| [test_sitl_contract.py](../tests/test_sitl_contract.py) | П'ять тестів пакетів, одиниць, осей і MSP |

Разом із попередніми тестами адаптера: **10 passed**. Повну suite проєкту на
цьому етапі не запускали; активний runtime/demo/мережа не змінювалися.

Віртуальний профіль використовує AETR, ANGLE limit 60°, ACTUAL rates 7/67
для кожної осі, zero expo, `yaw_motors_reversed=ON`. Ці значення взяті зі
збереженого dump. Водночас це **не точна копія всього G4**: board alignment
задано нульовим, бо сенсори вже в body frame; PWM використано замість DSHOT;
PID лишено default SITL, повний апаратний dump не імпортувався.
`AltitudeHold`, мережа та координатор відсутні.

## Що встановлено про контракт

1. У pinned SITL `USE_IMU_CALC` вимкнено. Орієнтація передається прямо;
   AHRS цим не перевіряється. У `imu.c` є спеціальні SITL-зміни знаків двох
   елементів rotation matrix. Наш адаптер враховує саме цю версію.
2. Для фізичної FLU-орієнтації `(roll,pitch,yaw)` MSP повертає
   `(roll,pitch,-yaw)` з yaw у `[0,360)`. Усі 5 проб, включно з комбінованою
   `(10°,20°,-30°)`, відповіли очікуванням.
3. Кутова швидкість у внутрішньому контурі узгоджена з фізичною FLU-системою.
   Не можна автоматично застосувати знак MSP yaw-angle до gyro Z: Betaflight
   сам інвертує RC-команду Yaw. FDM gyro відправляється як `(wx,-wy,-wz)`;
   `sitl.c` повертає потрібні знаки перед virtualGyro.
4. `MSP_RAW_IMU` повертає ADC-scaled gyro, попри назву `gyroRateDps` у коді.
   Для цієї збірки потрібно помножити його на `2000/32768`. Вхід 20°/с
   дав 327 ADC, тобто 19.9585°/с. Усі 4 проби після правильного декодування
   пройшли. Це була помилка трактування телеметрії в першій версії probe,
   не встановлений дефект гіроскопа прошивки.
5. `sitl.c` переставляє виходи моторів для Gazebo. Decoder відновлює native
   QUADX RR/FR/RL/FL. Пакет містить `(motorPWM-idlePulse)/1000`, а **не тягу
   і не RPM**. Подальша лінійна відповідність цього числа RPM — явно
   некаліброване припущення нашого smoke-стенду.
6. Напрямки моментів у `attempt_07`: Roll `+0.051918`, Pitch `+0.050800`,
   Yaw `−0.088914` за відповідних додатних RC-імпульсів. Це безрозмірні
   проксі з квадратів моторних виходів, не моменти в Н·м. Перед кожною
   пробою виконано disarm/arm для скидання накопичення I-term.

Підтвердження: [attempt_07/report.json](../output/sitl_bringup_v1/attempt_07/report.json).
У пізнішому `contract_final` сенсорні 5+4 проби також пройшли, але повний
цикл **не пройшов**: перед arming був RX_FAILSAFE, потім залишився
ARM_SWITCH latch. Назва папки не означає успіх. Цей результат збережено;
надалі runner відмовляється подавати ARM, якщо pre-arm статус ще не чистий.

## Фізичні спроби: не приховуємо невдачі

Дві серії, ті самі чотири сценарії. Ціль — 4 с фізики на сценарій.
Фізика 250 Гц, RC 50 Гц. У Roll/Pitch/Yaw +50 PWM від 1.0 до 1.5 с,
потім нейтраль. Throttle = 1213.415751 PWM, успадкований із попередньої
математичної моделі. Мотори до початку стоять на її hover RPM; під час
короткої початкової фази модель нерухома, після відпускання рух обчислює Bullet.
Це не зліт із підлоги і не експеримент утримання висоти.

| Сценарій | Серія 1 | Серія 2 |
|---|---|---|
| hover | MSP timeout до відпускання | 4 с завершено, z: 1 → 2.223 м |
| roll | Перервано на 1.396 с через motor timeout | 4 с завершено, max roll 1.453° |
| pitch | 4 с завершено, max pitch 1.402°, z до 2.240 м | 4 с завершено, max pitch 1.837°, z до 2.684 м |
| yaw | 4 с завершено, max поворот 7.118°, фінальний −6.913° | UART/MSP readiness не досягнуто, фізика не стартувала |

**5/8 completed — це кількість завершених запусків, не success rate
ухилення і не показник якості моделі.** Сцен із перешкодами тут немає.

Між серіями змінено лише startup handshake: готовність підтверджується
відповіддю MSP API, а не самим відкритим TCP-портом. Параметри фізики, PID,
ваги й RC-послідовності не підбиралися. Виправлення запуску не забезпечило
повної стабільності, тому робити загальний висновок «проблему закрито» не можна.

![Усі фізичні спроби](../output/sitl_bringup_v1/comparison.png)

[summary.json](../output/sitl_bringup_v1/summary.json) зберігає також невдалі
спроби. Для кожного сценарію є `trace.npz`, `report.json`, firmware log.
`integration_admitted=false` — навмисний результат цього етапу.

## Що лишилося невизначеним

### Час і транспорт

Upstream SITL працює у кількох потоках; `simRate` визначається через різницю
модельного та реального часу. Наша петля надсилає FDM і чекає motor packet,
але це **ще не доведений deterministic lockstep**. Зокрема, motor packet
не має номера кадру, є scheduler jitter, а atomic відповідність кроків
firmware/physics не підтверджена.

В успішних пробах median локального обміну близько 0.3–0.4 мс, але максимуми
досягали 20+ мс. Ліміт очікування motor packet — 30 мс; у фізичній фазі його
перевищення зупиняє прогін, без продовження на старій команді. Ці виміри
стосуються Docker loopback, **не Wi-Fi ESP32/Pi і не hardware latency budget**.

У першій спробі FDM потрапляв під час першої ініціалізації EEPROM і firmware
завершився. Джерело можливого startup crash видно в коді: virtual sensor
pointers ще можуть бути не ініціалізованими. Sender тепер чекає handshake.
Це не патч firmware, а порядок запуску адаптера.

Окремий залишковий ризик — readiness/відновлення RX після virtual acc calibration.
Захист firmware не обходили. Поточний runner завершує тест, якщо readiness
не досягнуто в обмежений час; у стабільному runner потрібні точний журнал
станів і підтверджений протокол відновлення, а не нескінченні спроби ARM.

### Throttle та фізична відповідність

Попереднє hover PWM було правильним для нашого прямого відображення
RC → normalized RPM. Betaflight має своє перетворення RC, idle/endpoints,
мікшер та фільтри. Тому той самий RC throttle не зобов'язаний створювати
ту саму сумарну тягу. Уже в нерухомій пробі 1250 RC дало близько 0.257
normalized motor output. Підставляти стару hover-константу без перевірки
навіть у готовий рушій неправильно.

Сильний набір висоти тут не є доведеною проблемою нейромережі або AltitudeHold:
обидва відсутні. Статичну motor/RC map треба встановити окремо; зміна висоти
при зміні мікшера не повинна маскуватися підбором поведінкових порогів.

## Наступний обмежений крок

1. Визначити й перевірити єдиний час/крок firmware та physics, прибрати
   неоднозначність startup, RX recovery і motor acknowledgement. Якщо потрібен
   патч upstream synchronization, зберігати його окремо з поясненням, без
   зміни PID/ANGLE математики.
2. На нерухомій моделі виміряти кілька RC throttle точок і motor outputs,
   зафіксувати явну actuator map. Це симуляційна ідентифікація, не калібрування
   справжніх моторів.
3. Повторити **ті самі** короткі маневри. Умова переходу: усі запуски завершені,
   знаки правильні, немає пропусків/невідомої прив'язки кадрів; повторюваність
   виміряна. Лише після цього додавати сенсори, незмінну мережу й кімнати.

Не потрібно зараз більше кімнат, RL чи донавчання конектоми.

## Команди повторення

З кореня проєкту, для вже створеного image:

```powershell
$repoPath=(Get-Location).Path
$resultsPath=(Resolve-Path output/sitl_bringup_v1).Path
docker run --rm --network none --cpus 4 --memory 2g --mount "type=bind,source=$repoPath,target=/work,readonly" --mount "type=bind,source=$resultsPath,target=/results" chong-fly-sitl:79065c96b-v1 python -u -m evaluate.sitl_probe --out /results/new_contract_run
```

Потрібна **нова** output-папка, щоб не підхопити старий EEPROM. Фізичний
дослід запускається окремо лише після перегляду й проходження contract probes:

```powershell
docker run --rm --network none --cpus 4 --memory 2g --mount "type=bind,source=$repoPath,target=/work,readonly" --mount "type=bind,source=$resultsPath,target=/results" chong-fly-sitl:79065c96b-v1 python -u -m evaluate.sitl_probe --physics-from /results/new_contract_run --out /results/new_physics_run
.\venv\Scripts\python.exe -m pytest tests/test_sitl_contract.py tests/test_ready_sim_contract.py -q -p no:cacheprovider --basetemp .test-tmp-sitl
```

Якщо readiness не пройдена, не вимикати failsafe й не переносити ці команди
на справжню плату: це матеріал для наступного виправлення тестового стенду.

Збірка (потребує pinned checkout `scratch/ready_sim/betaflight` та upstream
`scratch/ready_sim/gym-pybullet-drones`, описаного у попередньому звіті):

```powershell
docker build --progress plain -f scripts/sitl/Dockerfile -t chong-fly-sitl:79065c96b-v1 scratch/ready_sim
```

Основні перевірені джерела:
[SITL implementation](https://github.com/betaflight/betaflight/blob/79065c96ba0bb5cdc675e67d7093e05dab8b330e/src/platform/SIMULATOR/sitl.c),
[IMU/SITL signs](https://github.com/betaflight/betaflight/blob/79065c96ba0bb5cdc675e67d7093e05dab8b330e/src/main/flight/imu.c),
[virtual gyro](https://github.com/betaflight/betaflight/blob/79065c96ba0bb5cdc675e67d7093e05dab8b330e/src/main/drivers/accgyro/accgyro_virtual.c),
[MSP gyro scaling](https://github.com/betaflight/betaflight/blob/79065c96ba0bb5cdc675e67d7093e05dab8b330e/src/main/sensors/gyro_init.c).
