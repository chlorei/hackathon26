# dreamkit — библиотека плагинов для Dream-1

> Спецификация проекта для хакатона (Berlin Robotics & AI Hackathon, трек Dream Labs «Build a Tool for Dream-1 Users»).
> Дедлайн: воскресенье 22.11, 19:00. Демо: 3 минуты вживую + 2 минуты вопросов.
> Этот файл — контекст для Claude Code. Читай его целиком перед любой задачей.

---

## 1. Идея

**Проблема.** Каждый, кто управляет роботом через Dream-1, сам пишет вокруг модели одно и то же: проверку камер,
защиту от резких движений, повторы при неудаче, логи, проверку «выполнена ли задача». Каждый раз заново, без
стандарта. А когда попытка провалилась, непонятно почему: в лучшем случае смотрят видео глазами.

**Решение.** `dreamkit` — **библиотека** плагинов для Dream-1 с монитором. Это не приложение, которое запускает
задачи: пользователь встраивает её в **свой** цикл управления одной строкой —
`client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")` — и дальше пишет обычный код
(`client.policy(...)`, `set_observation(...)`, `get_action()`). Плагины срабатывают внутри этих вызовов. Каждый плагин
делает одну вещь, включается строчкой в конфиге, пишется за 10 минут и ставится из GitHub одной командой.
Веб-панель **только управляет плагинами и показывает монитор**: живой запуск (камеры, лента событий, таймлайн
попытки) и историю с видео провалов. Промпты и запуск задач через панель — нет: задачи ставит код пользователя.

**Главная фишка (на неё 80% времени демо):** плагин `failure-monitor` — говорит не просто ❌, а **почему** попытка
провалилась и **в какой момент** («уронил яблоко на 12.4 с»), с видео, перемотанным на этот момент.

**Для кого:** инженеры робототехнических стартапов, исследователи, ML-инженеры, энтузиасты с YAM / SO-101.
Запускают из терминала (часто по SSH на компьютер робота), результат смотрят в браузере.

**Критерии судей:** польза 25, надёжность 25, сложность 20, демо 20 (видео вместо живого демо — максимум 3/5),
подача 10 (проблема, результат, что не сработало, что дальше). Оригинальности среди критериев нет — выигрывает то,
что полезно и стабильно работает.

---

## 2. Исходные данные (что уже есть у Dream Labs)

Репозиторий: `https://github.com/dreamlabs-co/dreamlabs` (Apache 2.0). Установка: `pip install dreamlabs`
(Python 3.11–3.13). Ключ: `~/.dreamlabs/credentials` или `DREAM_API_KEY`.

Симулятор: `dream_library/examples/yam_mujoco/` (запуск `dreamcli example yam-mujoco`). Ключевые файлы:

- `yam_env.py` → класс `YamEnv`:
  - `env.reset()` — сброс сцены;
  - `env.state()` → `{"left": np.array(7), "right": np.array(7)}` (6 углов суставов в радианах + захват 0..1);
  - `env.step(left, right)` — один шаг 50 мс (20 Гц), физика 500 Гц; действие = 7 чисел на руку;
  - `env.images()` → `{"top", "left", "right"}`, RGB uint8 360×640;
  - `env.model`, `env.data` — MuJoCo. Позиции тел: `env.data.body("apple").xpos`.
    Тела: `apple`, `orange`, `banana`, `water_bottle`, `bowl`, `plate`. Свободные суставы фруктов: `apple_free` и т.д.
    Кончик захвата: `env.data.site("left_tcp")`, `env.data.site("right_tcp")`.
  - Стартовые позиции: apple (0.15, 0.105, 0.792), orange (0, 0.13, 0.794), banana (−0.115, 0.085, 0.79),
    bowl (0, −0.04, 0.76), plate (−0.22, −0.045, 0.764). Мир Z-вверх, метры.
- `dream_control.py` → цикл управления с моделью. Нужные куски:
  - `open_policy(policy_config(model, gateway, poses), key)` → открытый `dream.DreamClient`;
  - `capture(env, windows, keep)` — рендер камер, JPEG 320×176 в окна по 9 кадров;
  - `feed(...)` — `set_observation` в клиент;
  - `validate_chunk(env, reply, actions)` → `(left_rows, right_rows)` — проверенные действия;
  - `main()` — весь цикл: пока ждём ответ модели, физика стоит; потом выполняем до 48 строк действий по 50 мс.
  - Ключи камер для модели: top→`video.overhead`, left→`video.wrist_l`, right→`video.wrist_r`.
  - Модель по умолчанию: `dreamlabs/dream-1-beta`.
- `mock_pickups.py` — скриптовый захват фрукта без модели (IK), пригодится для тестов без кредитов.
- `python -m dream.check_loop` — заглушка вместо модели (бесплатная отладка цикла).
- `client.last_timing` — задержки последнего запроса (`rtt_ms`, `serve_infer_ms`, `network_ms`, …).

**Важные ограничения:**
- Успех задачи в симуляторе **не определяется** — это делаем мы.
- Модель в симуляторе часто ошибается (по их README: яблоко и апельсин в миску удались по разу, банан нестабилен).
- Каждый вызов модели тратит кредиты ($150 на человека). Отлаживать на скриптах и `check_loop`.
- Лимит: 3 запроса подряд, дальше 1 запрос в 0.7 с на сессию.
- Задержка > ~700 мс ухудшает управление.
- На macOS окно MuJoCo требует `mjpython` (`launch_passive`). Для панели запускаем симулятор **без окна**.

**⛔ Модель руки менять НЕЛЬЗЯ.** Робот в симуляторе должен оставаться таким, каким его сделали Dream Labs:
- не трогать геометрию, меши, суставы, их пределы, актуаторы, параметры захвата (трение, сила, ход пальцев),
  массы и инерции звеньев, положение баз рук;
- не трогать камеры (позиции, углы, разрешение, кроп) — модель обучена именно на этих ракурсах;
- не редактировать файлы симулятора Dream Labs (`yam_env.py`, `dream_control.py`, ассеты). Только импортировать
  и вызывать их функции; нужна другая логика — пишем свою обёртку в `dreamkit/`;
- не обходить `validate_chunk` и пределы суставов: действия модели идут в `env.step` только через их проверку.

Что можно: читать любое состояние (`env.data`, `env.state()`, контакты), двигать **предметы** (фрукты, миску,
тарелку) для рандомизации сцены, рендерить кадры, ограничивать действия **строже** (плагин `safety`), но не мягче.


---

## 3. Архитектура

```
┌──────────────── процесс пользователя (его цикл управления) ────────────────┐
│  env (сим или робот) ⇄ ЦИКЛ ПОЛЬЗОВАТЕЛЯ ⇄ dreamkit.wrap(DreamClient) ⇄ Dream API │
│                                   │  хуки before/after_model в get_action()   │
│                                   │  kit.step(env, ...) → on_step             │
│                                   ▼                                          │
│                     PluginManager → плагины → runs/<id>/                      │
│                     (run.json, events.jsonl, live/*.jpg, live.json, ep<N>/*.mp4)│
└──────────────────────────────────────────────────────────────────────────────┘
                 ▲ читает файлы (никаких сокетов в процесс пользователя)
     dreamkit ui → FastAPI (:8800) → React: Live / Plugins / History
```

**Связь панели и цикла — через папку `runs/<id>/`.** Обёртка пишет события в `events.jsonl`, живые кадры в
`live/<cam>.jpg`, пульс и телеметрию в `live.json`. Сервер следит за этими файлами и раздаёт их по WebSocket.
Плюсы: цикл пользователя не зависит от панели (панель можно открыть/закрыть в любой момент, упавшая панель не роняет
робота), MuJoCo/OpenGL и веб-сервер в разных процессах, работает и по SSH (папка общая).
## 4. Структура репозитория

```
dreamkit/                        (репозиторий github.com/chlorei/hackathon26)
├── SPEC.md
├── pyproject.toml               ← пакет dreamkit, CLI `dreamkit`
├── registry.yaml                ← каталог плагинов
├── dreamkit.example.yaml
├── dreamkit/
│   ├── __init__.py              ← wrap, Kit, Plugin, Context
│   ├── kit.py                   ← обёртка: Kit (wrap), episode(), step(), close()
│   ├── live.py                  ← запись живых кадров/пульса в runs/<id>/live
│   ├── plugin.py, manager.py, config.py, events.py, storage.py
│   ├── sim.py, scene.py, task.py ← помощники для симулятора Dream Labs
│   ├── mock.py                  ← MockClient: скриптовая «модель» с интерфейсом DreamClient (без кредитов)
│   ├── cli.py, server.py, registry.py
│   └── plugins/                 ← встроенные плагины
├── examples/sim_loop.py         ← цикл на симуляторе Dream Labs с dreamkit.wrap (то, что запускаем на демо)
├── plugins-external/gripper-check/  ← пример внешнего плагина (ставится из GitHub)
├── ui/                          ← React (Vite + TS + Tailwind)
└── tests/
```
## 5. API

### Встраивание (главное)

```python
import dream, dreamkit

client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")   # то же API, что у DreamClient
client.policy({"checkpoint": "dreamlabs/dream-1-beta", ...})

with client.episode("put apple into the bowl", env=env):   # одна попытка: on_episode_start / on_episode_end
    while client.t < 60:
        client.set_observation(frames, observation)         # как обычно
        reply = client.get_action()                         # before_model → модель → after_model
        for left, right in rows(reply):
            env.step(left, right)
            client.step(env, images=env.images(), command=(left, right))   # on_step (20 Гц)
    # плагин сказал «стоп» (camera-check) или «готово» (success-check) → выход из with, без исключения наружу

if client.retry_requested: ...                              # retry попросил повтор той же сцены
client.close()                                              # on_run_end (или with dreamkit.wrap(...) as client)
```

- `wrap()` возвращает `Kit`: все атрибуты `DreamClient` проходят насквозь (`__getattr__`), перехвачены только
  `policy`, `set_observation`, `get_action`, `reset_episode`, `close`.
- `get_action()`: `before_model` → `client.get_action()` → `ctx.timing = last_timing` → `ctx.actions` (из
  `q_l_chunk`/`q_r_chunk`) → `after_model` → изменённые плагинами строки записываются обратно в ответ. Код
  пользователя всё равно прогоняет ответ через свою проверку (`validate_chunk`) — плагины только ужесточают.
- `step(env, images=None, command=None)` — данные симулятора/робота для `on_step`: `ctx.env`, `ctx.state`
  (`env.state()`), кадры, исполняемая строка. Время попытки `ctx.t = step × 0.05 с`.
- `episode(task, env=None, images=None)` — контекст попытки. Без него `get_action()` открывает попытку сам
  (неявно), но тогда стоп плагина прилетает исключением `dreamkit.StopEpisode`.
- Без симулятора (`env` без MuJoCo) плагины физики (failure-monitor, success-check) молча пропускают свою часть;
  camera-check, latency-monitor, recorder, retry, report работают на любом роботе.

### Плагин

Базовый класс и хуки — без изменений (`on_run_start`, `on_episode_start`, `before_model`, `after_model`, `on_step`,
`on_episode_end`, `on_run_end`). Новое: `ctx.telemetry(name=value)` — живой график в панели; атрибут `group`
(`diagnosis` / `safety` / `output`) — блок на странице Plugins.

### Context (`ctx`)

| Поле | Тип | Что это |
|---|---|---|
| `env` | объект | то, что передали в `episode()`/`step()` (для сима — `YamEnv`) |
| `task` | `str` | промпт попытки |
| `model` | `str` | `checkpoint` из `policy()` |
| `run_id`, `episode`, `scene`, `attempt`, `step`, `t` | | запуск / попытка / сцена / номер попытки на сцене / шаг / секунды |
| `frames` | `dict[str, np.ndarray]` | последние RGB-кадры (из `step(images=...)`) |
| `state` | `dict` | `env.state()` |
| `reply` | `dict` | ответ модели целиком |
| `actions` | `(left_rows, right_rows) \| None` | строки ответа для YAM; можно менять в `after_model` |
| `command` | `(left, right) \| None` | строка, исполняемая на этом шаге |
| `timing` | `dict \| None` | `client.last_timing` (`{"mock": True}` у MockClient) |
| `result`, `store` | `dict` | итог попытки / общее хранилище |
| `emit`, `stop`, `end_episode`, `request_retry`, `telemetry` | методы | |

Порядок вызова = порядок в конфиге. Исключение в плагине не валит цикл пользователя: ловим, пишем событие `error`.

### Загрузка плагинов

Встроенные + entry points `dreamkit.plugins` (pip/GitHub) + `*.py` из `~/.dreamkit/plugins/` и `./plugins/`.
## 6. Конфиг `dreamkit.yaml`

```yaml
sim_path: ~/dreamlabs/dream_library/examples/yam_mujoco
model: dreamlabs/dream-1-beta
task: put apple into the bowl
episodes: 5
seconds: 60              # лимит симулированного времени на попытку
actions_per_chunk: 24    # 1..48
seed: 42                 # сцены воспроизводимы
randomize:
  objects: [apple, orange, banana]
  xy_jitter: 0.03        # метры
plugins:
  - camera-check
  - latency-monitor: {warn_ms: 700}
  - success-check
  - failure-monitor
  - retry: {attempts: 2, on: [not_reached, grasped_air]}
  - recorder: {fps: 10}
  - report
```

---

## 7. Встроенные плагины

### success-check (обязательный)
Парсит задачу: объект ∈ {apple, orange, banana}, цель ∈ {bowl, plate}. В `on_episode_end` (и в `on_step` для раннего
завершения, если объект в цели и захват открыт 1 с) проверяет:
горизонтальное расстояние объект–цель < R (bowl 0.07, plate 0.09) и `цель.z − 0.01 < объект.z < цель.z + 0.12`.
Пишет `ctx.result["success"]`. Проверено вручную: яблоко на столе → False, телепорт в миску → True.

### failure-monitor (главная фишка)
В `on_step` пишет историю: расстояние tcp→объект для каждой руки, высота объекта, открытость захватов,
контакты (`env.data.contact[:env.data.ncon]`, имена через `mujoco.mj_id2name`), скорость суставов.
В `on_episode_end`, если не успех, выбирает **первую по времени** причину:

| reason | правило (пороги подбираются, вынести в params) |
|---|---|
| `fell_off_table` | объект.z < стол − 0.05 |
| `not_reached` | min расстояние tcp→объект > 0.03 за всю попытку |
| `grasped_air` | захват закрылся (opening < 0.2) у объекта, но объект не поднялся > 0.02 |
| `dropped` | объект поднимали > 0.03, и он выпал **при закрытом захвате** (выскользнул сам) |
| `released_off_target` | захват **открылся** и отпустил объект **не над целью** (дальше 0.15 от неё) |
| `missed_target` | захват открылся над целью / рядом (≤ 0.15), но объект не в ней |
| `collision` | контакт руки со столом / миской / другой рукой (кроме пальцев с объектом) |
| `stalled` | суставы почти не двигаются > 3 с |
| `slow_model` | средний `rtt_ms` > 700 |
| `timeout` | ничего из выше, время вышло |

Пишет `reason`, `at` (секунда), `details`, эмитит событие. **Метрика качества:** прогнать 20 провалов, сравнить с
видео глазами, записать точность («17/20») — это цифра для судей.

### camera-check
`before_model`: кадр почти чёрный (mean < 10), почти однородный (std < 3) или не менялся N кадров подряд →
`ctx.stop("camera <name> ...")`. Для демо: параметр `simulate_failure: top` затемняет кадр камеры.

### retry
Если `success == False` и `reason` в списке `on` → `ctx.request_retry()`, но не больше `attempts`.
Повтор — та же сцена (тот же seed).

### latency-monitor
`after_model`: читает `ctx.timing`, эмитит `warn`, если `rtt_ms > warn_ms`; в конце — средняя и p95.

### recorder
Пишет MP4 по камерам (`imageio` / PyAV), `fps` параметр, в `runs/<id>/ep<N>/<cam>.mp4`. Нужен для видео провалов.

### safety (если останется время)
`after_model`: ограничивает изменение суставов между строками (макс. скорость) и запрещает tcp выходить из
бокса рабочей зоны (проверка через forward kinematics на копии `MjData`). Нарушение → обрезать строку + `warn`.

### report
`on_run_end`: `runs/<id>/report.html` — % успеха, разбивка причин, задержки, ссылки на видео. Самодостаточный HTML.

---

## 8. Пример цикла: `examples/sim_loop.py` (то, что запускаем на демо)

Цикл `dream_control.main`, переписанный как **код пользователя** с `dreamkit.wrap`: функции Dream Labs
(`policy_config`, `feed`, `capture`, `validate_chunk`) импортируются, файлы симулятора не меняются.

```
kit = dreamkit.wrap(DreamClient | MockClient, config)
kit.policy(dc.policy_config(model, gateway, poses))
for scene in range(episodes):                      # повтор той же сцены, пока retry просит
    env.reset(); positions = dreamkit.scene.randomize(env, objects, jitter, seed + scene); print(positions)
    заполнить окна кадров (как в main)
    with kit.episode(task, env=env, images=...):
        dc.feed(kit, ...every=0)
        while kit.t < seconds:
            reply = kit.get_action()               # физика стоит, пока ждём
            left, right = dc.validate_chunk(env, reply, actions)
            for l, r: env.step(l, r); capture/feed; kit.step(env, images, command=(l, r))
kit.close()
```

- `python examples/sim_loop.py --mock` — MockClient вместо модели (бесплатно). Поведения по попыткам:
  `success`, `dropped` (выбит рывком при закрытом захвате), `released_off_target`, `missed_target`, `not_reached`,
  `grasped_air`, `stalled`, `collision`, `fell_off_table`. Длительности фаз случайны по seed — попытки не одинаковые.
- Позиции предметов печатаются в начале каждой попытки и пишутся в событие `episode_start`.
- По умолчанию в реальном времени (50 мс на шаг), `--fast` — без пауз (тесты).
## 9. CLI

```bash
python examples/sim_loop.py [--mock] [--episodes N] [--task ...]  # цикл на симуляторе (демо)
dreamkit ui [--port 8800]                                  # панель: Live / Plugins / History
dreamkit list | search <слово> | install <name|owner/repo|url|path> | new <name>
dreamkit review | accuracy labels.csv                      # точность failure-monitor
```

Команды `dreamkit run` больше нет: задачи запускает код пользователя.
## 10. Бэкенд (FastAPI)

```
GET    /api/plugins               установленные + params + enabled + group
POST   /api/plugins/{name}        {enabled, params} → пишет dreamkit.yaml (подхватится при следующем wrap())
DELETE /api/plugins/{name}        удалить внешний плагин
GET    /api/registry, POST /api/install
GET    /api/live                  идёт ли запуск сейчас (run.json running + live.json свежее 10 с) → {id} | null
WS     /ws/live/{id}              хвост events.jsonl + кадры live/*.jpg + телеметрия live.json
GET    /api/runs, GET /api/runs/{id}, DELETE /api/runs/{id}
POST   /api/runs/{id}/reveal      открыть папку в Finder (путь в интерфейсе не показываем)
GET    /media/{run}/...           видео, отчёт
```

Сервер никогда не запускает робота и не пишет промпты. Слушать `127.0.0.1`; `--host 0.0.0.0` только явно.
## 11. Фронт (React)

Vite + React + TS + Tailwind, IBM Plex, тёмная графитовая тема; цвет только со смыслом (янтарный = «сейчас»,
зелёный ✓, красный ✗). Кадры и графики рисуются мимо React (дёшево по CPU).

1. **Live** — только смотрит. Идёт запуск → большая камера + две маленькие, таймлайн попытки (растёт), живые графики,
   лента событий плагинов, список попыток. Нет запуска → «Waiting for a run. Start your loop with
   dreamkit.wrap(...)» и пример кода.
2. **Plugins** — три блока: **Diagnosis** (success-check, failure-monitor — выделен как главный, gripper-check),
   **Safety** (camera-check, safety, latency-monitor), **Output** (recorder, recap, report, retry); плюс «Add from
   GitHub» и каталог.
3. **History** — монитор: запуски, попытка с видео на моменте провала, таймлайн, моменты (вердикт сверху,
   остальное по времени), recap, графики измерений; удаление запуска. Полный локальный путь не показываем.
## 12. План (новая концепция) — статус

| # | Что | Проверка |
|---|---|---|
| 1 | `dreamkit.wrap` / `Kit` | `pytest tests/test_kit.py` |
| 2 | `examples/sim_loop.py --mock` | `python examples/sim_loop.py --mock --episodes 2` |
| 3–4 | Live вместо Run | `dreamkit ui` + sim_loop в другом терминале |
| 5 | Plugins по группам | страница Plugins |
| 6 | History + удаление | кнопка Delete |
| 7–12 | исправления (mock latency, рандомизация, dropped vs released_off_target, путь, Moments, «13.2 s») | см. ответ |

Правило: плагины и обёртка важнее панели. Панель без работающей обёртки проигрывает.
## 13. Демо (3:00)

1. 0:00–0:20 — проблема: «каждый пишет проверки и защиту вокруг Dream-1 руками и не знает, почему робот ошибся».
2. 0:20–0:45 — код: `client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")` — одна строка в их цикле;
   `dreamkit.yaml` — весь конфиг.
3. 0:45–1:30 — терминал: `python examples/sim_loop.py`; панель Live подхватывает запуск сама: камеры, лента;
   провал → `failure-monitor: dropped apple at 12.4s` → History, видео на этом моменте, recap.
4. 1:30–1:50 — `retry` → вторая попытка успешна.
5. 1:50–2:10 — `camera-check`: включаем `simulate_failure` в Plugins → следующий запуск остановлен.
6. 2:10–2:30 — `dreamkit install gripper-check` из GitHub, плагин появился в панели.
7. 2:30–2:45 — отчёт: % успеха, причины, точность монитора «N/20».
8. 2:45–3:00 — что не сработало, что дальше.
## 14. Правила для Claude Code

- **Никогда не меняй модель руки и файлы симулятора Dream Labs** (см. раздел 2, «Модель руки менять НЕЛЬЗЯ»).
  Если задача, кажется, требует этого — остановись и спроси.
- Сначала прочитай нужные файлы симулятора Dream Labs, не угадывай их API.
- Не трать кредиты в тестах: по умолчанию `--mock`; реальную модель — только по явной просьбе.
- Каждый шаг должен запускаться и проверяться сразу (команда + ожидаемый вывод).
- Пороги и магические числа — в `params` плагина, не в коде.
- Плагин не должен ронять запуск: все хуки в try/except в `manager.py`.
- Небольшие коммиты после каждого рабочего шага.
