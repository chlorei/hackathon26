# dreamkit — библиотека плагинов для Dream-1

> Спецификация проекта для хакатона (Berlin Robotics & AI Hackathon, трек Dream Labs «Build a Tool for Dream-1 Users»).
> Дедлайн: воскресенье 22.11, 19:00. Демо: 3 минуты вживую + 2 минуты вопросов.
> Этот файл — контекст для Claude Code. Читай его целиком перед любой задачей.

---

## 1. Идея

**Проблема.** Каждый, кто управляет роботом через Dream-1, сам пишет вокруг модели одно и то же: проверку камер,
защиту от резких движений, повторы при неудаче, логи, проверку «выполнена ли задача». Каждый раз заново, без
стандарта. А когда попытка провалилась, непонятно почему: в лучшем случае смотрят видео глазами.

**Решение.** `dreamkit` — прослойка вокруг цикла управления Dream-1, в которую втыкаются плагины. Каждый плагин
делает одну вещь. Плагины включаются строчкой в конфиге, пишутся за 10 минут и ставятся из GitHub одной командой.
Сверху — веб-панель: живые камеры, лента событий плагинов, история запусков с видео провалов.

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
┌─────────────────────────── компьютер робота / ноутбук ───────────────────────────┐
│                                                                                  │
│  dreamkit ui  ─►  FastAPI (:8800)  ── раздаёт собранный React из dreamkit/ui_dist │
│                      │  REST + WebSocket                                         │
│                      ▼                                                           │
│              Runner (отдельный процесс)                                          │
│                      │  JSON-события построчно в stdout                          │
│                      ▼                                                           │
│   YamEnv (headless) ◄─► цикл управления ◄─► dream.DreamClient ◄─► Dream API       │
│                      │                                                           │
│                 PluginManager: хуки → плагины                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
GitHub: код, плагины в отдельных репо, registry.yaml (каталог), GitHub Pages (опционально)
```

**Почему Runner — отдельный процесс:** MuJoCo/OpenGL и веб-сервер в одном процессе конфликтуют (особенно на macOS).
Сервер запускает `python -m dreamkit.runner --config ... --json-events` и читает события из stdout. Стоп = SIGINT.

---

## 4. Структура репозитория

```
dreamkit/
├── SPEC.md                      ← этот файл
├── pyproject.toml               ← пакет dreamkit, entry point CLI `dreamkit`
├── registry.yaml                ← каталог плагинов
├── dreamkit.example.yaml        ← пример конфига
├── dreamkit/
│   ├── __init__.py              ← экспорт Plugin, Context
│   ├── plugin.py                ← базовый класс Plugin, Context, исключения StopRun
│   ├── manager.py               ← загрузка плагинов (entry points + локальная папка), вызов хуков
│   ├── config.py                ← чтение dreamkit.yaml (pydantic)
│   ├── runner.py                ← цикл управления с хуками (адаптация dream_control.main)
│   ├── scene.py                 ← рандомизация сцены по seed
│   ├── events.py                ← эмиттер событий (stdout JSON lines + в память)
│   ├── storage.py               ← папка запуска: runs/<id>/{run.json, events.jsonl, <cam>.mp4}
│   ├── cli.py                   ← dreamkit run | ui | list | search | install | new
│   ├── server.py                ← FastAPI: REST + WS, раздача ui_dist и media
│   ├── registry.py              ← чтение registry.yaml с GitHub, установка через pip
│   ├── templates/plugin/        ← шаблон для `dreamkit new`
│   └── plugins/                 ← встроенные плагины
│       ├── success_check.py
│       ├── failure_monitor.py
│       ├── camera_check.py
│       ├── retry.py
│       ├── latency_monitor.py
│       ├── recorder.py
│       ├── safety.py
│       └── report.py
├── ui/                          ← React (Vite + TS + Tailwind + shadcn/ui)
└── tests/
```

Симулятор Dream Labs подключаем как зависимость: склонировать их репо рядом и добавить
`dream_library/examples/yam_mujoco` в `sys.path` (путь задаётся в конфиге `sim_path`).

---

## 5. API плагинов

### Базовый класс

```python
class Plugin:
    name: str = "base"
    def __init__(self, **params): self.params = params

    # все хуки необязательны
    def on_run_start(self, ctx): ...        # начало запуска (серия попыток)
    def on_episode_start(self, ctx): ...    # начало попытки, сцена уже расставлена
    def before_model(self, ctx): ...        # перед get_action(); можно ctx.stop() / править ctx.task
    def after_model(self, ctx): ...         # ответ модели в ctx.actions; можно модифицировать
    def on_step(self, ctx): ...             # после каждого env.step (20 Гц)
    def on_episode_end(self, ctx): ...      # конец попытки; ctx.result заполняется плагинами
    def on_run_end(self, ctx): ...          # конец запуска
```

### Context (`ctx`)

| Поле | Тип | Что это |
|---|---|---|
| `env` | `YamEnv` | симулятор (доступ к `env.data`, `env.state()`) |
| `task` | `str` | текущий промпт (плагины могут менять) |
| `model` | `str` | id модели |
| `run_id`, `episode`, `step` | `str`, `int`, `int` | номер запуска / попытки / шага |
| `t` | `float` | симулированное время попытки, с (`step * 0.05`) |
| `frames` | `dict[str, np.ndarray]` | последние RGB-кадры камер `top/left/right` |
| `state` | `dict` | `env.state()` на этом шаге |
| `actions` | `tuple[np.ndarray, np.ndarray] \| None` | (left_rows, right_rows) последнего ответа |
| `timing` | `dict \| None` | `client.last_timing` |
| `result` | `dict` | итог попытки: `{"success": bool, "reason": str, "at": float, ...}` |
| `store` | `dict` | общее хранилище, плагины делятся данными |
| `emit(level, message, **data)` | метод | событие в ленту (`info` / `warn` / `error` / `success`) |
| `stop(reason)` | метод | остановить попытку (бросает `StopEpisode`) |
| `request_retry()` | метод | попросить повтор попытки |

Порядок вызова = порядок плагинов в конфиге. Исключение в плагине не валит запуск: ловим, пишем событие `error`
от имени плагина, продолжаем.

### Загрузка

```python
from importlib.metadata import entry_points
eps = entry_points(group="dreamkit.plugins")   # плагины из pip-пакетов
```
+ все `*.py` из `~/.dreamkit/plugins/` и `./plugins/` (подклассы `Plugin`) + встроенные из `dreamkit/plugins/`.

Внешний плагин регистрируется в своём `pyproject.toml`:
```toml
[project.entry-points."dreamkit.plugins"]
gripper-check = "gripper_check:GripperCheck"
```

---

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
| `dropped` | объект поднимали > 0.03, потом он лёг на стол вне цели |
| `missed_target` | объект отпущен, лежит в 0.05–0.15 от цели |
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

## 8. Runner (цикл управления)

Адаптировать `dream_control.main` (не переписывать логику модели, импортировать их функции):

```
load config → PluginManager(plugins) → hooks.on_run_start
for episode in range(episodes):
    env.reset(); scene.randomize(env, seed + episode)
    hooks.on_episode_start
    заполнить окна кадров (как в main) → open_policy → feed(every=0)
    while t < seconds and not stopped:
        hooks.before_model
        reply = client.get_action()      # физика стоит, пока ждём
        ctx.timing = client.last_timing
        left, right = validate_chunk(env, reply, actions_per_chunk, verbose=False)
        ctx.actions = (left, right); hooks.after_model
        for l, r in zip(*ctx.actions):
            env.step(l, r); capture/feed как в main; hooks.on_step
    hooks.on_episode_end   → при request_retry() повторить эпизод
    client.reset_episode()
hooks.on_run_end
```

- Режим `--headless` всегда в Runner; кадры для панели — `ctx.frames`, отправка ~5 fps.
- Режим `--mock`: вместо модели скриптовая политика из `mock_pickups.py` (или `check_loop`) — для разработки без кредитов.
- Рандомизация сцены: двигать `qpos` свободных суставов фруктов (`model.joint("apple_free").qposadr`), потом
  `mj_forward` и 20 шагов «успокоения» физики.
- Событие = одна JSON-строка в stdout: `{"type":"event","ts":...,"plugin":"failure-monitor","level":"error","message":"dropped apple","data":{...}}`.
  Также `{"type":"frame","cam":"top","jpeg":"<base64>"}`, `{"type":"episode_end",...}`, `{"type":"run_end",...}`.

---

## 9. CLI

```bash
dreamkit run [-c dreamkit.yaml] [--mock] [--episodes N]   # запуск в терминале, цветная лента событий (rich)
dreamkit ui [--port 8800]                                  # панель
dreamkit list                                              # установленные плагины
dreamkit search <слово>                                    # поиск в registry.yaml
dreamkit install <name|github-url>                         # pip install git+...
dreamkit new <name>                                        # шаблон плагина
```

`registry.yaml` в корне нашего репо, читается по raw-URL GitHub:
```yaml
- name: gripper-check
  repo: https://github.com/<team>/dreamkit-gripper-check
  description: Ловит захват, закрытый на пустоте
```

---

## 10. Бэкенд (FastAPI)

```
GET  /api/plugins                 установленные + params + enabled
POST /api/plugins/{name}          {enabled, params}
GET  /api/registry                каталог
POST /api/install                 {name}
POST /api/runs                    {task, model, episodes, mock} → {id}; стартует Runner-процесс
POST /api/runs/{id}/stop
GET  /api/runs                    история
GET  /api/runs/{id}               детали, эпизоды, причины, ссылки на видео
WS   /ws/runs/{id}                события + кадры из stdout Runner'а
GET  /media/{run}/{ep}/{cam}.mp4  видео
GET  /                            React из dreamkit/ui_dist
```

Слушать `127.0.0.1` по умолчанию; `--host 0.0.0.0` только явно.

---

## 11. Фронт (React)

Стек: Vite + React + TypeScript, Tailwind + shadcn/ui, lucide-react, Recharts, TanStack Query, framer-motion.
Тёмная тема по умолчанию. Цвета только со смыслом: зелёный ✓, жёлтый ⚠, красный ✗.

Разработка: `npm run dev` с прокси на `:8800`. Сборка: `npm run build` → `dreamkit/ui_dist/`, пакуется в wheel.
С самого начала фронт работает на **моках** по API из раздела 10, потом переключается на реальный бэкенд.

Экраны:
1. **Run** (главный на демо): 3 живые камеры, форма (задача, модель, число попыток, mock), Старт/Стоп,
   **лента событий плагинов** (иконка плагина, время, уровень, текст, анимация появления), прогресс по попыткам.
2. **Plugins**: карточки установленных (Switch + форма параметров), вкладка «Каталог» с кнопкой «Установить».
3. **History**: таблица запусков (✓/✗, причина, длительность), страница запуска: видео, **перемотанное на `at`**,
   графики (успех по задачам, разбивка причин, задержки).

---

## 12. План по времени и правила отсечения

| Когда | Что | Готово, если |
|---|---|---|
| Сб до 17:00 | `plugin.py`, `manager.py`, `config.py`, `runner.py` (+ `--mock`), `success-check` | `dreamkit run --mock` идёт по попыткам и печатает ✓/✗ |
| Сб до 20:00 | `failure-monitor`, `recorder`, `scene.py` | провал → причина + секунда + mp4 |
| Сб до 23:00 | `camera-check`, `retry`, `latency-monitor`, `report`; прогон на реальной модели | HTML-отчёт по 10+ попыткам |
| Параллельно с 15:00 | фронт на моках: Run → History → Plugins | экраны работают на фейковых данных |
| Сб ночь | `server.py`, связка фронт ↔ Runner | живая лента и камеры в браузере |
| Вс до 12:00 | `registry`, `install`, отдельный репо с 1 плагином, `safety` если успеваем | `dreamkit install gripper-check` работает |
| **Вс 12:00** | **заморозка фич** | дальше только баги, демо, репетиции |
| Вс днём | точность failure-monitor на 20 провалах, запасное видео, 3–4 репетиции | |

Правила: если к сб 20:00 ядро + success-check + failure-monitor не работают в терминале — фронт замораживается и все
помогают с ядром. Панель без работающих плагинов проигрывает.

---

## 13. Демо (3:00)

1. 0:00–0:20 — проблема: «каждый пишет проверки и защиту вокруг Dream-1 руками и не знает, почему робот ошибся».
2. 0:20–0:40 — `dreamkit.yaml`: «вот весь конфиг».
3. 0:40–1:30 — панель, Старт: камеры, лента событий; провал → `failure-monitor: dropped apple at 12.4s` → видео на этом моменте.
4. 1:30–1:50 — `retry` → вторая попытка успешна.
5. 1:50–2:10 — `camera-check`: «ломаем» камеру → робот остановлен.
6. 2:10–2:30 — `dreamkit install gripper-check` из GitHub, плагин появился в панели.
7. 2:30–2:45 — отчёт: % успеха, причины, точность монитора «17/20».
8. 2:45–3:00 — что не сработало, что дальше (реальный робот: причины по камере, а не по физике).

Перед демо: всё открыто заранее, модель «прогрета» (вызвать за 10–15 мин), запасное видео записано, интернет с телефона.

---

## 14. Правила для Claude Code

- **Никогда не меняй модель руки и файлы симулятора Dream Labs** (см. раздел 2, «Модель руки менять НЕЛЬЗЯ»).
  Если задача, кажется, требует этого — остановись и спроси.
- Сначала прочитай нужные файлы симулятора Dream Labs, не угадывай их API.
- Не трать кредиты в тестах: по умолчанию `--mock`; реальную модель — только по явной просьбе.
- Каждый шаг должен запускаться и проверяться сразу (команда + ожидаемый вывод).
- Пороги и магические числа — в `params` плагина, не в коде.
- Плагин не должен ронять запуск: все хуки в try/except в `manager.py`.
- Небольшие коммиты после каждого рабочего шага.
