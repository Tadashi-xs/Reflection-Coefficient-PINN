# PhysicsInformedNN

## Расположение в проекте

Ноутбук сам определяет корень проекта:

- если запуск идет из папки `PINN`, то корнем считается родительская директория;
- если запуск идет из корня проекта, пути также остаются рабочими.

Ожидаемая структура проекта:

```text
.
├── generate_data.py
├── requirements.txt
├── storage/
│   └── dataset_10k.csv
├── visualization/
└── PINN/
    ├── PhysicsInformedNN.ipynb
    └── saved_models/
```

## Входной датасет

По умолчанию ноутбук читает файл:

```text
storage/dataset_10k.csv
```

Путь задается в `Config.data_path`.

Минимально необходимые колонки:

| Колонка | Смысл |
|---|---|
| `kappa` | волновое число; единственный входной признак модели |
| `R` | коэффициент отражения; должен лежать в диапазоне `[0, 1]` |

## Конфигурация

Основные настройки находятся в классе `Config`.

### Пути

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `notebook_dir` | `Path.cwd()` | текущая рабочая директория ноутбука |
| `project_root` | определяется автоматически | корень проекта |
| `data_path` | `project_root / "storage" / "dataset_10k.csv"` | путь к датасету |
| `saved_models_dir` | `PINN/saved_models` | папка для сохранения моделей |
| `visualization_dir` | `project_root / "visualization"` | папка для сохранения графиков и таблиц |

### Загрузка и сохранение модели

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `load_existing_model` | `False` | загружать ли существующий checkpoint |
| `model_checkpoint_path` | `None` | путь к checkpoint; нужен при `load_existing_model=True` |
| `save_model_after_training` | `True` | сохранять ли модель после обучения |
| `model_run_id` | `None` | номер запуска; если `None`, выбирается следующий свободный номер |

Если `model_run_id=None`, ноутбук смотрит уже существующие модели и папки визуализаций с таким же базовым именем и выбирает следующий свободный номер запуска.

### Разбиение данных и шум

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `add_noise` | `False` | добавлять ли шум в обучающие значения `R` |
| `noise_level` | `0.0` | относительный уровень гауссовского шума |
| `noise_seed` | `42` | seed для генерации шума |
| `valid_size` | `0.15` | доля validation-выборки |
| `test_size` | `0.15` | доля test-выборки |
| `n_strat_bins` | `10` | число бинов для стратификации по `log1p(R)` |

Шум добавляется только к train-части. Validation и test остаются построенными по исходным значениям `R`.

## DataLoader и weighted sampler

Если `use_weighted_sampler=True`, train-выборка подается через `WeightedRandomSampler`. Фактическая формула весов в ноутбуке:

```text
weight = 1
       + sampler_high_R_scale * R
       + 2 * sampler_high_R_scale * I(R > 0.01)
       + 4 * sampler_high_R_scale * I(R > 0.05)
```

Затем веса нормируются на среднее значение.

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `batch_size` | `512` | размер батча |
| `use_weighted_sampler` | `True` | использовать ли weighted sampler |
| `sampler_high_R_scale` | `4.0` | коэффициент усиления веса точек с большим `R` |

Если `use_weighted_sampler=False`, используется обычное перемешивание train-выборки.

## Архитектура модели

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `hidden_dim` | `64` | ширина скрытых слоев |
| `num_hidden_layers` | `3` | число скрытых слоев |
| `dropout` | `0.0` | dropout после скрытых слоев |

## Loss-функция

Loss строится как комбинация трех ошибок:

```text
loss = R_loss_log_weight * SmoothL1(pred_logit_scaled, target_logit_scaled)
     + SmoothL1(R_pred, R_true)
     + R_loss_sqrt_weight * SmoothL1(sqrt(R_pred), sqrt(R_true))
```

Где:

- `pred_logit_scaled` — предсказанный нормализованный `logit_R`;
- `target_logit_scaled` — истинный нормализованный `logit_R`;
- `R_pred` — восстановленный коэффициент отражения после обратного scaling и `sigmoid`;
- `R_true` — истинный коэффициент отражения.

Параметры loss:

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `R_loss_eps` | `1.0e-6` | стабилизатор для `logit_R` и `sqrt(R)` |
| `R_loss_log_weight` | `0.85` | вес ошибки в пространстве `logit_R` |
| `R_loss_sqrt_weight` | `0.15` | вес ошибки по `sqrt(R)` |

Компонента `SmoothL1(R_pred, R_true)` имеет фиксированный вес `1.0`.

## Обучение AdamW

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `max_epochs` | `2000` | максимум эпох AdamW |
| `learning_rate` | `5.0e-4` | learning rate |
| `weight_decay` | `1.0e-6` | L2-регуляризация |
| `grad_clip_norm` | `5.0` | ограничение нормы градиента |
| `scheduler_factor` | `0.5` | множитель уменьшения learning rate |
| `scheduler_patience` | `70` | patience для `ReduceLROnPlateau` |
| `patience` | `200` | patience для early stopping |
| `min_delta` | `1.0e-8` | минимальное улучшение validation loss |

## Обучение L-BFGS

| Параметр | Значение по умолчанию | Смысл |
|---|---:|---|
| `use_lbfgs` | `True` | запускать ли L-BFGS после AdamW |
| `lbfgs_lr` | `1.0` | learning rate L-BFGS |
| `lbfgs_max_iter` | `5000` | максимум итераций |
| `lbfgs_max_eval` | `7500` | максимум вычислений функции |
| `lbfgs_history_size` | `100` | размер истории |
| `lbfgs_tolerance_grad` | `1.0e-9` | tolerance по градиенту |
| `lbfgs_tolerance_change` | `1.0e-11` | tolerance по изменению loss |
| `lbfgs_log_every` | `10` | частота логирования train loss |
| `lbfgs_valid_log_every` | `25` | частота проверки validation loss |

## Название модели

Базовое имя модели строится так:

```text
pinn_{sampler_tag}_N{N}_h{hidden_dim}x{num_hidden_layers}_bs{batch_size}_ep{max_epochs}_lr{learning_rate}_m{run_id}.pt
```

Пример при настройках по умолчанию и датасете на 10 000 строк:

```text
pinn_ws4_N10k_h64x3_bs512_ep2000_lr0p0005_m01.pt
```

Если weighted sampler выключен:

```text
pinn_plain_N10k_h64x3_bs512_ep2000_lr0p0005_m01.pt
```

| Фрагмент | Значение |
|---|---|
| `ws4` | weighted sampler включен, `sampler_high_R_scale=4.0` |
| `plain` | weighted sampler выключен |
| `N10k` | число строк после очистки датасета |
| `h64x3` | 3 скрытых слоя по 64 нейрона |
| `bs512` | batch size 512 |
| `ep2000` | максимум 2000 эпох AdamW |
| `lr0p0005` | learning rate `0.0005` |
| `m01` | номер запуска |

## Сохранение модели

Модели сохраняются в папку:

```text
PINN/saved_models/noise_{noise_percent}/
```

Пример:

```text
PINN/saved_models/noise_0/pinn_ws4_N10k_h64x3_bs512_ep2000_lr0p0005_m01.pt
```

Если запустить обучение с теми же параметрами еще раз, следующий файл получит номер `m02`, затем `m03` и так далее.

Если модель была загружена через `load_existing_model=True`, ноутбук не перезаписывает модель.

## Сохранение визуализаций

Итоговые файлы сохраняются в:

```text
visualization/noise_{noise_percent}/{N}k/{run_id}/
```

Пример:

```text
visualization/noise_0/10k/01/
```

Основные сохраняемые файлы:

| Файл | Смысл |
|---|---|
| `metrics.txt` | таблица метрик в текстовом виде |
| `metrics.png` | таблица метрик как изображение |
| `prediction_quality.png` | качество предсказаний на test |
| `relative_error.png` | анализ относительной ошибки |
| `spectrum.png` | сравнение истинного и предсказанного спектра `R(kappa)` |
| `error_spectrum.png` | абсолютная и относительная ошибка вдоль спектра |

## Как запустить

1. Установить зависимости из корня проекта:

```bash
pip install -r requirements.txt
```

2. Сгенерировать датасет, например:

```bash
python generate_data.py --num_samples 10000
```

3. Открыть ноутбук:

```text
PINN/PhysicsInformedNN.ipynb
```

4. При необходимости поменять путь к датасету в `Config`:

```python
data_path = project_root / "storage" / "dataset_10k.csv"
```

5. Запустить все ячейки ноутбука.

После выполнения checkpoint будет сохранен в `PINN/saved_models/noise_{noise_percent}/`, а итоговые графики и таблицы — в `visualization/noise_{noise_percent}/{N}k/{run_id}/`.
