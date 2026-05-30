# PhysicsInformedNN

Ноутбук сам определяет корень проекта. Если он запускается из папки `PINN`, то корнем считается родительская папка. Если ноутбук запускается из корня, пути также остаются рабочими.

## Конфигурация

Основные настройки находятся в `Config`.

### Пути

| Параметр | Смысл |
|---|---|
| `notebook_dir` | текущая рабочая директория ноутбука |
| `project_root` | корень проекта |
| `data_path` | путь к `.csv` датасету |
| `saved_models_dir` | папка для сохранения моделей |
| `visualization_dir` | папка для сохранения графиков и таблиц |

### Загрузка и сохранение модели

| Параметр | Смысл |
|---|---|
| `load_existing_model` | если `True`, модель загружается из чекпоинта |
| `model_checkpoint_path` | путь к чекпоинту; если `None`, используется автоматически построенный путь |
| `save_model_after_training` | сохранять ли модель после обучения |
| `model_run_id` | номер запуска; если `None`, выбирается следующий свободный номер |

Если `model_run_id=None`, ноутбук сам найдет следующий номер модели. 

### Физические параметры

| Параметр | Смысл |
|---|---|
| `plate_rho` | плотность материала пластины |
| `plate_h` | толщина пластины |
| `plate_E` | модуль Юнга |
| `plate_nu` | коэффициент Пуассона |
| `omega0` | собственная частота резонатора |
| `period_a` | период массива резонаторов |
| `resonator_damping_ratio` | демпфирование резонатора |
| `resonator_mu_scale` | масштаб параметра взаимодействия резонатора с пластиной |
| `spectrum_n_terms` | число членов суммы в каждую сторону при построении спектров |

Эти значения должны быть согласованы с параметрами, которые использовались при генерации датасета.

### Разбиение данных и шум

| Параметр | Смысл |
|---|---|
| `add_noise` | добавлять ли шум в обучающие данные |
| `noise_level` | уровень шума |
| `noise_seed` | seed для шума |
| `test_size` | доля тестовой выборки |
| `valid_size` | доля валидационной выборки |
| `n_strat_bins` | число бинов для стратификации по `R` |

Шум учитывается в структуре папок сохранения.

### Обучение

| Параметр | Смысл |
|---|---|
| `batch_size` | размер батча |
| `use_weighted_sampler` | использовать ли weighted sampler |
| `sampler_high_R_scale` | усиление веса объектов с большим `R` |
| `max_epochs` | максимальное число эпох AdamW |
| `learning_rate` | learning rate AdamW |
| `weight_decay` | L2-регуляризация в AdamW |
| `grad_clip_norm` | gradient clipping |
| `scheduler_factor` | множитель уменьшения learning rate |
| `scheduler_patience` | patience scheduler |
| `patience` | patience early stopping |
| `min_delta` | минимальное улучшение для early stopping |

### L-BFGS

| Параметр | Смысл |
|---|---|
| `use_lbfgs` | запускать ли L-BFGS после AdamW |
| `lbfgs_lr` | learning rate L-BFGS |
| `lbfgs_max_iter` | максимум итераций |
| `lbfgs_max_eval` | максимум вычислений функции |
| `lbfgs_history_size` | размер истории L-BFGS |
| `lbfgs_tolerance_grad` | tolerance по градиенту |
| `lbfgs_tolerance_change` | tolerance по изменению loss |
| `lbfgs_log_every` | частота логирования train loss |
| `lbfgs_valid_log_every` | частота логирования validation loss |

### Loss-функция

Общий loss имеет вид:

```text
loss = data_loss + lambda_phys * physics_loss + lambda_R * reflection_loss
```

| Параметр | Смысл |
|---|---|
| `lambda_phys` | вес physics loss |
| `lambda_R` | вес reflection loss |
| `R_loss_eps` | стабилизатор для вычисления ошибки по `R` |
| `R_loss_log_weight` | вес log-компоненты reflection loss |
| `R_loss_sqrt_weight` | вес sqrt-компоненты reflection loss |

### Архитектура

| Параметр | Смысл |
|---|---|
| `hidden_dim` | ширина скрытых слоев |
| `num_hidden_layers` | число скрытых слоев |
| `dropout` | dropout |

### Spectrum augmentation

| Параметр | Смысл |
|---|---|
| `use_spectrum_augmentation` | добавлять ли дополнительные физические точки в train |
| `spectrum_points_per_target_case` | число точек для целевых спектральных кейсов |
| `spectrum_random_cases` | число случайных геометрических кейсов |
| `spectrum_points_per_random_case` | число точек на случайный кейс |
| `spectrum_augmentation_seed` | seed аугментации |

Аугментация применяется только к train-части. Валидация и тест остаются построенными по исходному датасету.

### Визуализация

| Параметр | Смысл |
|---|---|
| `fixed_spectrum_points` | число точек для графиков спектров |
| `reference_psi` | опорный угол падения для спектров и heatmap |
| `near_zero_delta_y` | малый вертикальный сдвиг для почти одной линии резонаторов |
| `geometry_grid_size` | размер сетки для heatmap |
| `geometry_kappa_quantile_low` | нижний квантиль выбора `kappa` для heatmap |
| `geometry_kappa_quantile_high` | верхний квантиль выбора `kappa` для heatmap |
| `plot_sample_size` | число точек для scatter-графиков |
| `relative_error_min_R` | нижний порог `R` для относительной ошибки |

## Название модели

Чекпоинт сохраняется в формате:

```text
pinn_aug_ws1_N10k_h160x4_bs256_ep2000_lr0p0005_phys0p5_refl2_m01.pt
```

| Фрагмент | Значение |
|---|---|
| `aug` / `noaug` | использовалась или не использовалась spectrum augmentation |
| `ws1` | вес weighted sampler; `plain`, если sampler отключен |
| `N10k` | размер исходного датасета |
| `h160x4` | 4 скрытых слоя по 160 нейронов |
| `bs256` | batch size 256 |
| `ep2000` | максимум 2000 эпох AdamW |
| `lr0p0005` | learning rate `0.0005` |
| `phys0p5` | вес physics loss `0.5` |
| `refl2` | вес reflection loss `2.0` |
| `m01` | номер запуска с данным набором параметров |

## Сохранение модели

Модели сохраняются в:

```text
PINN/saved_models/noise_{noise_percent}/
```

Например:

```text
PINN/saved_models/noise_0/pinn_aug_ws1_N10k_h160x4_bs256_ep2000_lr0p0005_phys0p5_refl2_m01.pt
```

Если запустить модель с теми же параметрами еще раз, следующий файл получит номер `m02`, затем `m03` и так далее.

## Сохранение визуализаций

Визуализации сохраняются в корневую папку проекта:

```text
visualization/noise_{noise_percent}/{N}k/{run_id}/
```

Пример:

```text
visualization/noise_0/10k/01/
```