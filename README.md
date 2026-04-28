# oinkVision_belgau

Рабочий репозиторий для multi-label классификации состояния **задних конечностей** свиньи по 4 синхронным видео:

- `top`
- `right`
- `left`
- `rear`

Проект сделан под хакатонный режим: важны не только качество модели, но и понятный, защищаемый, воспроизводимый пайплайн.

## Задача

Для каждой свиньи нужно предсказать 4 бинарных признака:

- `bad_posture`
- `bumps`
- `soft_pastern`
- `x_shape`

Предсказание делается на уровне **животного целиком**, а не отдельного кадра и не отдельной камеры.

Основная метрика: **Macro F1-score**.

Это означает, что:

- каждый класс важен отдельно
- редкие классы сильно влияют на итог
- хороший `accuracy` здесь почти ничего не гарантирует

## Данные

Проект ожидает, что датасет лежит отдельно от кода, например:

```text
/root/data
```

Локально в текущем workspace он лежит здесь:

```text
/Users/banworddd/University/Hakaton/data
```

Ключевые папки:

- `data/train/raw/`
- `data/train/annotation/`
- `data/train/metadata/`
- `data/val/raw/`
- `data/val/metadata/`
- `data/test/`

## Что важно понимать про train / val / test

### `train`

Это основной источник обучения.

В `train/annotation/pig_{id}.json` лежат:

- `pig_id`
- пути к 4 видео
- `target` по задним конечностям
- покадровые bbox-аннотации

Именно `train` даёт нам настоящий supervised-сигнал по целевой задаче.

### `val`

У `val/raw` есть видео нескольких свиней, но открытой rear-label разметки для основной задачи пользователю не выдают.

Важно:

- `val/metadata` не является полноценным rear ground truth для целевой leaderboard-метрики
- hidden score по `val` считают организаторы/менторы на своей стороне

То есть `val` полезен как:

- набор для official hidden evaluation
- внешний sanity-check

Но локально для честного подбора модели основной опорой остаётся **internal split из train**.

### `test`

Финальный набор для отправки сабмита после freeze.

## Реальный размер обучающего сигнала

По объёму файлов датасет кажется большим, но по смыслу для модели он маленький:

- около `100` train-животных
- `4` камеры на каждую свинью
- много кадров, но это всё равно ограниченное число **размеченных животных**

Ключевой дисбаланс классов:

- `bad_posture`: `8` positive
- `bumps`: `84` positive
- `soft_pastern`: `10` positive
- `x_shape`: `1` positive

Это самая важная практическая особенность задачи.

## Текущая основная архитектура

Сейчас основной конфиг проекта:

- [baseline_server_v3.yaml](/Users/banworddd/University/Hakaton/oinkVision_belgau/configs/baseline_server_v3.yaml)

Это лучший стабильный baseline из текущих экспериментов.

### Архитектура модели

1. Для каждой свиньи берутся кадры из 4 камер.
2. Для train/internal valid в базовом режиме используются размеченные `frame_id` из `annotation.json`.
3. При наличии bbox выполняется crop информативной области:
   - `top`: свинья целиком
   - `left/right`: задняя нога
   - `rear`: две задние ноги
4. Каждый кадр кодируется CNN:
   - `efficientnet_b0`
   - `ImageNet pretrained`
5. Для каждого кадра получаем `4 logits`.
6. Затем идёт агрегация:
   - сначала по кадрам
   - затем по всем камерам
7. После `sigmoid` вероятности переводятся в `0/1` через per-class thresholds.

### Архитектура в терминах тензоров

Упрощённо:

1. `images`: `[batch, total_frames, 3, H, W]`
2. reshape в:
   - `[batch * total_frames, 3, H, W]`
3. CNN -> logits:
   - `[batch * total_frames, 4]`
4. reshape обратно:
   - `[batch, total_frames, 4]`
5. frame aggregation -> animal logits:
   - `[batch, 4]`
6. `sigmoid` -> вероятности:
   - `[batch, 4]`

## Полный текущий пайплайн

### 1. Построение индекса

Скрипт:

- [build_index.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/scripts/build_index.py)

Он собирает из `train/annotation/*.json` единый индекс:

- `pig_id`
- пути к `top/right/left/rear`
- target-метки
- статистику по аннотациям
- `annotation_path`

Результат:

- `outputs/index/train_index.csv`

### 2. Internal split

Скрипт:

- [build_internal_split.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/scripts/build_internal_split.py)

Делит `train` на:

- `train_split.csv`
- `valid_split.csv`

Разделение идёт **по животным**, а не по кадрам.

### 3. Dataset

Файл:

- [dataset.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/src/oinkvision/dataset.py)

Режимы:

- `annotated mode`
  - использует `annotation.json`
  - берёт размеченные кадры
  - умеет bbox-crop
- `raw mode`
  - работает без `annotation.json`
  - выбирает кадры из видео через quality-aware sampling
  - нужен для official `val/test`

### 4. Preextract frame cache

Скрипт:

- [preextract_annotated_frames.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/scripts/preextract_annotated_frames.py)

Назначение:

- заранее вытащить нужные кадры из видео
- сохранить их как картинки
- ускорить train/infer

Это не меняет смысл данных, а только ускоряет доступ к ним.

### 5. Обучение

Файл:

- [train.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/src/oinkvision/train.py)

Что делает:

- читает конфиг
- строит `DataLoader`
- использует конфигурируемый loss:
  - `BCEWithLogitsLoss + pos_weight`
  - или `AsymmetricLoss`
- обучает модель
- валидируется на internal valid
- сохраняет лучший чекпойнт:
  - `best_model.pt`
  - `train_summary.json`

### 6. Инференс

Файл:

- [infer.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/src/oinkvision/infer.py)

Что делает:

- загружает модель
- прогоняет индекс
- считает вероятности
- применяет thresholds
- сохраняет CSV с предсказаниями
- если у индекса есть target, считает метрики

Также поддерживает:

- `submission-only` CSV для official `val/test`
- unlabeled raw-index без падения

### 7. Threshold tuning

Скрипт:

- [tune_thresholds.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/scripts/tune_thresholds.py)

Назначение:

- подобрать отдельный threshold для каждого класса на internal valid

Это обязательно для `Macro F1`.

### 8. K-fold evaluation

Скрипты:

- [build_cv_splits.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/scripts/build_cv_splits.py)
- [run_kfold_cv.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/scripts/run_kfold_cv.py)

Назначение:

- уйти от переоценки/недооценки одного случайного split
- понять, насколько устойчив baseline

### 9. Geometry / hybrid layer

Файл:

- [geometry.py](/Users/banworddd/University/Hakaton/oinkVision_belgau/src/oinkvision/geometry.py)

Назначение:

- считать простые геометрические признаки по bbox
- использовать их как auxiliary post-processing сигнал

Сейчас эта часть уже заготовлена, но большого прироста пока не показала.

## Что уже пробовали

На уровне экспериментов проект прошёл через несколько семейств идей:

1. Базовый server baseline с `EfficientNet-B0`
2. Более тяжёлые train-конфиги:
   - больше `image_size`
   - больше `frames_per_camera`
   - больше `batch`
   - больше `epochs`
3. Более мягкое обучение:
   - меньше `lr`
   - scheduler
   - early stopping
4. Более сложная агрегация:
   - class-aware aggregation
   - camera weighting
5. Hybrid postprocess:
   - geometry helper

Лучший стабильный результат локально пока даёт именно аккуратный baseline уровня `v3`, а не самые сложные варианты.

## Самые слабые места текущего решения

Ниже самое важное и самое честное.

### 1. Train и real inference пока не полностью совпадают

Базовый train живёт в “удобном мире”:

- есть `annotation.json`
- есть размеченные `frame_id`
- есть bbox

А в боевом режиме на `val/test`:

- аннотаций нет
- bbox нет
- идеальных кадров нет

Проблема:

- internal validation получается более благоприятной, чем реальный production scenario
- baseline может переоценивать своё реальное качество

Это один из самых принципиальных рисков проекта.

### 2. Слишком мало позитивов в редких классах

Это главная математическая проблема.

Особенно:

- `x_shape = 1 positive`
- `soft_pastern = 10 positive`

Проблема:

- модель почти не видит разнообразия этих состояний
- `Macro F1` при этом требует хорошо работать и по ним
- любое threshold tuning тут нестабильно

Практический эффект:

- `bumps` учится хорошо
- `bad_posture` ещё можно улучшать
- `soft_pastern` и `x_shape` стабильно проседают

### 3. Один split может быть обманчив

Если ориентироваться только на один `train/valid split`, можно:

- переоценить хороший случайный разрез
- недооценить рабочую модель

Из-за экстремального дисбаланса это особенно опасно.

Поэтому `k-fold` сейчас важнее, чем очередной случайный новый `lr`.

### 4. Простая frame-CNN уже близка к своему потолку

По текущим экспериментам видно:

- brute-force увеличение `image_size / frames / epochs` не даёт стабильного прироста
- более сложная train-динамика не дала большого прорыва
- сложная агрегация сама по себе тоже не спасла

Проблема:

- bottleneck уже не только в оптимизации
- bottleneck в самом качестве сигнала

### 5. Геометрический helper пока слабый

Мы уже начали вводить rule-based geometry слой.

Проблема:

- в текущем виде геометрическая эвристика пока слишком простая
- она ещё не заменяет detector/cropper и не даёт надёжный рост сама по себе

То есть идея правильная, но реализация пока лёгкая, а не production-grade.

### 6. Нет отдельного detector/cropper в боевом пайплайне

Сейчас проект умеет работать с raw-видео и без annotation, но:

- bbox в test-time не восстанавливаются отдельной моделью
- значит crop quality в боевом сценарии пока ограничен

Почему это важно:

- для `rear/left/right` качество выделения ног критично
- особенно для редких классов

### 7. `Macro F1` делает редкие ошибки очень дорогими

Если один редкий класс провален в `0.0`, он заметно тянет всю метрику вниз.

Проблема:

- можно иметь очень хороший `bumps`
- неплохой `bad_posture`
- и всё равно получить ограниченный итог

## Что сейчас считается лучшим baseline

Лучший основной baseline:

- [baseline_server_v3.yaml](/Users/banworddd/University/Hakaton/oinkVision_belgau/configs/baseline_server_v3.yaml)

Почему именно он:

- он дал лучший стабильный internal valid score
- более тяжёлые архитектурные/агрегационные варианты его не побили стабильно

## Что делать дальше

### Практический приоритет

1. Стабилизировать оценку через `k-fold`
2. Зафиксировать лучший baseline
3. Использовать его для official `val`

### Исследовательский приоритет

1. Detector/cropper для test-time bbox
2. Более сильный geometric helper для `x_shape`
3. Multi-view fusion архитектура, а не просто mean aggregation

## Основные команды

### Индекс

```bash
python scripts/build_index.py \
  --metadata-path data/train/metadata/meta_data.xlsx
```

### Internal split

```bash
python scripts/build_internal_split.py --config configs/baseline_server_v3.yaml
```

### Rare-class expansion before split (`x_shape`)

```bash
python scripts/augment_xshape_index.py \
  --index-path outputs/index/train_index.csv \
  --output-path outputs/index/train_index_xshape_aug.csv \
  --target-xshape-count 12

python scripts/build_internal_split.py \
  --config configs/baseline_server_v3.yaml \
  --index-path outputs/index/train_index_xshape_aug.csv
```

### Обучение

```bash
python src/oinkvision/train.py \
  --config configs/baseline_server_v3.yaml \
  --train-index outputs/splits/train_split.csv \
  --valid-index outputs/splits/valid_split.csv
```

### Threshold tuning

```bash
python scripts/tune_thresholds.py \
  --config configs/baseline_server_v3.yaml \
  --index-path outputs/splits/valid_split.csv \
  --checkpoint outputs/server_run_v3/best_model.pt \
  --output-json outputs/server_run_v3/tuned_thresholds.json
```

### Final internal valid infer

```bash
python src/oinkvision/infer.py \
  --config configs/baseline_server_v3.yaml \
  --index-path outputs/splits/valid_split.csv \
  --checkpoint outputs/server_run_v3/best_model.pt \
  --thresholds-json outputs/server_run_v3/tuned_thresholds.json \
  --output-csv outputs/server_run_v3/valid_tuned_predictions.csv \
  --metrics-json outputs/server_run_v3/valid_tuned_metrics.json
```

### K-fold

```bash
python scripts/run_kfold_cv.py \
  --config configs/baseline_server_v3.yaml \
  --index-path outputs/index/train_index.csv \
  --n-splits 3 \
  --rebuild-splits
```

`cv_summary.json` теперь содержит:
- `macro_f1_mean` (классический macro по всем классам),
- `macro_f1_present_classes_mean` (без штрафа за классы с нулевым support в конкретном fold),
- `macro_f1_mean_xshape_present_folds` и `num_folds_with_xshape_positive`.

### Geometry fusion tuning

```bash
python scripts/tune_geometry_fusion.py \
  --config configs/baseline_server_v3.yaml \
  --index-path outputs/splits/valid_split.csv \
  --checkpoint outputs/server_run_v3/best_model.pt \
  --output-json outputs/server_run_v3/tuned_geometry_fusion.json
```

### Raw val/test index

```bash
python scripts/build_raw_index.py --split val
```

### Official val/test style inference

```bash
python src/oinkvision/infer.py \
  --config configs/baseline_server_v3.yaml \
  --index-path outputs/index/val_index.csv \
  --checkpoint outputs/server_run_v3/best_model.pt \
  --thresholds-json outputs/server_run_v3/tuned_thresholds.json \
  --output-csv outputs/server_run_v3/val_submission.csv \
  --submission-only
```

## Итог

Проект уже содержит:

- рабочий animal-level baseline
- train/internal valid пайплайн
- threshold tuning
- raw-index и unlabeled inference
- `k-fold`-оценку
- заготовки под detector и geometry helper

Главное ограничение сейчас не в том, что “не хватает ещё одной эпохи”, а в том, что:

- данных по редким классам очень мало
- train и production режимы пока не полностью совпадают
- простой CNN-baseline почти упёрся в потолок

Именно поэтому самые перспективные следующие шаги:

- честная `k-fold`-оценка
- detector/cropper
- более сильный hybrid подход для редких классов
