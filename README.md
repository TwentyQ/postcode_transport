Это новый репозиторий, поскольку старый является приватным. Доказательство владения и выполнения в файле proof_2.png.

---

# Сервис оценки загрузки грузового отсека

MVP сервиса для Почты России: принимает фотографию грузового отсека и номер перевозки, определяет процент загрузки кузова (0–100) с помощью нейросети, сохраняет результат и возвращает его через XML API.

## Возможности

- **Распознавание загрузки** - прогноз процента загрузки по фотографии (0–100)
- **Привязка к перевозке** - сохранение и получение результата по `transport_id`
- **XML API** - приём фото и получение результата через интерфейс
- **Обработка ошибок** - понятные ответы при повреждённом/неподдерживаемом изображении и некорректном XML
- **Дообучение** - воспроизводимое дообучение модели на новых размеченных фото через CLI и API
- **Уверенность прогноза и флаг недогруза** - дополнительная информация для оператора

## Результаты

| Метрика | Значение |
|---|---|
| Локальная валидация (MAE) | ~6.0 п.п. |
| Public-лидерборд (MAE) | 6.3 п.п. |
| Доля прогнозов с ошибкой ≤10 п.п. | ~71% |
| Место в ML-рейтинге | 16 |

Модель - ConvNeXt-Tiny, предобученная на ImageNet, обучена с 5-fold кросс-валидацией по группам (`GroupKFold`), с multi-scale TTA (288 и 320, отражение по горизонтали)

## Архитектура

- `app.py` - Flask-сервер, XML API, UI
- `id_transportation.py` - хранение результатов в JSON
- `ml/model.py` - загрузка модели и функция `predict()`
- `ml/retrain.py` - дообучение на новых данных
- `ml/best_model.pth` - веса модели
- `templates/index.html`, `static/style.css` - веб-интерфейс

## Установка и запуск

### Зависимости

```bash
pip install -r requirements.txt
```

### Запуск сервера

```bash
python app.py
```

## XML API

Все методы принимают и возвращают XML (UTF-8). Изображение передаётся как **base64 внутри XML**.

### 1. Прогноз загрузки

**`POST /api/predict`**

**Запрос:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<request>
  <transport_id>T-DEMO-001</transport_id>
  <image_base64>...</image_base64>
</request>
```

- `transport_id` - номер перевозки
- `image_base64` - изображение `.jpg`/`.png`, закодированное в base64

**Успешный ответ (200):**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<response>
  <status>success</status>
  <transport_id>T-DEMO-001</transport_id>
  <load_pct>65.30</load_pct>
  <confidence>0.87</confidence>
  <is_underloaded>false</is_underloaded>
</response>
```

**Ошибки (400):**

```xml
<response>
  <status>error</status>
  <message>Некорректный XML</message>
</response>
```

Возможные сообщения об ошибках:
- `Некорректный XML` - тело запроса не парсится
- `Не указан transport_id` - отсутствует тег
- `Не передано изображение` - отсутствует `image_base64`
- `Некорректный base64` - не удалось декодировать
- `Файл повреждён или не является изображением` - картинка битая

### 2. Получение сохранённого результата

**`GET /api/result/<transport_id>`**

**Успешный ответ:**

```xml
<response>
  <status>success</status>
  <transport_id>T-DEMO-001</transport_id>
  <load_pct>65.30</load_pct>
</response>
```

**Если перевозки нет (404):**

```xml
<response>
  <status>error</status>
  <message>Перевозка T-DEMO-001 не найдена</message>
</response>
```

### 3. Дообучение модели

**`POST /api/retrain`**

**Запрос:**

```xml
<request>
  <csv_base64>aW1hZ2VfaWQsbG9hZF9wY3QKLi4u</csv_base64>
  <image_base64 image_id="img_001"></image_base64>
  <image_base64 image_id="img_002"></image_base64>
</request>
```

- `csv_base64` - CSV с колонками `image_id,load_pct`, в base64
- `image_base64` - по одному тегу на каждое фото, атрибут `image_id` = имя файла без `.jpg`

**Ответ:**

```xml
<response>
  <status>started</status>
  <job_id>a1b2c3d4</job_id>
  <image_count>5</image_count>
</response>
```

**`GET /api/retrain_status/<job_id>`** - статус задачи

```xml
<response>
  <status>running</status>
  <job_id>a1b2c3d4</job_id>
  <log>Epoch 1: MAE=6.42...</log>
</response>
```

Когда задача завершена:

```xml
<status>done</status>
<model_path>ml/best_model_a1b2c3d4.pth</model_path>
```

### 4. Веб-интерфейс

- `GET /` - страница с формой загрузки фото и формой дообучения
- `GET /ui/result/<transport_id>` - JSON с результатом

## Поведение при повторном обращении к одной перевозке

- `POST /api/predict` с существующим `transport_id` - перезаписывает сохранённый результат новым прогнозом
- `GET /api/result/<transport_id>` всегда возвращает последний сохранённый результат
- История обращений сохраняется в поле `history` (функция `save_result_full`)

## Дообучение через CLI

```bash
python ml/retrain.py \
  --csv new_data.csv \
  --images new_images/ \
  --base ml/best_model.pth \
  --out ml/best_model_v2.pth \
  --arch convnext_tiny \
  --epochs 3 \
  --lr 1e-5
```

Аргументы:

| Аргумент | Описание |
|---|---|
| `--csv` | CSV с колонками `image_id,load_pct` |
| `--images` | Папка с новыми `.jpg` (имя файла = `<image_id>.jpg`) |
| `--base` | Текущие веса |
| `--out` | Куда сохранить новые веса |
| `--arch` | `convnext_tiny` или `resnet18` |
| `--epochs` | Число эпох (по умолчанию 3) |
| `--lr` | Learning rate (по умолчанию 1e-5) |
| `--batch` | Размер батча (по умолчанию 8) |

! После дообучения заменить `ml/best_model.pth` на новый файл и перезапустить сервер

## Обучение модели с нуля

1. Открыть `ml_part.ipynb` в Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Загрузить `train.zip` и `test.zip` в рабочую директорию
4. Запустить ячейки последовательно:
   - загрузка данных, разбиение по группам
   - обучение 5 фолдов ConvNeXt-Tiny
   - выбор лучшего фолда - `best_model.pth`
   - предсказание на тесте с multi-scale TTA - `submission_v2.csv`

**Оборудование:** Google Colab, GPU. Общее время обучения 5 фолдов - ~1 час

## Зависимости

`requirements.txt`:
- flask
- torch
- torchvision
- pillow
- numpy
- requests
