# Credit Card Default Prediction Service

ML-сервис для прогнозирования дефолта по кредитным картам на основе датасета UCI Credit Card.  
Принимает финансовые признаки клиента и возвращает вероятность невыплаты в следующем месяце.

---

## Структура репозитория

```
credit-card-ml/
├── app/
│   └── api.py              # Flask REST API
├── models/
│   ├── model_v1.pkl        # RandomForestClassifier
│   └── model_v2.pkl        # LogisticRegression + StandardScaler
├── notebooks/              # EDA и эксперименты
├── tests/                  # Unit и интеграционные тесты
├── train_model.py          # Скрипт обучения моделей
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── ab_test_plan.md
```

---

## Запуск локально

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Обучить модели (нужен UCI_Credit_Card.csv в корне)
python train_model.py

# 3. Запустить сервис
python app/api.py
# Сервис доступен на http://localhost:5000
```

---

## Запуск через Docker

```bash
# Собрать образ и запустить контейнер
docker-compose up --build -d

# Остановить
docker-compose down

# Логи
docker-compose logs -f
```

---

## Примеры запросов

### GET /health

```bash
curl http://localhost:5000/health
```

```json
{"status": "healthy"}
```

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 5000, "BILL_AMT2": 4000, "BILL_AMT3": 3000,
    "BILL_AMT4": 2000, "BILL_AMT5": 1000, "BILL_AMT6": 500,
    "PAY_AMT1": 1000, "PAY_AMT2": 800, "PAY_AMT3": 600,
    "PAY_AMT4": 400, "PAY_AMT5": 200, "PAY_AMT6": 100
  }'
```

```json
{"prediction": 0, "probability": 0.27}
```

| Поле | Описание |
|---|---|
| `prediction` | `0` — дефолт маловероятен, `1` — высокий риск дефолта |
| `probability` | Вероятность дефолта от 0.0 до 1.0 |

---

## Docker Hub

Образ опубликован публично:

```bash
docker pull docker.io/manasbegaliev777/credit-card-ml:v1
docker run -p 5000:5000 manasbegaliev777/credit-card-ml:v1
```

**Ссылка:** https://hub.docker.com/r/manasbegaliev777/credit-card-ml

---

## Архитектура

### Текущий подход: монолит

Сервис реализован как единое Flask-приложение, в котором модель загружается при старте и обслуживает все запросы.

**Преимущества монолита на текущем этапе:**
- Простота деплоя — один контейнер, один порт
- Низкая латентность — нет сетевых вызовов между сервисами
- Простота отладки и логирования
- Достаточно для нагрузки до ~500 RPS на одном инстансе

**Когда переходить на микросервисы:**

| Сигнал | Решение |
|---|---|
| Разные модели требуют независимого скейлинга | Выделить каждую модель в отдельный сервис |
| Feature engineering тяжёлый и переиспользуется | Отдельный Feature Store сервис |
| Нужен онлайн A/B роутинг | API Gateway + Model Router |
| Команды работают над разными компонентами независимо | Микросервисная декомпозиция |

---

## MLOps концепты

### DVC (Data Version Control)
Версионирует датасеты и модели аналогично тому, как Git версионирует код. Позволяет воспроизвести любой эксперимент, откатиться к предыдущей версии данных, хранить большие файлы в S3/GCS вне репозитория.

```bash
dvc add UCI_Credit_Card.csv
dvc push  # загрузить в удалённое хранилище
```

### MLflow
Платформа для трекинга экспериментов: логирует параметры, метрики, артефакты моделей. Позволяет сравнивать запуски, регистрировать модели в Model Registry и управлять их жизненным циклом (Staging → Production → Archived).

```python
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("roc_auc", 0.7506)
    mlflow.sklearn.log_model(rf, "model")
```

### RabbitMQ
Брокер сообщений для асинхронной обработки запросов. Актуален когда:
- Предсказания не требуют мгновенного ответа (batch scoring)
- Нужно сгладить пиковую нагрузку (очередь запросов)
- Требуется гарантированная доставка и retry-логика

Типовая схема: `Client → RabbitMQ → Worker (model inference) → Results DB`

---

## Бизнес-метрики

### Снижение финансовых потерь
Каждый пропущенный дефолт (False Negative) — прямой убыток банка.  
При среднем кредитном лимите ~50 000 руб. и доле дефолтов 22% модель с ROC-AUC 0.75 позволяет:
- Заблокировать ~60% будущих дефолтов до выдачи кредита
- Снизить потери портфеля на 10–15% при пороге отсечения 0.5

### Доля одобренных заявок
Ключевой баланс — не отклонять слишком много заявок (False Positives снижают выручку).  
Бизнес-метрика: **Approval Rate** — доля одобренных заявок от общего числа.  
Целевой диапазон: сохранить Approval Rate ≥ 75% при одновременном снижении дефолтов на 10%+.

### KPI для мониторинга в проде
| Метрика | Целевое значение |
|---|---|
| F1-score | ≥ 0.50 |
| ROC-AUC | ≥ 0.75 |
| Approval Rate | ≥ 75% |
| Среднее время ответа API | ≤ 100 мс |
| Data drift (PSI) | < 0.2 |
