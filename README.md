# ✈️ Airline Sentiment Analysis & Emotion Classification

**Нейросеть для анализа тональности и классификации эмоций (адаптация для авиации)**

> Магистрант: Комаров Николай Александрович  
> Практика: 09.04.2026 – 20.04.2026  
> МУИТ (IITU), Алматы

---

## 📋 Описание проекта

Система автоматического анализа отзывов и обращений авиапассажиров, выполняющая три задачи:

| Задача | Описание | Классы |
|--------|----------|--------|
| **Sentiment** | Тональность отзыва | positive / neutral / negative |
| **Category** | Категория проблемы | baggage, booking, delay, in-flight, check-in, customer_service, other |
| **Criticality** | Уровень критичности | low / medium / high |

## 🏗️ Архитектура

```
Input Text → Preprocessing → Feature Extraction → Classification → API Response
                                    │
                         ┌──────────┴──────────┐
                    TF-IDF + LogReg        DistilBERT
                     (baseline)           (fine-tuned)
                         │                     │
                    3 классификатора    Multi-Task Head
                   (sent/cat/crit)    (sent + cat + crit)
```

## 🚀 Быстрый старт

### Требования
- Ubuntu 20.04 / 22.04 / 24.04 LTS
- Python 3.10+

### Установка и запуск (2 команды на чистой Ubuntu)

```bash
chmod +x setup.sh && ./setup.sh
./run.sh
```

Первая команда: ставит системные пакеты, создаёт venv, ставит зависимости, генерирует данные, прогоняет тесты и обучает baseline.

Вторая команда: запускает API на `http://localhost:8000` (Swagger: `http://localhost:8000/docs`).

### Ручной запуск (если нужен контроль)

```bash
sudo apt install python3-venv -y
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py --baseline-only
```

### Запуск API

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Запуск тестов

```bash
python -m pytest tests/ -v
```

## 📁 Структура проекта

```
airline-sentiment/
├── src/
│   ├── __init__.py
│   ├── config.py          # Константы, пути, гиперпараметры
│   ├── data.py            # Загрузка, очистка, авто-разметка
│   ├── baseline.py        # TF-IDF + LogisticRegression
│   └── bert_model.py      # DistilBERT multi-task fine-tuning
├── scripts/
│   └── train.py           # Главный пайплайн обучения
├── api/
│   ├── __init__.py
│   └── server.py          # FastAPI REST API
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py   # Тесты (pytest)
├── data/                  # Данные (не в git)
├── models/                # Сохранённые модели (не в git)
├── reports/               # Метрики, отчёты
├── notebooks/             # Jupyter ноутбуки (эксперименты)
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

## 📊 Метрики

### Baseline (TF-IDF + LogisticRegression)

| Task | Accuracy | F1-macro |
|------|----------|----------|
| Sentiment | ~0.78 | ~0.75 |
| Category | ~0.65 | ~0.55 |
| Criticality | ~0.72 | ~0.60 |

### BERT (DistilBERT fine-tuned)

| Task | Accuracy | F1-macro |
|------|----------|----------|
| Sentiment | ~0.88 | ~0.86 |
| Category | ~0.75 | ~0.68 |
| Criticality | ~0.82 | ~0.75 |

*Метрики обновляются после обучения → `reports/metrics.json`*

## 🔌 API Endpoints

| Method | Endpoint | Описание |
|--------|----------|----------|
| `GET` | `/health` | Статус сервиса |
| `POST` | `/predict` | Классификация одного текста |
| `POST` | `/predict/batch` | Батч-классификация (до 50 текстов) |
| `GET` | `/docs` | Swagger UI документация |

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My flight was delayed 5 hours and luggage was lost!"}'
```

### Пример ответа

```json
{
  "sentiment": {
    "label": "negative",
    "confidence": 0.94,
    "probabilities": {"negative": 0.94, "neutral": 0.04, "positive": 0.02}
  },
  "category": {
    "label": "delay",
    "confidence": 0.72
  },
  "criticality": {
    "label": "high",
    "confidence": 0.85
  },
  "model_type": "bert"
}
```

## 🖥️ Вычислительные ресурсы

| Ресурс | Использование |
|--------|---------------|
| **Baseline** | CPU, ~1 минута обучения |
| **BERT** | Google Colab Free (GPU T4), ~15 минут |
| **API** | CPU (любой сервер) |

## 🛠️ Технологический стек

- **Python 3.11** — основной язык
- **scikit-learn** — baseline модели, метрики
- **transformers + PyTorch** — DistilBERT fine-tuning
- **FastAPI** — REST API
- **pandas, numpy** — обработка данных
- **pytest** — тестирование

## 📄 Лицензия

Проект создан в рамках учебной практики МУИТ (IITU).
