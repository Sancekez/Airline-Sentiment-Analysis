# ✈️ Airline Sentiment Analysis & Emotion Classification

**Нейросеть для анализа тональности и классификации эмоций в отзывах авиапассажиров**

> Магистрант: Комаров Николай Александрович  
> МУИТ (IITU), Алматы, 2026

---

## 📋 Что делает этот проект?

Система принимает текст отзыва авиапассажира и автоматически определяет:

| Задача | Что определяет | Примеры классов |
|--------|---------------|-----------------|
| **Тональность** | Настроение отзыва | positive / neutral / negative |
| **Категория** | О чём жалоба/отзыв | багаж, задержка, регистрация, бронирование, обслуживание на борту, сервис |
| **Критичность** | Насколько срочно реагировать | low / medium / high |

**Пример:** текст `"My flight was delayed 5 hours!"` → тональность: **negative (98%)**, категория: **delay (60%)**, критичность: **medium (95%)**

---

## 🚀 Запуск проекта (пошаговая инструкция)

### Требования

- Операционная система: **Ubuntu 20.04 / 22.04 / 24.04** (или WSL2 на Windows)
- Python: **3.10 или новее**
- Интернет: нужен при первом запуске (скачивание пакетов и модели DistilBERT ~260 МБ)
- Оперативная память: минимум **4 ГБ** (рекомендуется 8 ГБ)

### Шаг 1. Скачать проект

Откройте терминал (Ctrl+Alt+T) и выполните:

```bash
git clone https://github.com/Sancekez/Airline-Sentiment-Analysis.git
```

Перейдите в папку проекта:

```bash
cd Airline-Sentiment-Analysis
```

> **Если у вас нет git**, установите его: `sudo apt install git -y`

### Шаг 2. Автоматическая установка

Запустите скрипт установки:

```bash
chmod +x setup.sh
./setup.sh
```

Скрипт автоматически выполнит:
- ✅ Установку системных пакетов (`python3-venv`, `python3-pip`)
- ✅ Создание виртуального окружения Python
- ✅ Установку всех библиотек (scikit-learn, PyTorch, transformers, FastAPI и др.)
- ✅ Генерацию датасета (14 600 отзывов авиапассажиров)
- ✅ Запуск 19 тестов для проверки работоспособности
- ✅ Обучение baseline-модели (TF-IDF + LogisticRegression, ~10 секунд)

> ⏱ Время выполнения: **3–5 минут** (зависит от скорости интернета)
>
> 🔑 Скрипт попросит пароль один раз — это для установки системных пакетов через `sudo`

**Как понять что всё прошло успешно?** В конце увидите:

```
✅ SETUP COMPLETE!
```

### Шаг 3. Запуск API-сервера

```bash
chmod +x run.sh
./run.sh
```

Увидите в терминале:

```
Starting Airline Sentiment API on http://0.0.0.0:8000
Swagger docs: http://localhost:8000/docs
Press Ctrl+C to stop
```

**Сервер запущен!** Не закрывайте этот терминал — он должен работать пока вы пользуетесь API.

---

## 🔌 Использование API

### Способ 1: Через браузер (Swagger UI) — самый простой

1. Откройте в браузере: **http://localhost:8000/docs**
2. Вы увидите интерактивную документацию API

#### Анализ одного текста:

1. Найдите **POST /predict** → нажмите на него
2. Нажмите кнопку **"Try it out"** (справа)
3. В поле Request body вставьте:

```json
{
  "text": "My flight was delayed 5 hours and nobody helped!"
}
```

4. Нажмите синюю кнопку **"Execute"**
5. Ниже появится ответ с тональностью, категорией и критичностью

#### Анализ нескольких текстов сразу:

1. Найдите **POST /predict/batch** → **"Try it out"**
2. Вставьте массив текстов:

```json
[
  "My flight was delayed 5 hours and nobody helped us!",
  "Amazing crew, best flight ever! Comfortable seats.",
  "Lost my luggage and customer service hung up on me.",
  "Average flight, nothing special but okay for the price.",
  "Check-in took 2 hours. Absolutely unacceptable!"
]
```

3. Нажмите **"Execute"** — получите результат для всех текстов за один запрос

#### Проверка статуса сервиса:

1. Найдите **GET /health** → **"Try it out"** → **"Execute"**
2. Покажет загружена ли модель и какая (baseline или bert)

### Способ 2: Через командную строку (curl)

Откройте **второй терминал** (первый занят сервером) и выполните:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My flight was delayed 5 hours and nobody helped!"}'
```

Ответ придёт в формате JSON:

```json
{
  "sentiment": {"label": "negative", "confidence": 0.98},
  "category": {"label": "delay", "confidence": 0.60},
  "criticality": {"label": "medium", "confidence": 0.95},
  "model_type": "bert"
}
```

### Способ 3: Через Python-код

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Amazing crew, best flight ever!"}
)

result = response.json()
print(f"Тональность: {result['sentiment']['label']}")
print(f"Категория: {result['category']['label']}")
print(f"Критичность: {result['criticality']['label']}")
```

### Способ 4: Через Postman

1. Откройте Postman → **New Request**
2. Метод: **POST**
3. URL: `http://localhost:8000/predict`
4. Вкладка **Body** → **raw** → **JSON**
5. Вставьте: `{"text": "Flight was cancelled without notice!"}`
6. Нажмите **Send**

---

## 📊 Пример ответа API

```json
{
  "sentiment": {
    "label": "negative",
    "confidence": 0.98,
    "probabilities": {
      "negative": 0.98,
      "neutral": 0.01,
      "positive": 0.01
    }
  },
  "category": {
    "label": "delay",
    "confidence": 0.60,
    "probabilities": {
      "baggage": 0.03,
      "booking": 0.22,
      "delay": 0.60,
      "in-flight": 0.02,
      "check-in": 0.05,
      "customer_service": 0.06,
      "other": 0.03
    }
  },
  "criticality": {
    "label": "medium",
    "confidence": 0.95,
    "probabilities": {
      "low": 0.02,
      "medium": 0.95,
      "high": 0.03
    }
  },
  "model_type": "bert"
}
```

**Расшифровка:**
- **label** — предсказанный класс
- **confidence** — уверенность модели (от 0 до 1, чем ближе к 1 — тем увереннее)
- **probabilities** — вероятности для всех классов (в сумме дают 1.0)
- **model_type** — какая модель использована (`bert` или `baseline`)

---

## 🛑 Остановка и повторный запуск

**Остановить сервер:** нажмите `Ctrl+C` в терминале где запущен сервер

**Запустить снова:**

```bash
cd Airline-Sentiment-Analysis
./run.sh
```

---

## 🧪 Запуск тестов

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

Должно показать: `19 passed ✅`

---

## 📁 Структура проекта

```
Airline-Sentiment-Analysis/
├── src/                        # Исходный код
│   ├── config.py               #   Константы и гиперпараметры
│   ├── data.py                 #   Загрузка и предобработка данных
│   ├── baseline.py             #   Baseline-модель (TF-IDF + LogReg)
│   └── bert_model.py           #   DistilBERT multi-task модель
├── scripts/                    # Скрипты запуска
│   ├── train.py                #   Главный пайплайн обучения
│   ├── train_bert.sh           #   Обучение BERT одной командой
│   └── generate_data.py        #   Генератор датасета
├── api/                        # REST API
│   └── server.py               #   FastAPI-сервер
├── tests/                      # Тесты
│   └── test_pipeline.py        #   19 unit-тестов (pytest)
├── notebooks/                  # Jupyter-ноутбуки
│   └── train_bert_colab.ipynb  #   Обучение BERT в Google Colab
├── data/                       # Данные (генерируются автоматически)
├── models/                     # Обученные модели
│   ├── baseline/               #   TF-IDF + LogReg
│   └── bert/                   #   DistilBERT (через Git LFS)
├── reports/                    # Отчёты и метрики
│   ├── REPORT.md               #   Отчёт по практике
│   └── metrics.json            #   Метрики моделей
├── setup.sh                    # Автоустановка (одна команда)
├── run.sh                      # Запуск API-сервера
├── requirements.txt            # Python-зависимости
├── Makefile                    # Альтернативные команды
├── .gitignore                  # Исключения для Git
└── README.md                   # Этот файл
```

---

## 🖥️ Вычислительные ресурсы

| Этап | Ресурс | Время |
|------|--------|-------|
| Установка | Любой CPU + интернет | ~3–5 минут |
| Baseline обучение | CPU | ~10 секунд |
| BERT обучение | CPU (Intel i9-13900H) | ~2.5 часа |
| BERT обучение | GPU (Google Colab T4) | ~10 минут |
| API inference | CPU | <100 мс/запрос |

---

## ❓ Частые проблемы

**`python3-venv: No such file or directory`**
```bash
sudo apt install python3-venv -y
```

**`git: command not found`**
```bash
sudo apt install git -y
```

**`Permission denied` при запуске скриптов**
```bash
chmod +x setup.sh run.sh scripts/train_bert.sh
```

**Порт 8000 уже занят**
```bash
# Найти процесс на порту 8000
lsof -i :8000
# Убить процесс (заменить PID на число из вывода)
kill -9 PID
# Или запустить на другом порту
python -m uvicorn api.server:app --host 0.0.0.0 --port 8080
```

**API отвечает "Model not loaded"**
```bash
# Нужно сначала обучить модель
source venv/bin/activate
python scripts/generate_data.py
python scripts/train.py --baseline-only
./run.sh
```

---

## 🛠️ Технологический стек

| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.11 |
| ML | scikit-learn |
| Deep Learning | PyTorch + HuggingFace Transformers |
| NLP-модель | DistilBERT (distilbert-base-uncased) |
| API | FastAPI + Uvicorn |
| Тестирование | pytest (19 тестов) |

---

## Тренировка Модели

Обучение BERT-модели

Baseline-модель уже работает, но для лучшего качества обучите DistilBERT:

```bash
chmod +x scripts/train_bert.sh
./scripts/train_bert.sh
```

> ⏱ Время обучения:
> - На CPU: **~30 минут — 2.5 часа** (зависит от процессора, я пробовал обучать на Intel i9-13900H)
> - На GPU (Google Colab): **~10 минут**
>
> 💡 Если BERT-модель уже есть в репозитории (папка `models/bert/`), этот шаг можно пропустить.

**Как проверить что модель есть?**

```bash
ls models/bert/model.pt
```

Если файл существует — BERT готов к использованию.


## 📄 Лицензия

Проект создан в рамках учебной практики МУИТ (IITU), Алматы, 2026.
