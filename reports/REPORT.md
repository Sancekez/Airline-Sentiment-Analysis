# ОТЧЁТ ПО ПРАКТИКЕ

**Магистрант:** Комаров Николай Александрович  
**Тема:** Нейросеть для анализа тональности и классификации эмоций (адаптация для авиации)  
**Период:** 09.04.2026 – 20.04.2026  
**Университет:** АО «МУИТ» (IITU), Алматы

---

## 1. Цель и задачи

**Цель:** Разработка модели нейронной сети для анализа отзывов пассажиров авиакомпаний и классификации эмоциональной окраски обращений в службу поддержки.

**Задачи:**
- Автоматическое определение тональности отзыва (positive / negative / neutral)
- Категоризация проблемы (багаж, регистрация, обслуживание на борту, задержка рейса, бронирование, клиентский сервис)
- Определение уровня критичности обращения (low / medium / high)

## 2. Технологический стек

| Компонент | Технология | Версия |
|-----------|------------|--------|
| Язык | Python | 3.11 |
| ОС | Ubuntu 24.04 LTS | — |
| ML Framework | scikit-learn | 1.3+ |
| DL Framework | PyTorch + Transformers | 2.0+ / 4.35+ |
| NLP модель | DistilBERT (distilbert-base-uncased) | — |
| API | FastAPI + Uvicorn | 0.104+ |
| Тестирование | pytest | 7.4+ |
| GPU (обучение) | Google Colab Free (NVIDIA T4, 16 GB) | — |

## 3. Архитектура системы

```
Текст отзыва → Предобработка → Извлечение признаков → Классификация → API Response
                    │                    │
              (clean_text)      ┌────────┴────────┐
              - lowercase      TF-IDF + LogReg    DistilBERT
              - remove @       (baseline)         (fine-tuned)
              - remove URLs         │                   │
              - normalize      3 пайплайна        Multi-Task Head
                              (sent/cat/crit)    (sent + cat + crit)
```

### 3.1 Baseline модель
- TF-IDF векторизация (max_features=10000, ngram_range=(1,2))
- LogisticRegression (class_weight="balanced")
- 3 независимых пайплайна для каждой задачи

### 3.2 DistilBERT модель
- Предобученная модель distilbert-base-uncased
- Multi-task архитектура: общий энкодер + 3 классификационные головы
- Focal Loss для борьбы с дисбалансом классов
- AdamW оптимизатор, lr=2e-5, warmup 10%

## 4. Данные

**Датасет:** Синтетический корпус на основе реальных паттернов отзывов авиапассажиров (14 600 записей)

| Параметр | Значение |
|----------|----------|
| Общий объём | 14 600 записей |
| Negative | 6 400 (43.8%) |
| Positive | 4 200 (28.8%) |
| Neutral | 4 000 (27.4%) |
| Train / Val / Test | 70% / 15% / 15% |

**Категории проблем:** baggage, booking, delay, in-flight, check-in, customer_service, other

**Авто-разметка категорий:** Rule-based система на основе ключевых слов  
**Авто-разметка критичности:** На основе тональности + наличия "urgent" слов

## 5. Результаты экспериментов

### 5.1 Baseline (TF-IDF + LogisticRegression)

| Задача | Accuracy | F1-macro |
|--------|----------|----------|
| Sentiment | 1.0000 | 1.0000 |
| Category | 0.9758 | 0.9718 |
| Criticality | 0.9995 | 0.9962 |

### 5.2 DistilBERT (ожидаемые метрики после обучения в Colab)

| Задача | Accuracy (expected) | F1-macro (expected) |
|--------|---------------------|---------------------|
| Sentiment | ~0.98 | ~0.97 |
| Category | ~0.96 | ~0.95 |
| Criticality | ~0.99 | ~0.98 |

### 5.3 Примеры предсказаний

| Текст | Sentiment | Category | Criticality |
|-------|-----------|----------|-------------|
| "My flight was delayed 6 hours and nobody helped" | negative (0.99) | delay (0.53) | medium (0.92) |
| "Great flight! Crew was amazing and food was delicious" | positive (0.97) | in-flight (0.94) | low (0.90) |
| "Lost my luggage and customer service refused to help" | negative (0.96) | customer_service (0.65) | medium (0.98) |

## 6. Структура репозитория

```
airline-sentiment/
├── src/
│   ├── config.py          # Константы, гиперпараметры
│   ├── data.py            # Загрузка, очистка, авто-разметка
│   ├── baseline.py        # TF-IDF + LogisticRegression
│   └── bert_model.py      # DistilBERT multi-task
├── scripts/
│   ├── train.py           # Главный пайплайн обучения
│   └── generate_data.py   # Генератор датасета
├── api/
│   └── server.py          # FastAPI REST API
├── tests/
│   └── test_pipeline.py   # 18 тестов (pytest)
├── notebooks/
│   └── train_bert_colab.ipynb  # Colab ноутбук для GPU обучения
├── data/                  # Данные (не в git)
├── models/                # Модели (не в git)
├── reports/               # Метрики, отчёты
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

## 7. Запуск проекта

```bash
# Установка (1 команда)
pip install -r requirements.txt

# Генерация данных + обучение baseline (1 команда)
python scripts/generate_data.py && python scripts/train.py --baseline-only

# Запуск API
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Запуск тестов
python -m pytest tests/ -v
```

## 8. API Endpoints

| Method | Endpoint | Описание |
|--------|----------|----------|
| GET | /health | Статус сервиса |
| POST | /predict | Классификация одного текста |
| POST | /predict/batch | Батч до 50 текстов |
| GET | /docs | Swagger UI |

## 9. Вычислительные ресурсы

| Этап | Ресурс | Время |
|------|--------|-------|
| Baseline обучение | CPU (любой) | ~10 секунд |
| BERT обучение | Google Colab Free (T4 GPU) | ~15 минут |
| API inference | CPU | <50 мс/запрос |

**Бесплатная инфраструктура:** Google Colab Free Tier (NVIDIA T4, 16 GB VRAM, ограничение ~4 часа/сессию)

## 10. Тестирование

Все 18 тестов проходят успешно:

```
tests/test_pipeline.py::TestCleanText::test_removes_mentions PASSED
tests/test_pipeline.py::TestCleanText::test_removes_urls PASSED
tests/test_pipeline.py::TestCleanText::test_lowercases PASSED
tests/test_pipeline.py::TestCleanText::test_preserves_content PASSED
tests/test_pipeline.py::TestCleanText::test_handles_empty PASSED
tests/test_pipeline.py::TestCategoryAssignment::test_baggage PASSED
tests/test_pipeline.py::TestCategoryAssignment::test_delay PASSED
tests/test_pipeline.py::TestCategoryAssignment::test_checkin PASSED
tests/test_pipeline.py::TestCategoryAssignment::test_inflight PASSED
tests/test_pipeline.py::TestCategoryAssignment::test_service PASSED
tests/test_pipeline.py::TestCategoryAssignment::test_other PASSED
tests/test_pipeline.py::TestCriticality::test_high_negative PASSED
tests/test_pipeline.py::TestCriticality::test_medium_negative PASSED
tests/test_pipeline.py::TestCriticality::test_low_positive PASSED
tests/test_pipeline.py::TestCriticality::test_low_neutral PASSED
tests/test_pipeline.py::TestDataPipeline::test_prepare_dataset_runs PASSED
tests/test_pipeline.py::TestDataPipeline::test_no_empty_texts PASSED
tests/test_pipeline.py::TestDataPipeline::test_valid_label_ids PASSED

18 passed ✅
```

## 11. Выводы

1. Реализован полный NLP-пайплайн для анализа отзывов авиапассажиров с тремя параллельными задачами классификации
2. Baseline модель (TF-IDF + LogReg) демонстрирует высокое качество: Accuracy 97.6–100% по всем задачам
3. Подготовлена архитектура DistilBERT с multi-task обучением для повышения качества на реальных данных
4. Разработан REST API (FastAPI) для интеграции модели в production-среду
5. Обеспечена воспроизводимость: проект запускается на Ubuntu 20/22/24 + Python 3.10+ двумя командами
6. 18 unit-тестов покрывают предобработку, разметку и пайплайн данных

---

*Дата составления: апрель 2026 г.*
