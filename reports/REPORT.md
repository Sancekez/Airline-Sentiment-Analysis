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
| ML Framework | scikit-learn | 1.6.1 |
| DL Framework | PyTorch + Transformers | 2.6.0 / 4.51.3 |
| NLP модель | DistilBERT (distilbert-base-uncased) | — |
| API | FastAPI + Uvicorn | 0.115.12 / 0.34.2 |
| Тестирование | pytest | 8.3.5 |

## 3. Архитектура системы

```
Текст отзыва → Предобработка → Извлечение признаков → Классификация → API Response
                    │                    │
              (clean_text)      ┌────────┴────────┐
              - lowercase      TF-IDF + LogReg    DistilBERT
              - remove @       (baseline)         (fine-tuned)
              - remove URLs         │                   │
              - negation       3 пайплайна        Multi-Task Head
                handling      (sent/cat/crit)    (sent + cat + crit)
```

### 3.1 Baseline модель
- TF-IDF векторизация (max_features=10000, ngram_range=(1,2), sublinear_tf=True)
- LogisticRegression (class_weight="balanced", solver="lbfgs", C=1.0)
- 3 независимых пайплайна для каждой задачи
- Обработка отрицаний: "no delays" → "no_delays" (единый токен)

### 3.2 DistilBERT модель
- Предобученная модель distilbert-base-uncased (66M параметров)
- Multi-task архитектура: общий энкодер + 3 классификационные головы
- Каждая голова: Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→N_classes)
- Focal Loss для борьбы с дисбалансом классов (γ=2)
- AdamW оптимизатор, lr=2e-5, weight_decay=0.01, warmup 10%
- Gradient clipping: max_norm=1.0

## 4. Данные

**Датасет:** Синтетический корпус на основе реальных паттернов отзывов авиапассажиров

| Параметр | Значение |
|----------|----------|
| Общий объём | 14 600 записей |
| Negative | 6 400 (43.8%) |
| Positive | 4 200 (28.8%) |
| Neutral | 4 000 (27.4%) |
| Train / Val / Test | 70% / 15% / 15% (10 220 / 2 190 / 2 190) |

**Категории проблем:** baggage, booking, delay, in-flight, check-in, customer_service, other

**Авто-разметка категорий:** Rule-based система на основе ключевых слов (6 словарей, ~50 ключевых слов)  
**Авто-разметка критичности:** На основе тональности + наличие urgent-слов (worst, terrible, lawyer, unsafe и др.)

## 5. Метрики качества моделей

### 5.1 Baseline (TF-IDF + LogisticRegression) — Test Set

**Sentiment Analysis:**

| Класс | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| negative | 1.0000 | 1.0000 | 1.0000 | 960 |
| neutral | 1.0000 | 1.0000 | 1.0000 | 600 |
| positive | 1.0000 | 1.0000 | 1.0000 | 630 |
| **Accuracy** | | | **1.0000** | **2 190** |
| **Macro avg** | **1.0000** | **1.0000** | **1.0000** | **2 190** |

**Category Classification:**

| Класс | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| baggage | 0.9697 | 1.0000 | 0.9846 | 96 |
| booking | 0.9962 | 1.0000 | 0.9981 | 528 |
| delay | 0.9885 | 0.9451 | 0.9663 | 273 |
| in-flight | 0.9941 | 0.9805 | 0.9873 | 514 |
| check-in | 0.9850 | 0.9292 | 0.9563 | 353 |
| customer_service | 0.9315 | 0.9855 | 0.9577 | 207 |
| other | 0.9087 | 1.0000 | 0.9522 | 219 |
| **Accuracy** | | | **0.9758** | **2 190** |
| **Macro avg** | **0.9677** | **0.9772** | **0.9718** | **2 190** |

**Criticality Classification:**

| Класс | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| low | 1.0000 | 1.0000 | 1.0000 | 1 230 |
| medium | 1.0000 | 0.9989 | 0.9995 | 914 |
| high | 0.9400 | 1.0000 | 0.9691 | 46 |
| **Accuracy** | | | **0.9986** | **2 190** |
| **Macro avg** | **0.9800** | **0.9996** | **0.9891** | **2 190** |

### 5.2 DistilBERT (fine-tuned) — демонстрационное тестирование

Для оценки BERT на сложных кейсах проведено демонстрационное тестирование на 8 контрольных примерах:

| Текст | Sentiment | Category | Criticality | Результат |
|-------|-----------|----------|-------------|-----------|
| "My flight was delayed 5 hours and nobody helped us!" | ✅ negative (98%) | ✅ delay (60%) | ✅ medium (95%) | 3/3 |
| "Amazing crew, best flight ever! Comfortable seats." | ✅ positive (96%) | ✅ in-flight (87%) | ✅ low (94%) | 3/3 |
| "Lost my luggage and customer service hung up on me." | ✅ negative (99%) | ⚠️ booking (69%) | ✅ medium (94%) | 2/3 |
| "Average flight, nothing special but okay for the price." | ✅ neutral (96%) | ⚠️ booking (90%) | ✅ low (94%) | 2/3 |
| "The flight was fast and there were no delays." | ✅ positive (97%) | ✅ delay (93%) | ✅ low (95%) | 3/3 |
| "Check-in took 2 hours. Absolutely unacceptable!" | ✅ negative (97%) | ✅ check-in (47%) | ✅ medium (91%) | 3/3 |
| "Boarding was smooth and crew was very friendly." | ✅ positive (95%) | ✅ in-flight (84%) | ✅ low (96%) | 3/3 |
| "They cancelled my flight without notice. Had to pay $500!" | ✅ negative (99%) | ✅ booking (93%) | ✅ medium (96%) | 3/3 |

**Общий результат на контрольных примерах: 22/24 (92%)**

### 5.3 Ключевое преимущество BERT над Baseline

| Кейс | Baseline | BERT |
|------|----------|------|
| "The flight was fast and there were no delays" | ❌ negative (66%) | ✅ positive (97%) |
| "I can't say I wasn't impressed. Not bad at all." | ❌ negative | ✅ positive (95%) |
| "Not a single issue with the airline." | ⚠️ зависит от контекста | ✅ positive (95%) |

BERT корректно обрабатывает отрицания, двойные отрицания и контекстные зависимости, что критически важно для production-развёртывания.

### 5.4 Известные ограничения

- **Сарказм:** "Oh great, another delay. Thanks so much!" → positive вместо negative
- **Тихий негатив:** мягкие жалобы без сильных слов не всегда определяются как negative
- **Категория:** при нескольких темах в одном отзыве модель выбирает доминирующую

## 6. Структура репозитория

```
Airline-Sentiment-Analysis/
├── src/
│   ├── config.py          # Константы, гиперпараметры
│   ├── data.py            # Загрузка, очистка, авто-разметка
│   ├── baseline.py        # TF-IDF + LogisticRegression
│   └── bert_model.py      # DistilBERT multi-task
├── scripts/
│   ├── train.py           # Главный пайплайн обучения
│   ├── train_bert.sh      # Обучение BERT одной командой
│   └── generate_data.py   # Генератор датасета
├── api/
│   └── server.py          # FastAPI REST API
├── tests/
│   └── test_pipeline.py   # 19 тестов (pytest)
├── notebooks/
│   └── train_bert_colab.ipynb  # Colab ноутбук для GPU
├── data/                  # Данные (генерируются автоматически)
├── models/
│   ├── baseline/          # TF-IDF + LogReg (joblib)
│   └── bert/              # DistilBERT (PyTorch, Git LFS)
├── reports/
│   ├── REPORT.md          # Этот отчёт
│   └── metrics.json       # Метрики в JSON
├── setup.sh               # Автоустановка
├── run.sh                 # Запуск API
├── requirements.txt       # Зависимости (точные версии)
├── .gitignore
└── README.md
```

## 7. Запуск проекта

```bash
git clone https://github.com/Sancekez/Airline-Sentiment-Analysis.git
cd Airline-Sentiment-Analysis
chmod +x setup.sh && ./setup.sh
./run.sh    # API → http://localhost:8000/docs
```

## 8. Вычислительные ресурсы

| Этап | Ресурс | Время |
|------|--------|-------|
| Baseline обучение | CPU | ~10 секунд |
| BERT обучение | CPU (Intel Core i9-13900H, 20 потоков) | ~2.5 часа |
| API inference | CPU | <100 мс/запрос |

**Инфраструктура:** Локальная рабочая станция
- **CPU:** 13th Gen Intel® Core™ i9-13900H × 20
- **ОС:** Ubuntu
- **Python:** 3.10+
- **GPU:** Не использовалось (обучение на CPU)

## 9. Тестирование

19 тестов, все проходят успешно (pytest):
- 6 тестов предобработки текста (clean_text, negation handling)
- 6 тестов категоризации (rule-based assignment)
- 4 теста критичности
- 3 теста пайплайна данных

## 10. Future Work

1. **Интеграция реальных датасетов.** Текущая модель обучена на синтетическом корпусе, что ограничивает обобщающую способность. Планируется интеграция реальных датасетов: Twitter US Airline Sentiment (14 640 твитов с ручной разметкой, Kaggle), Skytrax Airline Reviews (41 000+ отзывов со звёздными рейтингами), Airline Quality Dataset (многоязычные отзывы с детальными категориями). Переход на реальные данные позволит повысить качество категоризации за счёт естественного распределения тем, улучшить обработку сарказма и неоднозначных формулировок, а также обеспечить валидацию метрик на данных, приближённых к production-сценарию.

2. **Мультимодальное расширение: анализ голоса (Speech Emotion Recognition).** Перспективным направлением является добавление модуля распознавания эмоций в речевых сигналах на основе архитектуры Bi-LSTM + Attention с извлечением MFCC-признаков. Это позволит анализировать не только текстовые обращения, но и телефонные звонки пассажиров в контактный центр авиакомпании, обеспечивая комплексную оценку эмоционального состояния клиента.

3. **Распознавание сарказма.** Добавление модуля детекции сарказма на основе контрастивного обучения или fine-tuning на специализированных датасетах (iSarcasm, SemEval-2018 Task 3).

4. **Multi-label категоризация.** Переход на multi-label классификацию с sigmoid-активацией вместо softmax для определения всех релевантных категорий в одном отзыве.

5. **Docker-контейнеризация.** Создание Docker-образа для гарантированной воспроизводимости развёртывания и устранения потенциальных конфликтов библиотек.

## 11. Выводы

1. Реализован полный NLP-пайплайн для анализа отзывов авиапассажиров с тремя параллельными задачами классификации
2. Baseline модель (TF-IDF + LogReg): Accuracy 97.6–100%, F1-macro 97.2–100% по всем задачам
3. DistilBERT с multi-task архитектурой корректно обрабатывает сложные лингвистические конструкции (отрицания, двойные отрицания), где baseline ошибается
4. REST API (FastAPI) с эндпоинтами /predict и /predict/batch, Swagger UI документацией
5. Воспроизводимость: проект запускается на Ubuntu 20/22/24 + Python 3.10+ двумя командами
6. 19 unit-тестов, все зависимости зафиксированы с точными версиями

---

*Дата составления: 20 апреля 2026 г.*
