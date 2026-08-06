# Telecom Churn — MLOps Pipeline

Модель предсказывает, какие клиенты телеком-компании уйдут (churn). Проект покрывает
полный жизненный цикл: обучение с трекингом экспериментов (MLflow), сервинг через API
(FastAPI), веб-интерфейс (Streamlit), автоматические тесты и CI/CD (GitHub Actions).

## Датасет

IBM Telco Customer Churn — 7 043 реальных клиента, 21 признак (`data/telco_churn.csv`).

## Результаты

- Лучшая модель: **Logistic Regression**, ROC-AUC = **0.8416** (лучше RandomForest: 0.8220)
- Пример высокого риска (месячный контракт, fiber optic, короткий tenure): ~90% вероятность оттока
- Пример низкого риска (2-летний контракт, долгий tenure): ~15% вероятность оттока

## Живые ссылки

- **API (FastAPI, Render):** https://churn-mlops-tubl.onrender.com
  - `GET /health` — проверка статуса
  - `POST /predict` — предсказание оттока для одного клиента
  - `/docs` — автодокументация Swagger
- **Веб-форма (Streamlit Cloud):** https://rosychks-churn-mlops-dashboard-3ejqrv.streamlit.app

*Бесплатный тариф Render "засыпает" при отсутствии активности — первый запрос
после паузы может занять до ~50 секунд.*

## Структура проекта
churn-mlops/
├─ data/telco_churn.csv датасет (7 043 клиента)
├─ src/
│ ├─ preprocess.py подготовка данных (общая для train и serve)
│ └─ train.py обучение, трекинг в MLflow, сохранение лучшей модели
├─ api/
│ ├─ schema.py типизированная схема входных данных
│ └─ main.py FastAPI: /predict, /health
├─ dashboard.py Streamlit веб-приложение
├─ tests/test_pipeline.py quality gate для CI
├─ models/model.pkl обученная модель
├─ Dockerfile
├─ requirements.txt
└─ .github/workflows/
├─ ci.yml тесты на каждый push
└─ retrain.yml переобучение раз в неделю (+ вручную)
## Запуск локально

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 1. Обучение (сравнивает 2 модели в MLflow, сохраняет лучшую)
python src/train.py
mlflow ui        # http://127.0.0.1:5000

# 2. API
uvicorn main:app --app-dir api --reload      # http://127.0.0.1:8000/docs

# 3. Веб-форма (в отдельном терминале, API должен работать)
streamlit run dashboard.py                    # http://127.0.0.1:8501

# 4. Тесты
pytest -v
```

## CI/CD

- **`.github/workflows/ci.yml`** — прогоняет `pytest` на каждый push/PR в `main`.
  Если ROC-AUC падает ниже 0.78 — тест красный, изменение блокируется.
- **`.github/workflows/retrain.yml`** — переобучает модель каждый понедельник в 03:00 UTC
  (или вручную через вкладку Actions → Retrain → Run workflow) и коммитит обновлённый
  `model.pkl`, если он изменился.

## Деплой

- API задеплоен на **Render** через `Dockerfile` — автодеплой при каждом push в `main`.
- Веб-форма задеплоена на **Streamlit Community Cloud**, переменная окружения
  `API_URL` указывает на Render-адрес API.

## Ключевые технические решения

- **Дисбаланс классов** — только ~26% клиентов уходят, поэтому используется
  `class_weight="balanced"` и метрика ROC-AUC вместо accuracy.
- **Без train/serve skew** — один и тот же pipeline (`preprocess.py`) используется
  и при обучении, и внутри `model.pkl`, который грузит API.
- **Трекинг экспериментов** — каждый запуск обучения логируется в MLflow с метриками
  и параметрами, что позволяет объективно сравнивать модели.
- **Quality gate** — тест в CI не даёт задеплоить модель хуже базовой линии.