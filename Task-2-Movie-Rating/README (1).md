# 🎬 Movie Rating Prediction

A machine learning project that predicts the IMDb rating of Indian movies based on
features like genre, director, cast, duration, votes, and release year.

## 📊 Approach

This is a **regression** task (predicting a continuous rating, e.g. 7.2), unlike
Task 1 which was classification (survive / not survive).

| Model | Metric used |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Non-linear, tuned via GridSearchCV |
| Gradient Boosting Regressor | Tuned via GridSearchCV |

Evaluation metrics: **MAE**, **RMSE**, **R²** (not accuracy/F1, since this isn't classification).

---

## 📁 Project Structure

```
Task-2-Movie-Rating/
├── main.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Downloads dataset via kagglehub
│   ├── preprocess.py      # Cleaning, feature engineering, encoding
│   └── train.py           # Trains models, tunes, evaluates, saves
├── data/
│   └── movies.csv
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
├── visualizations/
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_importances.png
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
pip install -r requirements.txt
python main.py
```

The first run will download the dataset via `kagglehub`
(`adrianmcmahon/imdb-india-movies`) and cache it to `data/movies.csv`.

---

## 🔧 Feature Engineering

- **Year / Duration / Votes** — parsed from raw text (e.g. `"(2019)"` → `2019`, `"109 min"` → `109`)
- **Votes** — log-transformed (`log1p`) since vote counts are heavily right-skewed
- **Primary Genre** — first genre in the comma-separated list, one-hot encoded
- **Director** — frequency-encoded (too many unique values for one-hot encoding)
- **Cast** (Actor 1/2/3 combined) — frequency-encoded the same way

All encoders are fit **only on the training set** to avoid data leakage, then applied
to the test set.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
kagglehub
joblib
```

---

## 👩‍💻 Author

**Ridhi**
CodSoft Internship — Task 2
