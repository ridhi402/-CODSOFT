# 🌸 Iris Flower Classification

A machine learning project that classifies iris flowers into one of three species
(*setosa*, *versicolor*, *virginica*) based on sepal/petal length and width.

## 📊 Results (on sample test run)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | **93.3%** | 0.933 | 0.933 | 0.933 |
| SVM | **93.3%** | 0.933 | 0.933 | 0.933 |
| Random Forest | 90.0% | 0.902 | 0.900 | 0.900 |

> Actual numbers will vary slightly depending on the Kaggle dataset's exact rows.

---

## 📁 Project Structure

```
Task-3-Iris-Classification/
├── main.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Downloads dataset via kagglehub
│   ├── preprocess.py      # Encoding, scaling, train/test split
│   └── train.py           # Trains, tunes, evaluates, saves models
├── data/
│   └── iris.csv
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
├── visualizations/
│   ├── confusion_matrix.png
│   ├── feature_importances.png
│   └── feature_pairplot.png
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
pip install -r requirements.txt
python main.py
```

The first run downloads the dataset via `kagglehub` (`arshid/iris-flower-dataset`)
and caches it to `data/iris.csv`.

---

## 🧠 Methodology

1. **Data Loading** — fetched via kagglehub, validated for shape/columns
2. **Preprocessing** — label-encode species, scale 4 numeric features with `StandardScaler`
3. **Models compared**:
   - Logistic Regression (baseline)
   - SVM (tuned via GridSearchCV — kernel & C)
   - Random Forest (tuned via GridSearchCV — depth & estimators)
4. **Evaluation** — Accuracy, Precision, Recall, F1 (macro-averaged for multi-class)
5. **Visualizations** — confusion matrix, feature importances, pairplot of features by species

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
CodSoft Internship — Task 3
