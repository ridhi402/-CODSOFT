# 🚢 Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using classification models.

## 📊 Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | **83.8%** | 0.785 | **0.870** |
| Random Forest | 81.0% | 0.738 | 0.852 |
| Gradient Boosting | 81.0% | 0.742 | 0.838 |

> ✅ Best Model: **Logistic Regression** with 83.8% accuracy and 0.87 ROC-AUC

---

## 📁 Project Structure

```
task 1/
├── main.py                  # Entry point — runs full pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Downloads & loads dataset
│   ├── preprocess.py        # Feature engineering & cleaning
│   └── train.py             # Model training & evaluation
├── data/
│   └── titanic.csv          # Raw dataset
├── models/
│   ├── best_model.pkl       # Saved best model
│   ├── preprocessor.pkl     # Saved preprocessing pipeline
│   └── metrics.json         # Evaluation metrics
├── visualizations/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importances.png
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python main.py
```

---

## 🔧 Features Used

- **Title** — Extracted from passenger name (Mr, Mrs, Miss, Master, Rare)
- **FamilySize** — SibSp + Parch + 1
- **IsAlone** — Whether passenger is travelling alone
- **HasCabin** — Binary: cabin info available or not
- **AgeBin / FareBin** — Discretized age and fare groups
- **Pclass, Sex, Embarked** — Encoded categorical features

---

## 🧠 Methodology

1. **Data Loading** — Dataset fetched via `kagglehub` (yasserh/titanic-dataset)
2. **Preprocessing** — Missing value imputation, feature engineering, scaling
3. **Training** — 3 models compared with 5-fold cross-validation
4. **Hyperparameter Tuning** — GridSearchCV on Random Forest & Gradient Boosting
5. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC

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
CodSoft Internship — Task 1
