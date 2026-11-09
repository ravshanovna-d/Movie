<div align="center">

# 🎬 Movie Rating Prediction

**Predicting how users rate movies on MovieLens data: EDA, classic ML regressors and SVD collaborative filtering.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-model-EB5E28)
![Surprise](https://img.shields.io/badge/Surprise-SVD-6A5ACD)
![Colab](https://img.shields.io/badge/Google%20Colab-notebook-F9AB00?logo=googlecolab&logoColor=white)

</div>

---

## 📌 Overview

This project studies a **MovieLens-style dataset** of about 640K ratings from 3.8K users on 3.6K movies. The goal is to predict the rating (1 to 5) a user would give a movie. The project compares **feature-based regressors** with **matrix factorization (SVD)**, which is the method behind many recommender systems.

## 🔍 Pipeline

1. **Data preparation**: convert `.dat` files (`ratings`, `users`, `movies`) to CSV and merge them
2. **Exploratory analysis**: rating distribution, ratings per year, user activity, movie popularity, genres
3. **Feature engineering**: time features (year, month, weekday), user and movie attributes, outlier removal
4. **Splits**: random and **temporal** train/validation/test splits
5. **Modeling**: KNN, Decision Tree, Random Forest, XGBoost and SVD
6. **Tuning**: grid search over SVD hyperparameters

## 📊 Results

Validation set:

| Model | RMSE ↓ | MAE ↓ |
|---|---|---|
| KNN Regressor | 1.262 | 1.007 |
| XGBoost | 1.090 | 0.895 |
| Decision Tree | 1.071 | 0.867 |
| Random Forest | 1.050 | 0.854 |
| **SVD (Surprise)** | **0.974** | **0.771** |
| **Tuned SVD** | **0.876** | **0.696** |

Best SVD parameters: `n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.1`

**Takeaway:** collaborative filtering clearly outperforms feature-based regressors. Learned user and movie embeddings capture taste much better than demographics and time features do.

## 🧰 Tech Stack

`pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost` · `scikit-surprise`

## 🚀 Getting Started

```bash
git clone https://github.com/ravshanovna-d/Movie.git
cd Movie
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scikit-surprise
```

Put the MovieLens files (`ratings.dat`, `users.dat`, `movies.dat`) next to the notebook. Then open `Movie.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/), or run the script version:

```bash
python Movie.py
```

## 📁 Files

| File | Description |
|---|---|
| `Movie.ipynb` | Full notebook with EDA, plots and model comparison |
| `Movie.py` | Script export of the notebook |

## 🗺 Roadmap

- [ ] Top-N movie recommendations for a given user
- [ ] Hybrid model (SVD + content features from genres)
- [ ] Small Streamlit demo
