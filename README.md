# House Price Prediction (Machine Learning)

A beginner-friendly machine learning project that predicts house prices using the
[California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).
It trains and compares four regression models, saves the best one, and lets you
predict the price of a new house from the command line.

## Features
- Automatic dataset download (scikit-learn, no manual CSV needed)
- Simple feature engineering (`RoomsPerPerson`, `BedroomRatio`)
- Model comparison with 5-fold cross-validation:
  Linear Regression, Ridge, Random Forest, Gradient Boosting
- Metrics: MAE, RMSE, R²
- Plots: actual vs predicted, feature importance
- CLI prediction script

## Project structure
```
house-price-prediction/
├── train.py            # train, compare, evaluate, save best model
├── predict.py          # predict price for a new house
├── requirements.txt
├── models/             # saved model (created by train.py)
└── outputs/            # metrics.json and plots (created by train.py)
```

## Getting started
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train
```bash
python train.py
```
This prints a comparison table and writes `models/best_model.joblib`,
`outputs/metrics.json` and the plots in `outputs/`.

### Predict
```bash
python predict.py --medinc 5.2 --houseage 20 --averooms 6 --avebedrms 1.1 \
    --population 1500 --aveoccup 3 --latitude 34.05 --longitude -118.25
```

Inputs follow the dataset's columns (`MedInc` is in tens of thousands of dollars).
Predictions are returned in US dollars.

## Results
Run `python train.py` to see the numbers on your machine; typical R² for the
tree-based models on this dataset is around 0.80 or higher.

## Ideas to extend
- Hyperparameter tuning with `GridSearchCV` / `Optuna`
- Try XGBoost or LightGBM
- Build a Streamlit or Flask web app around `predict.py`
- Use your own dataset (e.g. Kaggle's Ames or Indian housing data)

## License
MIT
