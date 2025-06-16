# Medical Insurance Price Prediction (Flask Deployment)

This project predicts medical insurance costs based on user input such as age, gender, BMI, number of children, smoking status, and region. The model is trained using regression algorithms and deployed via a Flask web app.



Features

- Predicts insurance charges using trained ML models
- User-friendly interface built with HTML & CSS
- Deployed locally via Flask framework
- Includes custom feature engineering and outlier handling


Machine Learning

Dataset:
- Source: [Kaggle Insurance Dataset](https://www.kaggle.com/mirichoi0218/insurance)
- Features used:
  - `age`
  - `sex`
  - `bmi`
  - `children`
  - `smoker`
  - `region`
  - `charges` (target)

Final Model:
- **Random Forest Regressor**
- Log-transformed target variable
- Best performance among tested models (SVR, XGBoost, CatBoost)


Tools & Technologies

- Python (Pandas, NumPy, scikit-learn, joblib)
- Flask
- HTML, CSS
- VS Code
- Git & GitHub



