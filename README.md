# Tennis Match Outcome Predictor

A machine learning project that predicts professional tennis match outcomes using historical ATP/WTA match data, engineered player statistics, and a scikit-learn prediction model.

The project includes a machine learning pipeline for preprocessing, training, and serving match predictions through a web interface hosting service from Streamlit.

---

## Features

- Predicts tennis match winners based on player and match statistics
- Uses historical ATP/WTA match data for model training
- Engineers player-based features such as ranking, recent form, win percentage, surface performance, and head-to-head indicators
- Provides a web interface where users can input two players and receive a prediction
- Designed for deployment on platform with Streamlit


## Project Structure

```text
tennis-predictor/
│
│
├── src/
│   ├── train_model.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   └── saved_models/
│
├── data/
│   ├── raw/
│   └── processed/
│
│
├── .env.example
├── README.md
└── requirements.txt