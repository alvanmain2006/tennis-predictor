
import streamlit as st
import pickle
import pandas as pd
import os

MODEL_PATH = "models/logistic_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    return model, feature_columns


def make_features(player1_rank, player2_rank, surface):
    features = {
        "player_rank": player1_rank,
        "opponent_rank": player2_rank,
        "rank_diff": player1_rank - player2_rank,
        "is_higher_ranked": 1 if player1_rank < player2_rank else 0,
        "rank_ratio": player1_rank / (player2_rank + 1),

        "player_form": 0.6,
        "player_surface_form": 0.6,
        "h2h_win_rate": 0.5,
        "rest_days": 7,
        "matches_last_30_days": 4,
        "ranking_momentum": 0,
        "tournament_win_rate": 0.5,
        "tournament_matches": 5,
        "player_experience": 100,

        "surface_Hard": 1 if surface == "Hard" else 0,
        "surface_Clay": 1 if surface == "Clay" else 0,
        "surface_Grass": 1 if surface == "Grass" else 0,
    }

    return features


st.title("Tennis Match Predictor")

st.write("Enter two players' rankings and choose a surface.")

player1_name = st.text_input("Player 1 name", "Player 1")
player2_name = st.text_input("Player 2 name", "Player 2")

player1_rank = st.number_input("Player 1 ranking", min_value=1, max_value=1000, value=10)
player2_rank = st.number_input("Player 2 ranking", min_value=1, max_value=1000, value=20)

surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])

if st.button("Predict winner"):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Missing model file: {MODEL_PATH}")
    elif not os.path.exists(FEATURES_PATH):
        st.error(f"Missing feature file: {FEATURES_PATH}")
    else:
        model, feature_columns = load_model()

        features = make_features(player1_rank, player2_rank, surface)

        input_df = pd.DataFrame([features])

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        probability = model.predict_proba(input_df)[0, 1]
        prediction = model.predict(input_df)[0]

        player1_prob = probability
        player2_prob = 1 - probability

        st.subheader("Prediction")

        st.write(f"{player1_name} win probability: **{player1_prob:.1%}**")
        st.write(f"{player2_name} win probability: **{player2_prob:.1%}**")

        winner = player1_name if prediction == 1 else player2_name
        st.success(f"Predicted winner: {winner}")