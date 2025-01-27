from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import os

app = Flask(__name__)

# Global variables for storing data and model
player_data = pd.DataFrame(columns=["Player", "PastPoints", "TeamStrength", "OpponentRank", "FuturePoints"])
model = None

# Load saved data and model at startup
if os.path.exists("player_data.csv"):
    player_data = pd.read_csv("player_data.csv")

if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
else:
    print("No pre-trained model found. Please train the model.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/add_data", methods=["POST"])
def add_data():
    try:
        global player_data
        player = request.form["player"]
        past_points = int(request.form["past_points"])
        team_strength = int(request.form["team_strength"])
        opponent_rank = int(request.form["opponent_rank"])
        future_points = int(request.form["future_points"])

        # Validate input ranges
        if past_points < 0 or team_strength < 1 or team_strength > 10 or opponent_rank < 1 or opponent_rank > 10:
            return jsonify({"status": "error", "message": "Invalid input values. Please ensure all inputs are in the correct range."})

        # Add data to the DataFrame
        new_data = pd.DataFrame([{
            "Player": player,
            "PastPoints": past_points,
            "TeamStrength": team_strength,
            "OpponentRank": opponent_rank,
            "FuturePoints": future_points
        }])
        player_data = pd.concat([player_data, new_data], ignore_index=True)

        # Save the data to a CSV file
        player_data.to_csv("player_data.csv", index=False)

        return jsonify({"status": "success", "message": "Data added successfully!"})
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid data format. Please enter numeric values."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"})

@app.route("/train_model", methods=["POST"])
def train_model():
    global model
    global player_data

    if player_data.empty:
        return jsonify({"status": "error", "message": "No data available to train the model."})

    try:
        # Split data for training
        X = player_data[["PastPoints", "TeamStrength", "OpponentRank"]]
        y = player_data["FuturePoints"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        rf = RandomForestRegressor(random_state=42)

        # Define the parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"]
        }

        # Perform Grid Search
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best model
        model = grid_search.best_estimator_

        # Save the trained model
        joblib.dump(model, "model.pkl")

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        feature_importances = model.feature_importances_
        importance_message = ", ".join(
            [f"{feature}: {importance:.2f}" for feature, importance in zip(X.columns, feature_importances)]
        )

        return jsonify({
            "status": "success",
            "message": f"Model trained successfully! Best Parameters: {grid_search.best_params_}. "
                       f"Mean Squared Error: {mse:.2f}. Feature Importances: {importance_message}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"An error occurred during training: {str(e)}"})

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"status": "error", "message": "Model is not trained yet. Please train the model first."})

    try:
        past_points = int(request.form["past_points"])
        team_strength = int(request.form["team_strength"])
        opponent_rank = int(request.form["opponent_rank"])

        # Validate input ranges
        if past_points < 0 or team_strength < 1 or team_strength > 10 or opponent_rank < 1 or opponent_rank > 10:
            return jsonify({"status": "error", "message": "Invalid input values. Please ensure all inputs are in the correct range."})

        # Make prediction
        prediction = model.predict([[past_points, team_strength, opponent_rank]])
        return jsonify({"status": "success", "prediction": prediction[0]})
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid data format. Please enter numeric values."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"An error occurred during prediction: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
