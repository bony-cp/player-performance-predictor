from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Global variables for storing data and model
player_data = pd.DataFrame(columns=["Player", "PastPoints", "TeamStrength", "OpponentRank", "FuturePoints"])
model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/add_data", methods=["POST"])
def add_data():
    global player_data
    player = request.form["player"]
    past_points = int(request.form["past_points"])
    team_strength = int(request.form["team_strength"])
    opponent_rank = int(request.form["opponent_rank"])
    future_points = int(request.form["future_points"])  # This is the target value

    # Add data to the DataFrame
    player_data = pd.concat([
        player_data,
        pd.DataFrame([{
            "Player": player,
            "PastPoints": past_points,
            "TeamStrength": team_strength,
            "OpponentRank": opponent_rank,
            "FuturePoints": future_points
        }])
    ], ignore_index=True)

    print("Current DataFrame:")
    print(player_data)  # Debugging

    return jsonify({"status": "success", "message": "Data added successfully!"})

@app.route("/train_model", methods=["POST"])
def train_model():
    global model
    global player_data

    if player_data.empty:
        return jsonify({"status": "error", "message": "No data available to train the model."})

    # Split data for training
    X = player_data[["PastPoints", "TeamStrength", "OpponentRank"]]
    y = player_data["FuturePoints"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return jsonify({"status": "success", "message": f"Model trained successfully! Mean Squared Error: {mse:.2f}"})

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"status": "error", "message": "Model is not trained yet."})

    past_points = int(request.form["past_points"])
    team_strength = int(request.form["team_strength"])
    opponent_rank = int(request.form["opponent_rank"])

    prediction = model.predict([[past_points, team_strength, opponent_rank]])
    return jsonify({"status": "success", "prediction": prediction[0]})

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
