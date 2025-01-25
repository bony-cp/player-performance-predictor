import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# Lists to store data
historical_data = []  # Stores historical stats for ML training
players = []
predictions = []

# Function to add historical data for a player
def add_historical_data():
    player_name = entry_historical_player.get()
    past_points = entry_past_points.get()
    team_strength = entry_team_strength.get()
    opponent_rank = entry_opponent_rank.get()

    if not player_name or not past_points.isdigit() or not team_strength.isdigit() or not opponent_rank.isdigit():
        messagebox.showerror("Error", "Please fill out all fields correctly.")
        return

    historical_data.append({
        "Player": player_name,
        "PastPoints": int(past_points),
        "TeamStrength": int(team_strength),
        "OpponentRank": int(opponent_rank),
    })

    entry_historical_player.delete(0, tk.END)
    entry_past_points.delete(0, tk.END)
    entry_team_strength.delete(0, tk.END)
    entry_opponent_rank.delete(0, tk.END)

    messagebox.showinfo("Success", f"Historical data for {player_name} added!")

# Train the machine learning model
def train_model():
    global model

    if not historical_data:
        messagebox.showerror("Error", "No historical data available for training.")
        return

    # Convert historical data to a DataFrame
    df = pd.DataFrame(historical_data)

    # Prepare features and target
    X = df[["PastPoints", "TeamStrength", "OpponentRank"]]
    y = df["PastPoints"]  # Use PastPoints as both input and proxy target for simplicity

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    messagebox.showinfo("Model Trained", f"Model trained successfully! Mean Squared Error: {mse:.2f}")

# Predict future performance
def add_player_and_predict():
    global model  # Use the trained model

    if 'model' not in globals():
        messagebox.showerror("Error", "Please train the model first!")
        return

    player_name = entry_player.get()
    past_points = entry_past_points_new.get()
    team_strength = entry_team_strength_new.get()
    opponent_rank = entry_opponent_rank_new.get()

    if not player_name or not past_points.isdigit() or not team_strength.isdigit() or not opponent_rank.isdigit():
        messagebox.showerror("Error", "Please fill out all fields correctly.")
        return

    # Predict performance
    features = [[int(past_points), int(team_strength), int(opponent_rank)]]
    predicted_points = model.predict(features)[0]

    players.append(player_name)
    predictions.append(predicted_points)

    entry_player.delete(0, tk.END)
    entry_past_points_new.delete(0, tk.END)
    entry_team_strength_new.delete(0, tk.END)
    entry_opponent_rank_new.delete(0, tk.END)

    messagebox.showinfo("Success", f"Prediction for {player_name}: {predicted_points:.2f} points")

# Create a performance chart
def create_chart():
    if not players:
        messagebox.showerror("Error", "No players added for predictions.")
        return

    fig = go.Figure(data=[go.Bar(x=players, y=predictions)])
    fig.update_layout(title="Predicted Player Performance", xaxis_title="Player", yaxis_title="Predicted Points")
    fig.show()

# Tkinter GUI
root = tk.Tk()
root.title("Player Performance Predictor")

# Historical Data Input
tk.Label(root, text="Add Historical Data").grid(row=0, column=0, columnspan=2, pady=10)

tk.Label(root, text="Player Name:").grid(row=1, column=0)
entry_historical_player = tk.Entry(root)
entry_historical_player.grid(row=1, column=1)

tk.Label(root, text="Past Points:").grid(row=2, column=0)
entry_past_points = tk.Entry(root)
entry_past_points.grid(row=2, column=1)

tk.Label(root, text="Team Strength (1-10):").grid(row=3, column=0)
entry_team_strength = tk.Entry(root)
entry_team_strength.grid(row=3, column=1)

tk.Label(root, text="Opponent Rank (1-10):").grid(row=4, column=0)
entry_opponent_rank = tk.Entry(root)
entry_opponent_rank.grid(row=4, column=1)

btn_add_historical = tk.Button(root, text="Add Historical Data", command=add_historical_data)
btn_add_historical.grid(row=5, column=0, columnspan=2, pady=10)

# Train Model Button
btn_train = tk.Button(root, text="Train Model", command=train_model)
btn_train.grid(row=6, column=0, columnspan=2, pady=10)

# New Player Prediction Input
tk.Label(root, text="Predict Future Performance").grid(row=7, column=0, columnspan=2, pady=10)

tk.Label(root, text="Player Name:").grid(row=8, column=0)
entry_player = tk.Entry(root)
entry_player.grid(row=8, column=1)

tk.Label(root, text="Past Points:").grid(row=9, column=0)
entry_past_points_new = tk.Entry(root)
entry_past_points_new.grid(row=9, column=1)

tk.Label(root, text="Team Strength (1-10):").grid(row=10, column=0)
entry_team_strength_new = tk.Entry(root)
entry_team_strength_new.grid(row=10, column=1)

tk.Label(root, text="Opponent Rank (1-10):").grid(row=11, column=0)
entry_opponent_rank_new = tk.Entry(root)
entry_opponent_rank_new.grid(row=11, column=1)

btn_predict = tk.Button(root, text="Add Player & Predict", command=add_player_and_predict)
btn_predict.grid(row=12, column=0, columnspan=2, pady=10)

# Create Chart Button
btn_chart = tk.Button(root, text="Create Chart", command=create_chart)
btn_chart.grid(row=13, column=0, columnspan=2, pady=10)

# Start GUI
root.mainloop()
