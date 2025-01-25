import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go

# Function to collect data from the form
def add_player():
    player_name = entry_player.get()
    player_points = entry_points.get()
    team_name = entry_team.get()

    # Validation
    if not player_name or not player_points.isdigit() or not team_name:
        messagebox.showerror("Error", "Please fill out all fields correctly.")
        return

    players.append(player_name)
    points.append(int(player_points))
    teams.append(team_name)

    # Clear fields for next input
    entry_player.delete(0, tk.END)
    entry_points.delete(0, tk.END)
    entry_team.delete(0, tk.END)

    messagebox.showinfo("Success", f"Player {player_name} added!")

# Function to finish data entry and create the chart
def finish_and_create_chart():
    if not players:
        messagebox.showerror("Error", "No data entered.")
        return

    # Create a bar chart
    data = {"Player": players, "Points": points, "Team": teams}
    fig = go.Figure(
        data=[
            go.Bar(
                x=data["Player"],
                y=data["Points"],
                text=data["Team"],
                marker_color=["blue", "green", "red", "orange"][: len(data["Player"])]
            )
        ]
    )

    fig.update_layout(
        title="Interactive Player Points Chart",
        xaxis_title="Player",
        yaxis_title="Points",
        template="plotly_white"
    )

    # Display the chart
    fig.show()

    # Save the chart as HTML
    fig.write_html("interactive_chart.html")
    print("Chart saved as interactive_chart.html")
    root.destroy()

# Lists to store the data
players = []
points = []
teams = []

# Create the main Tkinter window
root = tk.Tk()
root.title("Enter Player Stats")

# Labels and entry fields
tk.Label(root, text="Player Name:").grid(row=0, column=0, padx=10, pady=10)
entry_player = tk.Entry(root, width=30)
entry_player.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Points:").grid(row=1, column=0, padx=10, pady=10)
entry_points = tk.Entry(root, width=30)
entry_points.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Team:").grid(row=2, column=0, padx=10, pady=10)
entry_team = tk.Entry(root, width=30)
entry_team.grid(row=2, column=1, padx=10, pady=10)

# Buttons
btn_add = tk.Button(root, text="Add Player", command=add_player)
btn_add.grid(row=3, column=0, padx=10, pady=10)

btn_finish = tk.Button(root, text="Finish & Create Chart", command=finish_and_create_chart)
btn_finish.grid(row=3, column=1, padx=10, pady=10)

# Start the Tkinter main loop
root.mainloop()