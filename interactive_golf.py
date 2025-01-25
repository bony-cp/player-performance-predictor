import plotly.graph_objects as go

def get_player_stats():
    players = []
    points = []
    teams = []

    while True:
        # Collect player data
        print("\nEnter player stats (leave Player name empty to finish):")
        player_name = input("Player Name: ")
        if player_name == "":
            break

        try:
            player_points = int(input("Points: "))
        except ValueError:
            print("Points must be a number! Please try again.")
            continue

        team_name = input("Team: ")

        # Append the entered data to the lists
        players.append(player_name)
        points.append(player_points)
        teams.append(team_name)

    return {"Player": players, "Points": points, "Team": teams}

def create_chart(data):
    # Create a bar chart with user-entered data
    fig = go.Figure(
        data=[
            go.Bar(
                x=data["Player"],  # Players on the x-axis
                y=data["Points"],  # Points on the y-axis
                text=data["Team"],  # Hover text showing the team
                marker_color=["blue", "green", "red", "orange"][: len(data["Player"])]
            )
        ]
    )

    # Customize chart layout
    fig.update_layout(
        title="Interactive Player Points Chart",
        xaxis_title="Player",
        yaxis_title="Points",
        template="plotly_white"
    )

    # Show the chart
    fig.show()

    # Save the chart as an HTML file
    fig.write_html("interactive_chart.html")
    print("Chart saved as interactive_chart.html")

if __name__ == "__main__":
    # Collect player stats interactively
    data = get_player_stats()

    if data["Player"]:
        # Create and display the chart
        create_chart(data)
    else:
        print("No data entered. Exiting.")
