import plotly.graph_objects as go
from fetch_data import scrape_player_data


# Sample data
data = scrape_player_data()

# Create a bar chart
fig = go.Figure(
    data=[
        go.Bar(
            x=data['Player'],  # Players on the x-axis
            y=data['Points'],  # Points on the y-axis
            text=data['Team'],  # Hover text showing the team
            marker_color=['blue', 'green', 'red', 'orange']  # Bar colors
        )
    ]
)

# Customize chart layout
fig.update_layout(
    title='Interactive Player Points Chart',
    xaxis_title='Player',
    yaxis_title='Points',
    template='plotly_white'
)

# Show the chart in the browser
fig.show()

# Save the chart as an HTML file
fig.write_html("interactive_chart.html")
print("Chart saved as interactive_chart.html")
