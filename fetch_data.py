import requests

def fetch_golf_data():
    # Replace this with the endpoint you're targeting
    url = "https://replay.sportsdata.io/api/v3/nfl/pbp/json/playbyplaydelta/2023reg/2/all?key="
    api_key = "004f9947dba84e29a86a8fa1160a680c"  # Replay API key

    # Set headers
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    # Make the API request
    response = requests.get(url, headers=headers)

    # Check the response status
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Test the function
if __name__ == "__main__":
    golf_data = fetch_golf_data()
    if golf_data:
        print(golf_data)  # Print or process the data
