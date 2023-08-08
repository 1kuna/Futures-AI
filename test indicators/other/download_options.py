import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_polygon_options_data(api_key, ticker="/ES", days_back=60, granularity="minute"):
    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Convert dates to string format for API request
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Base URL for the Polygon.io API
    base_url = f"https://api.polygon.io/v3/reference/tickers/types?asset_class=options&locale=us&apiKey=hUUJkP0egCdBMJNUajP_I9yM9T3Mxneb"
    
    # Parameters for the request
    params = {
        "apiKey": api_key
    }
    
    # Make the API request
    response = requests.get(base_url, params=params)
    data = response.json()

    print(response)
    
    # # Extract and save the data to a CSV file
    # df = pd.DataFrame(data["results"])
    # df.to_csv(f"K:\\Git\\KUNA\\test indicators\\{ticker}_options_data.csv", index=False)
    
    # print(f"Data saved to {ticker}_options_data.csv")

# Example usage
api_key = ""
fetch_polygon_options_data(api_key)
