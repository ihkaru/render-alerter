import yfinance as yf
import pandas as pd
import os
TEST_DATA_DIR = 'tests'
TEST_DATA_FILE = os.path.join(TEST_DATA_DIR, 'test_data.csv')
def generate_test_data():
    """
    Performs a one-time download of real historical data and saves it
    to a CSV file for use in unit tests.
    """
    print("--- Test Data Generator ---")
    # Ensure the 'tests' directory exists
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Creating directory: {TEST_DATA_DIR}")
        os.makedirs(TEST_DATA_DIR)

    if os.path.exists(TEST_DATA_FILE):
        print(f"Test data file already exists at: {TEST_DATA_FILE}")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower()
        if overwrite != 'y':
            print("Operation cancelled. Using existing file.")
            return

    print("Fetching real market data for NVDA from yfinance...")
    # Fetch a known volatile period for NVDA to ensure signals are generated
    try:
        df = yf.download(
            tickers="NVDA",
            start="2023-05-01",
            end="2023-08-01",
            interval="1h",
            progress=True,  # Show progress for manual run
            auto_adjust=True
        )
        if df.empty:
            print("\nERROR: Failed to download any data. The ticker might be wrong or yfinance may be blocked.")
            return

        # Standardize column names and index
        df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        df.index.name = 'timestamp'

        print(f"\nSuccessfully downloaded {len(df)} rows.")
        print(f"Saving new test data to: {TEST_DATA_FILE}")
        df.to_csv(TEST_DATA_FILE)
        print("\n--- Success! Test data file has been created. ---")

    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please check your network connection and try again.")

if __name__ == "__main__":
    generate_test_data()