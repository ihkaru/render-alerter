import pytest
import pandas as pd
import os

# Define the path for our static test data file
TEST_DATA_FILE = os.path.join(os.path.dirname(__file__), 'test_data.csv')

@pytest.fixture(scope="session")
def test_data_provider():
    """
    This fixture provides real historical data for testing by loading it
    from a local 'test_data.csv' file.
    
    It now assumes the file MUST exist. If it doesn't, it will fail
    with an instruction to run the data generation script.
    """
    if not os.path.exists(TEST_DATA_FILE):
        pytest.fail(
            f"Test data file not found at: {TEST_DATA_FILE}\n"
            "Please generate it first by running 'python create_test_data.py' from your project's root directory.",
            pytrace=False # Prevents a long traceback for this clear instruction
        )
    
    df = pd.read_csv(TEST_DATA_FILE, index_col='timestamp', parse_dates=True)
    
    # Ensure timezone is set correctly after loading from CSV
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
        
    return df