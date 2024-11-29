import ssl
import pandas as pd
import mysql.connector
import logging
from urllib.request import urlopen

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MySQL connection details
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "root"
DB_NAME = "SPYStocks"

# Function to get S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    """Retrieve the list of S&P 500 companies from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Create an SSL context to ignore certificate errors
    ssl_context = ssl._create_unverified_context()
    #sp500_table = pd.read_html(url)[0]
    sp500_table = pd.read_html(urlopen(url, context=ssl_context))[0]
    sp500_table.rename(
        columns={
            "Symbol": "symbol",
            "Security": "security",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "Headquarters Location": "headquarters_location",
        },
        inplace=True,
    )
    sp500_table = sp500_table[["symbol", "security", "gics_sector", "gics_sub_industry", "headquarters_location"]]
    sp500_table.dropna(inplace=True)  # Clean missing data
    return sp500_table

# Function to insert data into the MySQL database
def insert_data_into_mysql(data):
    """Insert the S&P 500 company data into the Companies table."""
    conn = None  # Initialize connection variable
    try:
        # Establish a connection to MySQL
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
        )
        cursor = conn.cursor()

        # SQL query to insert data
        insert_query = """
        INSERT INTO Companies (symbol, security, gics_sector, gics_sub_industry, headquarters_location)
        VALUES (%s, %s, %s, %s, %s)
        """
        for _, row in data.iterrows():
            print(row)
            try:
                cursor.execute(insert_query, tuple(row))
            except mysql.connector.Error as e:
                logging.error(f"Failed to insert row {row}: {e}")

        # Commit the transaction
        conn.commit()
        logging.info(f"Inserted {cursor.rowcount} rows into the Companies table.")

    except mysql.connector.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        try:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
        except mysql.connector.Error as e:
            logging.error(f"Error closing connection: {e}")


# Main execution
if __name__ == "__main__":

    # Fetch S&P 500 company data
    sp500_data = get_sp500_tickers()
    logging.info(f"Fetched {len(sp500_data)} rows of data.")
    print("xx")
    print(sp500_data.head())  # Debug: Print first few rows

    # Insert data into MySQL
    insert_data_into_mysql(sp500_data)
