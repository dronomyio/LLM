import mysql.connector
import yfinance as yf

# Database credentials
DB_HOST = "localhost"  # Replace with your MySQL server host, e.g., "localhost"
DB_USER = "root"  # Replace with your MySQL username
DB_PASSWORD = "root"  # Replace with your MySQL password
DB_NAME = "SPYStocks"  # Name of your database

# Establish a connection to MySQL
def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )

# Fetch Yahoo Finance stock data
def fetch_yahoo_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist

# Insert data into the database
def insert_stock_data(cursor, ticker, stock_data):
    insert_query = """
    INSERT INTO StockPrices 
    (Ticker, Date, Price, AdjClose, Close, High, Low, Open, Volume) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    Price = VALUES(Price), AdjClose = VALUES(AdjClose), Close = VALUES(Close), 
    High = VALUES(High), Low = VALUES(Low), Open = VALUES(Open), Volume = VALUES(Volume);
    """
    for index, row in stock_data.iterrows():
        # Handle missing 'Adj Close' column
        adj_close = row['Adj Close'] if 'Adj Close' in row else None
        
        try:
            cursor.execute(insert_query, (
                ticker,
                index.date(),
                row['Close'],  # Price as the closing price
                adj_close,
                row['Close'],
                row['High'],
                row['Low'],
                row['Open'],
                int(row['Volume'])
            ))
        except mysql.connector.Error as e:
            print(f"Error inserting data for {ticker} on {index.date()}: {e}")

# Main logic to fetch and store data
def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # List of tickers
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # Establish a connection to the database
    conn = get_connection()
    cursor = conn.cursor()

    try:
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            stock_data = fetch_yahoo_stock_data(ticker, start_date, end_date)

            if not stock_data.empty:
                print(f"Inserting data for {ticker} into the database...")
                insert_stock_data(cursor, ticker, stock_data)
                conn.commit()
                print(f"Data for {ticker} inserted successfully.")
            else:
                print(f"No data found for {ticker}.")
    finally:
        cursor.close()
        conn.close()

# Run the script
if __name__ == "__main__":
    main()

