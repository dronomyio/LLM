import mysql.connector
import json
from datetime import date

# Database credentials
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "root"
DB_NAME = "SPYStocks"

# Establish a connection to MySQL
def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )

# Fetch companies from the Companies table
def fetch_companies(cursor):
    query = """
    SELECT id, symbol, security, gics_sector, gics_sub_industry, headquarters_location 
    FROM Companies;
    """
    cursor.execute(query)
    return cursor.fetchall()

# Fetch stock prices from the StockPrices table
def fetch_stock_prices(cursor):
    query = """
    SELECT Ticker, Date, Price, AdjClose, Close, High, Low, Open, Volume 
    FROM StockPrices;
    """
    cursor.execute(query)
    return cursor.fetchall()

# Generate relationships for companies and industries
def fetch_industry_relationships(companies):
    relationships = []
    for company in companies:
        relationships.append({"symbol": company[1], "industry": company[3]})
    return relationships

# Generate relationships for companies and sub-industries
def fetch_subindustry_relationships(companies):
    relationships = []
    for company in companies:
        relationships.append({"symbol": company[1], "subIndustry": company[4]})
    return relationships

# Main logic to generate the :params JSON object
def generate_params():
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Fetch data
        companies = fetch_companies(cursor)
        stock_prices = fetch_stock_prices(cursor)

        # Generate companies JSON
        companies_json = [
            {
                "id": company[0],
                "symbol": company[1],
                "security": company[2],
                "gics_sector": company[3],
                "gics_sub_industry": company[4],
                "headquarters_location": company[5],
            }
            for company in companies
        ]

        # Generate industries JSON
        industries_json = [
            {"name": company[3]} for company in companies
        ]
        industries_json = list({v['name']: v for v in industries_json}.values())  # Remove duplicates

        # Generate sub-industries JSON
        subindustries_json = [
            {"name": company[4]} for company in companies
        ]
        subindustries_json = list({v['name']: v for v in subindustries_json}.values())  # Remove duplicates

        # Generate stock prices JSON
        stock_prices_json = [
            {
                "ticker": price[0],
                "date": price[1].strftime("%Y-%m-%d"),  # Format date
                "price": float(price[2]),
                "adj_close": float(price[3]) if price[3] is not None else None,
                "close": float(price[4]),
                "high": float(price[5]),
                "low": float(price[6]),
                "open": float(price[7]),
                "volume": int(price[8]),
            }
            for price in stock_prices
        ]

        # Generate relationships
        company_industry_relationships = fetch_industry_relationships(companies)
        company_subindustry_relationships = fetch_subindustry_relationships(companies)
        company_stock_price_relationships = [
            {"symbol": price[0], "ticker": price[0], "date": price[1].strftime("%Y-%m-%d")}
            for price in stock_prices
        ]

        # Combine into final :params structure
        params = {
            "companies": companies_json,
            "industries": industries_json,
            "subIndustries": subindustries_json,
            "companyIndustryRelationships": company_industry_relationships,
            "companySubIndustryRelationships": company_subindustry_relationships,
            "stockPrices": stock_prices_json,
            "companyStockPriceRelationships": company_stock_price_relationships,
        }

        # Dump JSON into a file
        with open("params.json", "w") as file:
            json.dump(params, file, indent=2)
        print("JSON dumped to params.json")

    finally:
        cursor.close()
        conn.close()

# Run the script
if __name__ == "__main__":
    generate_params()

