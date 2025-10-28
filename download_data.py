"""
download_data.py

Handles the collection of financial data for 10 specified stock tickers.
- Fetches company CIKs (Central Index Keys) from SEC.
- Downloads recent SEC filings (10-K, 10-Q, 8-K).
- Retrieves historical stock data (daily, hourly) and recommendations from yfinance.
- Fetches recent news headlines via NewsAPI and scrapes article/header text

Designed to be run once a day to gather raw data for subsequent processing.
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from datetime import timedelta
from dotenv import load_dotenv


#############################################
# --- Configuration ---
#############################################

load_dotenv() # load environment variables from .env file

# SEC EDGAR database requires a declared user agent when sending requests
# see: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data  
HEADERS = {'User-Agent': 'duenasfd@gmail.com'}


# Initial list of tickers for data collection (a lot of memory required). 
# Can be expanded or made dynamic.
TICKERS = ["AAPL", "META", "NVDA", "GOOGL", "MSFT",
           "AMZN", "AVGO", "JPM", "NFLX", "ORCL"]

# FUTURE IMPLEMENTATION 
# All company tickers w/ filings from SEC
# companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json", headers = headers)


# API Key retrieval
# Note: NewsAPI free tier limits requests, restricting the number of tickers for daily news pulls.
NEWSAPI_KEY = os.getenv("NEWS_API_KEY")
if not NEWSAPI_KEY:
    print("❌ FATAL ERROR: NEWS_API_KEY not found in .env file or environment variables.")
    sys.exit("Script terminated: Missing NewsAPI Key.")

NEWSAPI_CLIENT = NewsApiClient(api_key = NEWSAPI_KEY)



#############################################
# --- Functions for Data Retrieval ---
#############################################


def get_company_ciks(tickers, headers):
    """
    Retrieves Central Index Keys (CIKs) from the SEC for an input list of stock tickers.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
        headers (dict): HTTP headers to use for the request, including User-Agent.

    Returns:
        dict: A dictionary mapping tickers to their CIKs (zero-padded string),
              or None if not found.
    """
    print("Fetching company CIKs...")
    try: 
        company_tickers_response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
        company_tickers_response.raise_for_status()    # Raise HTTPError for bad responses
        company_data = company_tickers_response.json()
        
        # Create a mapping from ticker to CIK for easy lookup
        ticker_to_cik = {item['ticker']: str(item['cik_str']).zfill(10) for item in company_data.values()}
        
        ciks = {ticker: ticker_to_cik.get(ticker) for ticker in tickers}
        print("CIKs fetched successfully.")
        return ciks
    except Exception as e:
        print(f"❌ Error fetching CIKs: {e}")
        return {ticker: None for ticker in tickers}


def download_sec_filings(ticker, cik, headers):
    """
    Downloads recent 10-K, 10-Q, and 8-K filings directly from SEC EDGAR database.
    Saves filings as HTML files in a structured directory.
    Note: This currently downloads all filings listed in the 'recent' endpoint
          without checking for existing files. For incremental updates, add a
          check like `if os.path.exists(save_path): continue`.

    Args:
        ticker (str): Stock ticker symbol.
        cik (str): Company CIK (10-digit zero-padded string).
        headers (dict): HTTP headers for requests.
    """
    print(f"\n--- Fetching filings for {ticker} ---")
    # Get the list of all recent filings
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    submissions_response = requests.get(submissions_url, headers=headers)
    submissions_response.raise_for_status()
    filings = submissions_response.json()['filings']['recent']
    
    # Check if 'filings' or 'recent' keys exist/are not empty
    if not filings:
            print(f"    🟡 No 'recent' filings found in submissions JSON for {ticker}.")
            return
    
    filings_df = pd.DataFrame(filings)

    if filings_df.empty:
             print(f"    🟡 DataFrame creation from filings JSON failed or was empty for {ticker}.")
             return
    
    # Filter for only the reports we want, could be expanded for more reports later
    target_forms = ["10-K", "10-Q", "8-K"]
    # Ensure 'form' column exists before filtering
    if 'form' not in filings_df.columns:
        print(f"    🟡 'form' column missing in filings data for {ticker}. Skipping.")
        return
    relevant_filings = filings_df[filings_df['form'].isin(target_forms)]
    
    if relevant_filings.empty:
            print(f"    No recent {', '.join(target_forms)} filings found for {ticker}.")
            return
    
    print(f"    Found {len(relevant_filings)} relevant filings to download.")


    for index, row in relevant_filings.iterrows():
        try:
            # Construct the direct URL to the filing's primary HTML document
            accession_num = row['accessionNumber'].replace('-', '')
            primary_doc_name = row['primaryDocument']
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_num}/{primary_doc_name}"
            
            print(f"Downloading: {primary_doc_name}...")
            doc_response = requests.get(doc_url, headers=headers)
            doc_response.raise_for_status() # check for errors like 404 not found
            
            # Create a structured path to save the file
            save_dir = f"sec-edgar-filings/{ticker}/{row['form']}/{accession_num}"
            save_path = os.path.join(save_dir, primary_doc_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the clean HTML file
            with open(save_path, 'wb') as f:
                f.write(doc_response.content)

            # SEC limits requests to 10 per second, so we add a small delay
            time.sleep(0.1)

        except Exception as e:
            print(f"Could not process document {row.get('primaryDocument', 'N/A')}: {e}")



YFINANCE_REQUEST_DELAY = 3.0 # Delay BETWEEN history/recs calls for SAME ticker
YFINANCE_TICKER_DELAY = 5.0  # Delay BETWEEN DIFFERENT tickers


def download_financial_data(tickers):
    """
    Downloads historical stock data (daily, hourly) and analyst recommendations
    using yfinance. Overwrites existing files with the latest data.

    Args:
        tickers (list): List of stock ticker symbols.
    """
    print("\n--- Downloading Historical Stock Data & Forecasts ---")

    for ticker in tickers:
        try:
            # Creates Ticker object for current company
            stock = yf.Ticker(ticker)
            
            # Daily Data (10 years)
            hist_data_daily = stock.history(period="10y", interval="1d")
            
            if hist_data_daily is None or hist_data_daily.empty:
                print(f"    🟡 No 10y daily data returned by yfinance.")
            else:
                hist_data_daily.to_csv(f"stock_data/{ticker}_10y_daily.csv")
                print(f"    ✅ Saved/Overwritten 10-year daily data.")
            time.sleep(YFINANCE_REQUEST_DELAY)


            # Hourly Data (2 years/730 days)
            print(f"    Downloading 2-year hourly data...")
            hist_data_hourly = stock.history(period="730d", interval="1h")

            if hist_data_hourly is None or hist_data_hourly.empty:
                 print(f"    🟡 No 2y hourly data returned by yfinance.")
            else:
                hist_data_hourly.to_csv(f"stock_data/{ticker}_2y_hourly.csv")
                print(f"    ✅ Saved/Overwritten 2-year hourly data.")
            time.sleep(YFINANCE_REQUEST_DELAY)
        
            

            # Get analyst recommendations (like 'buy', 'hold', or 'sell') for specified ticker
            recs = stock.recommendations
            # Checks if rec data exists and is not empty
            try:
                recs = stock.recommendations
                if recs is None or recs.empty:
                    print(f"    🟡 No recommendation data found.")
                else:
                    recs.to_csv(f"stock_data/{ticker}_recommendations.csv")
                    print(f"    ✅ Saved/Overwritten analyst recommendations.")
            except requests.exceptions.HTTPError as http_err:
                 # Specifically handle rate limiting errors for recommendations
                if http_err.response.status_code == 429:
                     print(f"    🟡 Rate limited (429) fetching recommendations. Skipping.")
                else:
                     print(f"    🟡 HTTP Error fetching recommendations: {http_err}") # Log other HTTP errors
            except Exception as rec_err:
                print(f"    🟡 Error fetching recommendations: {rec_err}")


        except Exception as e:
            print(f"❌ Could not download yfinance data for {ticker}: {e}")

        
    

def download_company_news(tickers):
    """
    Downloads recent news article headlines via NewsAPI for each ticker.
    Attempts to scrape the full text content from the article URL.
    Overwrites the existing news JSON file for each ticker.

    Args:
        tickers (list): List of stock ticker symbols.
        headers (dict): HTTP headers for scraping requests.
    """
    print("\n--- Downloading Recent Company News ---")
    for ticker in tickers:
        try:
            # Fetch top 30 relevant/recent headlines 
            # Note: Free tier NewsAPI only provides headlines/snippets, not full text.
            all_articles = NEWSAPI_CLIENT.get_everything(
                q=ticker,
                language='en',
                sort_by='relevancy',  # how relevant the result is to the query 
                page_size=30
            )
            


            print(f"    Found {len(all_articles['articles'])} articles. Now scraping full text...")
            
            # --- Loop through article headers and scrape text ---
            for article in all_articles['articles']:
                if not article['url']:
                    article['full_text'] = None
                    continue # Skip if no URL

                try:
                    # Use the same headers and a timeout
                    page_response = requests.get(article['url'], headers=HEADERS, timeout=10)
                    page_response.raise_for_status()
                    
                    # Parse the HTML
                    soup = BeautifulSoup(page_response.content, 'html.parser')
                    
                    # Extract all text, strip whitespace, and join with spaces
                    full_text = soup.get_text(separator=' ', strip=True)
                    
                    # Add the full text to our article dictionary
                    article['full_text'] = full_text
                    print(f"    ✅ Scraped: {article['title'][:50]}...")

                except Exception as e:
                    print(f"    🟡 Could not scrape {article['url']}: {e}")
                    article['full_text'] = None # Set to None on failure
                
                time.sleep(0.2)



            # Save the news data to a JSON file

            filepath = f"news_data/{ticker}_news_with_text.json"

            # Safely opens a new file for writing, or in this case
            # placing the articles into the new json
            with open(filepath, 'w') as f:
                json.dump(all_articles, f, indent=4)
            print(f"✅ Saved news articles for {ticker}")
            
        except Exception as e:
            print(f"❌ Could not download news for {ticker}: {e}")


#############################################
# --- Main Execution ---
#############################################

if __name__ == "__main__":

    start_time = time.time()
    print(f"Starting data collection for: {TICKERS}")
    
    os.makedirs("stock_data", exist_ok=True)
    os.makedirs("news_data", exist_ok=True)
    os.makedirs("sec-edgar-filings", exist_ok=True)
    
    # Get the CIK for each of our target tickers
    company_ciks = get_company_ciks(TICKERS, HEADERS)
    
    # Download SEC filings
    for ticker in TICKERS:
        cik = company_ciks.get(ticker)
        if cik:
            download_sec_filings(ticker, cik, HEADERS)
        else:
            print(f"⚠️ Skipping SEC filings for {ticker} - No CIK found.")
        time.sleep(1)

    # Download stock and news data
    download_financial_data(TICKERS)
    download_company_news(TICKERS)

    end_time = time.time()
    print("\n🎉 All data collection complete!")
    print(f"Total runtime: {timedelta(seconds=end_time - start_time)}")

