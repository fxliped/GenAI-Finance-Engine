"""
process_data.py

Processes raw financial data downloaded by the download_data.py script.
Performs two main tasks:
1.  Builds/Updates a Vector Database (ChromaDB) for RAG:
    - Parses HTML SEC filings.
    - Extracts text from news JSON files (using title/description).
    - Splits text into chunks using LangChain.
    - Generates embeddings using Sentence Transformers.
    - Stores chunks, metadata, and embeddings in ChromaDB.
2.  Generates Sentiment Time Series for ML:
    - Analyzes sentiment of news headlines/descriptions using a Hugging Face pipeline.
    - Aggregates sentiment scores daily for each ticker.
    - Saves results as CSV files for potential use in ML models (e.g., LSTM).

Assumes raw data exists in ./sec-edgar-filings/, ./news_data/, and expects
to write outputs to ./chroma_db/ and ./sentiment_data/.
"""

import os
import sys
import json
import glob
import warnings
import time
from datetime import timedelta
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from sentence_transformers import SentenceTransformer # New: For embedding
from transformers import pipeline, Pipeline # For sentiment/type hinting


#############################################
# --- Configuration ---
#############################################

TICKERS_TO_PROCESS = ["AAPL", "META", "NVDA", "GOOGL", "MSFT", 
           "AMZN", "AVGO", "JPM", "NFLX", "ORCL"]

# Output Directories
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A fast, high-quality model

# RAG/Embedding Parameters
CHUNK_SIZE = 1000        # Max characters per text chunk
CHUNK_OVERLAP = 200      # Character overlap over adjacent chunks

# Suppress warnings from HTML parsing and from libraries like Hugging Face/Pandas
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#############################################
# --- RAG DATABASE FUNCTIONS ---
#############################################


def process_sec_filings(ticker, collection, text_splitter, embedding_model):
    """
    Finds, parses, chunks, embeds SEC filings for a ticker, and upserts into ChromaDB.
    Uses file paths to determine metadata (form type, accession number).

    Args:
        ticker (str): Stock ticker symbol.
        collection (Collection): ChromaDB collection object.
        text_splitter (TextSplitter): LangChain text splitter instance.
        embedding_model (SentenceTransformer): Sentence Transformer model instance.
    """
    print(f"\n--- Processing SEC filings for {ticker} ---")

    # Search for htm and html filings in the ticker's directory within system
    filing_paths = glob.glob(f'./sec-edgar-filings/{ticker}/*/*/*.htm*')
    
    if not filing_paths:
        print(f"🟡 No filing files found for {ticker}.")
        return

    all_docs = []
    all_metadatas = []
    all_ids = []
    
    for path in filing_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'lxml')
            clean_text = soup.get_text(separator=' ', strip=True)
            
            if not clean_text:
                continue

            # Get metadata from the file path
            parts = path.split(os.sep)
            form_type = parts[-3]
            accession_num = parts[-2]
            filename = parts[-1]
            
            # Chunk the clean text
            chunks = text_splitter.split_text(clean_text)
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "ticker": ticker,
                    "source": "SEC",
                    "form_type": form_type,
                    "accession_number": accession_num,
                    "filename": filename
                }
                # Create a unique ID for each chunk
                chunk_id = f"sec_{ticker}_{accession_num}_{i}"
                
                all_docs.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)

        except Exception as e:
            print(f"❌ Error processing file {path}: {e}")
    
    if not all_docs:
        print(f"🟡 No text chunks generated for {ticker} SEC filings.")
        return


    # Embed and update all chunks/data in ChromaDB in batches (much faster)
    print(f"    Embedding {len(all_docs)} text chunks for {ticker} SEC filings...")
    embeddings = embedding_model.encode(all_docs, show_progress_bar=True)
    
    try:
        collection.upsert(
            embeddings=embeddings.tolist(),  # Need a list for ChromaDB
            documents=all_docs,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"✅ Successfully added {len(all_docs)} SEC chunks to vector DB.")
    except Exception as e:
        print(f"❌ Error adding SEC chunks to ChromaDB: {e}")


def process_news_data(ticker, collection, embedding_model):
    """
    Loads news data JSON, extracts title/description, embeds, and upserts into ChromaDB.
    Assumes title + description is short enough to not require further chunking.

    Args:
        ticker (str): Stock ticker symbol.
        collection (Collection): ChromaDB collection object.
        embedding_model (SentenceTransformer): Sentence Transformer model instance.
    """
    print(f"\n--- Processing News articles for {ticker} ---")
    json_path = f"news_data/{ticker}_news_with_text.json"
    
    if not os.path.exists(json_path):
        print(f"🟡 No news file found for {ticker}.")
        return

    all_docs = []
    all_metadatas = []
    all_ids = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            news = json.load(f)
        
        for article_index, article in enumerate(news['articles']):
            
            title = article.get('title') or ""
            description = article.get('description') or ""
            # Combine title and description for embedding chunk
            content_to_embed = f"Headline: {title}. Description: {description}"
            
            # Skip if there's no meaningful text content
            if not content_to_embed:
                continue 
            
            metadata = {
                "ticker": ticker,
                "source": "News",
                "headline": title,
                "news_source": article.get('source', {}).get('name', ''),
                "published_at": article.get('publishedAt', ''),
                "url": article.get('url', '') 
            }
            # Create a unique ID for each chunk (using index as publishedAt may not be unique)
            chunk_id = f"news_{ticker}_{article.get('publishedAt', '')}_{article_index}"
            
            all_docs.append(content_to_embed)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)
                
    except Exception as e:
        print(f"❌ Error processing news file {json_path}: {e}")

    if not all_docs:
        print(f"🟡 No valid entries generated for {ticker}.")
        return
        
    # Embed all chunks in a batch and upsert
    try:
        print(f"    Embedding {len(all_docs)} text chunks for {ticker} news...")
        embeddings = embedding_model.encode(all_docs, show_progress_bar=True)
    
        # Add to ChromaDB
        collection.upsert(
            embeddings=embeddings.tolist(),
            documents=all_docs,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"✅ Successfully added {len(all_docs)} news chunks to vector DB.")
    except Exception as e:
        print(f"❌ Error adding news chunks to ChromaDB: {e}")


#############################################
# --- SENTIMENT FEATURE ENGINEERING ---
#############################################


def create_sentiment_timeseries(tickers, sentiment_pipeline):
    """
    Analyzes news articles for each ticker using a sentiment analysis pipeline
    and creates a daily aggregated sentiment time series CSV file.
    This output is intended for ML model training.

    Args:
        tickers (list): List of stock ticker symbols.
        sentiment_pipeline (Pipeline): Hugging Face sentiment analysis pipeline.
    """
    print("\n--- Creating Sentiment Time Series (for LSTM) ---")
    os.makedirs("sentiment_data", exist_ok=True)
    
    for ticker in tickers:
        print(f"Analyzing sentiment for {ticker}...")
        json_path = f"news_data/{ticker}_news_with_text.json" 
        
        if not os.path.exists(json_path):
            print(f"    🟡 News JSON file not found for {ticker}. Skipping sentiment analysis.")
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                news = json.load(f)
            
            if not news['articles']:
                print(f"🟡 No news articles found for {ticker}.")
                continue

            # Analyze sentiment on title + description for the pipeline
            texts = []
            for article in news['articles']:
                title = article.get('title', '') or ""
                description = article.get('description', '') or ""
                texts.append(title + " " + description)
            
            sentiments = sentiment_pipeline(texts)
            
            # Combine results with dates
            for i, article in enumerate(news['articles']):
                article['sentiment'] = sentiments[i]
            
            # Create a DataFrame and calculate daily average sentiment
            df = pd.DataFrame(news['articles'])
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            df['date'] = df['publishedAt'].dt.date
            
            # Convert sentiment labels to numeric scores (-1, 0, 1)
            # Calculated score: label score * confidence
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            df['sentiment_score'] = df['sentiment'].apply(
                lambda x: sentiment_map.get(x['label'].lower(), 0) * x['score']
            )
            
            # Group by date and calculate the mean score
            daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
            daily_sentiment.to_csv(f"sentiment_data/{ticker}_sentiment.csv", index=False)
            print(f"✅ Saved daily sentiment data for {ticker}")

        except Exception as e:
            print(f"❌ Error creating sentiment data for {ticker}: {e}")


#############################################
# --- Main Execution ---
#############################################

if __name__ == "__main__":
    
    overall_start_time = time.time()

    try:
    # --- Initialize Models and  RAG Vector Database ---
        print("Initializing RAG processing...")
        
        # Initialize ChromaDB client ( will create the 'chroma_db' folder)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initializing the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Create or get the main collection
        collection = client.get_or_create_collection(
            name="asset_edge"
            )
        
        print("Vector database and models initialized.")

    except Exception as e:
        print(f"❌ FATAL ERROR during resource initialization: {e}")
        sys.exit("Script terminated: Failed to initialize resources.")

    # -- Process data for RAG
    rag_start_time = time.time()
    for ticker in TICKERS_TO_PROCESS:
        print(f"\n==================== PROCESSING {ticker} for RAG ====================")
        process_sec_filings(ticker, collection, text_splitter, embedding_model)
        process_news_data(ticker, collection, text_splitter, embedding_model)
        # Small pause between tickers during processing
        time.sleep(0.5)
    rag_end_time = time.time()
    print("\n--- RAG Data Processing Summary")
    print(f"Total documents in database: {collection.count()}")
    print(f"RAG processing duration: {timedelta(seconds=rag_end_time - rag_start_time)}")
    
    # --- Create Sentiment Time-Series Data (for LSTM) ---
    print("\nInitializing sentiment analysis pipeline (FinBERT)...")
    # This will download the model the first time you run it
    sentiment_start_time = time.time()
    try: 
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert")
        create_sentiment_timeseries(TICKERS_TO_PROCESS, sentiment_pipeline)
    except Exception as e:
         print(f"❌ Error initializing or running sentiment pipeline: {e}")
         print("--- Skipping sentiment analysis ---")
    sentiment_end_time = time.time()
    print(f"Sentiment analysis duration: {timedelta(seconds=sentiment_end_time - sentiment_start_time)}")
    
    overall_end_time = time.time()
    print("\n All data processing complete!")
    print(f"Total script runtime: {timedelta(seconds=overall_end_time - overall_start_time)}")