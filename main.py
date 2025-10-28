"""
main.py

FastAPI backend server for the Asset Edge application.

Provides API endpoints for:
- Interacting with the LangChain RAG agent (/chat).
- Fetching historical stock data for charting (/stock/{ticker}).
- Retrieving recent news articles (/news/{ticker}).
- Getting daily market movers (/market-movers).
- Generating LSTM-based stock price predictions (/predict/{ticker}).

Loads models and initializes agent on startup. Uses .env for API keys.
"""

import os
import sys
import json
import pickle
import time
from datetime import datetime, timedelta
import asyncio # For running agent in async endpoint
from typing import List, Dict, Union, Optional

import pandas as pd
import chromadb
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv # To load API key from .env file
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler

# --- LangChain Imports ---
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool 
from langchain.tools.retriever import create_retriever_tool # Tool wrapper for retrievers
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


# --- Pydantic Models for Request/Response ---
class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatQuery(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    response: str

# --- LSTM Model Definition  ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#############################################
# --- Configuration ---
#############################################

load_dotenv() # Load environment variables from .env file (for OPENAI_API_KEY)

CHROMA_PATH = "chroma_db"
STOCK_DATA_PATH = "stock_data"
NEWS_DATA_PATH = "news_data"
SENTIMENT_DATA_PATH = "sentiment_data" # Needed for prediction
MODEL_SAVE_PATH = "models"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-5-mini" # Your model
TICKERS = ["AAPL", "META", "NVDA", "GOOGL", "MSFT",
            "AMZN", "AVGO", "JPM", "NFLX", "ORCL"]

# LSTM Model/Data Config (Must match train_lstm.py)
SEQUENCE_LENGTH = 60
FEATURES = ['Close', 'Volume', 'sentiment_score']
TARGET_COLUMN = 'Close'
INPUT_SIZE = len(FEATURES)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# --- Initialize FastAPI App ---
app = FastAPI(title="Asset Edge API")

# --- CORS Middleware (Allow requests from your React frontend) ---
origins = [
    "http://localhost:3000", # Default React dev server port
    #"localhost:3000" # Sometimes needed without http
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables / Caching (Load models once on startup) ---
agent_executor = None
loaded_models = {} # Cache for {ticker: (model, scaler, target_scaler)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
# --- Langchain Tool Definitions ---
#############################################

@tool
def get_stock_performance(ticker: str) -> str:
    """
    Looks up recent stock performance (price, 30d/1y change) for a ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        str: A formatted string summarizing performance or an error message.
    """
    print(f"--- Calling Stock Tool for {ticker} ---")

    ticker = ticker.upper()
    filepath = os.path.join(STOCK_DATA_PATH, f"{ticker}_10y_daily.csv")

    if not os.path.exists(filepath):
        return f"Error: No stock data file found for {ticker} at {filepath}"

    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        df.dropna(subset=['Date'], inplace=True)
        df = df.sort_values(by='Date', ascending=False)

        if df.empty:
            return f"Error: No valid date entries found in stock data for {ticker}."
        
        latest_data = df.iloc[0]
        latest_price = latest_data['Close']
        latest_date = latest_data['Date'].strftime('%Y-%m-%d')

        now_naive = pd.Timestamp.now().tz_localize(None)
        
        # 30-Day Change Calculation
        date_30_days_ago = now_naive - pd.DateOffset(days=30)
        past_data_30d = df[df['Date'] <= date_30_days_ago]
        if past_data_30d.empty:
             return f"Error: Not enough historical data to calculate 30-day change for {ticker}."
        
        past_price_30d = past_data_30d.iloc[0]['Close']
        change_30d = ((latest_price - past_price_30d) / past_price_30d) * 100

        # 1-Year Change Calculation
        date_1_year_ago = now_naive - pd.DateOffset(years=1)
        past_data_1y = df[df['Date'] <= date_1_year_ago]
        if past_data_1y.empty:
             return f"Error: Not enough historical data to calculate 1-year change for {ticker}."
        past_price_1y = past_data_1y.iloc[0]['Close']
        change_1y = ((latest_price - past_price_1y) / past_price_1y) * 100

        return (
            f"Stock performance for {ticker} (as of {latest_date}):\n"
            f"- Latest Price: ${latest_price:,.2f}\n"
            f"- 30-Day Change: {change_30d:+.2f}%\n"
            f"- 1-Year Change: {change_1y:+.2f}%"
        )

    except Exception as e:
        return f"Error processing stock data for {ticker}: {type(e).__name__} - {e}"

#############################################
# --- Helper Function to Initialize Agent  ---
#############################################

def initialize_agent():
    """Initializes the LangChain agent components (LLM, tools, prompt, executor)."""
    global agent_executor  # Allow modification of the global variable 
    print("Initializing LangChain agent...")
    try:
        # Initialize LLM (Ensure OPENAI_API_KEY is in .env or environment)
        llm = ChatOpenAI(model=LLM_MODEL, temperature=1.0, streaming=False) # Keep streaming=False
        embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Initialize RAG Retriever Tool
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            vector_store = Chroma(
                client=client, collection_name="asset_edge", embedding_function=embedding_model
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        except Exception as chroma_err:
             print(f"    ❌ FATAL: Failed to connect or get collection from ChromaDB: {chroma_err}")
             raise # Re-raise to stop initialization
        
        rag_tool = create_retriever_tool(
            retriever,
            "sec_and_news_retriever",
            "Searches and retrieves relevant context ONLY from SEC filings (10-K, 10-Q, 8-K) and recent news article headlines/descriptions. Use this tool to find qualitative information like company risks, future outlook, management discussion, reasons for performance, recent developments, and specific details mentioned in official filings or news reports.",
        )
        print("✅ RAG tool initialized.")
        
        # Define Tools List
        tools = [rag_tool, get_stock_performance]

        # Create Agent Prompt
        system_prompt = (
            "You are a helpful and powerful financial analyst assistant named 'Asset Edge'. "
            "Your goal is to provide accurate, concise, and informative answers based *only* on the context provided by the available tools. "
            "You have access to two tools:\n"
            "1. `get_stock_performance`: Provides recent quantitative stock price data (latest price, 30-day change, 1-year change) for a specific ticker.\n"
            "2. `sec_and_news_retriever`: Searches a knowledge base of recent SEC filings (10-K, 10-Q, 8-K) and news headlines/descriptions for qualitative information about companies.\n\n"
            "## Instructions:\n"
            "- **Analyze the User Query:** Understand what information is being asked for (e.g., price data, risks, news, outlook).\n"
            "- **Select Appropriate Tool(s):** Use `get_stock_performance` for price/performance questions. Use `sec_and_news_retriever` for questions about risks, news, filings, outlook, reasons, etc.\n"
            "- **Use History:** Refer to the `chat_history` to understand the context of the conversation and avoid asking repetitive questions.\n"
            "- **Synthesize Information:** Combine information from the tools and chat history to form a comprehensive answer.\n"
            "- **Cite Sources:** ALWAYS cite the source of your information. For `get_stock_performance`, mention it came from the tool. For `sec_and_news_retriever`, specify if it's from an SEC filing (mention form type like 10-K if known from context) or a news source (mention source name like 'Reuters' if available in metadata).\n"
            "- **Provide URLs:** If citing news retrieved by `sec_and_news_retriever`, include the URL provided in the document's metadata.\n"
            "- **Be Concise:** Answer directly. Avoid unnecessary pleasantries or asking clarifying questions *if* the necessary information can be inferred or found using the tools.\n"
            "- **DO NOT Hallucinate:** If the tools do not provide relevant information to answer the question, clearly state that the information is not available in the provided documents or data. Do not make up facts or figures.\n"
            "- **Handle Errors:** If a tool returns an error message, relay that information to the user."
            )
    
        

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create Agent and Executor
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) 
        print("✅ LangChain agent initialized successfully.")

    except Exception as e:
        print(f"❌ FATAL ERROR initializing LangChain agent: {e}")
        agent_executor = None   # Ensure 'None' on failure 

#############################################
# --- Helper Function to Load Prediction Assets ---
#############################################

def load_prediction_assets(ticker):
    """
    Loads the saved LSTM model state, feature scaler, and target scaler for a given ticker.
    Caches loaded assets in the `loaded_models` global dictionary.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        tuple[Optional[nn.Module], Optional[MinMaxScaler], Optional[MinMaxScaler]]:
            A tuple containing the loaded model, feature scaler, and target scaler,
            or (None, None, None) if any asset fails to load.
    """
    if ticker in loaded_models:
        return loaded_models[ticker]

    print(f"Loading prediction assets for {ticker}...")
    model_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}_lstm_model.pth")
    scaler_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}_target_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Warning: Model or scaler not found for {ticker}. Prediction unavailable.")
        return None, None 

    try:
        # Load model state 
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode

        # Load feature scaler
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        print(f"    Feature scaler loaded from: {scaler_path}")
        
        # Load target scaler
        with open(scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)

       
        # stock_file = os.path.join(STOCK_DATA_PATH, f"{ticker}_2y_hourly.csv")
        # sentiment_file = os.path.join(SENTIMENT_DATA_PATH, f"{ticker}_sentiment.csv")
        # stock_df = pd.read_csv(stock_file)
        # sentiment_df = pd.read_csv(sentiment_file)
        # stock_df['Datetime'] = pd.to_datetime(stock_df['Datetime'], utc=True).dt.tz_localize(None)
        # sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        # stock_df.set_index('Datetime', inplace=True)
        # stock_df['date'] = stock_df.index.date
        # stock_df['date'] = pd.to_datetime(stock_df['date'])
        # merged_df = pd.merge_ordered(stock_df.reset_index(), sentiment_df, on='date', fill_method='ffill')
        # merged_df.set_index('Datetime', inplace=True)
        # merged_df.dropna(inplace=True)

        # if merged_df.empty:
        #      raise ValueError(f"No data found for {ticker} to fit feature scaler.")

        # from sklearn.preprocessing import MinMaxScaler # Import if not already done
        # feature_scaler = MinMaxScaler(feature_range=(0, 1))
        # feature_scaler.fit(merged_df[FEATURES]) # Fit on historical data


        loaded_models[ticker] = (model, feature_scaler, target_scaler)
        print(f"✅ Prediction assets loaded for {ticker}.")
        return model, feature_scaler, target_scaler

    except Exception as e:
        print(f"❌ Error loading prediction assets for {ticker}: {e}")
        return None, None, None

#############################################
# --- FastAPI Endpoints ---
#############################################

@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the server starts."""
    initialize_agent()
    # Pre-load models for common tickers if desired
    # for ticker in TICKERS:
    #     load_prediction_assets(ticker)


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: Request): 
    """
    Handles chat requests, manages history, invokes the agent, and returns the response.
    Manually parses the request body to handle potential Pydantic issues.
    """
    if agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    # --- Manually Parse Request Body ---
    try:
        body = await request.json()
        print(f"\nDEBUG: Raw request body received by /chat:\n{body}\n") # Verify body
        user_query = body.get("query")
        history_data = body.get("history", []) # Default to empty list if not found

        if not user_query:
             raise HTTPException(status_code=400, detail="Missing 'query' field in request body")

    except Exception as e:
         print(f"Error reading request body: {e}")
         raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")


    # --- Convert history data to LangChain format ---
    chat_history_langchain_format = []
    if history_data: 
        for msg_data in history_data:
            role = msg_data.get("role")
            content = msg_data.get("content")
            if role == "user" and content:
                chat_history_langchain_format.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                 chat_history_langchain_format.append(AIMessage(content=content))
            else:
                 print(f"Warning: Skipping invalid history item: {msg_data}") # Log invalid items

    # Invoke Agent
    try:
        agent_input = {
            "input": user_query, 
            "chat_history": chat_history_langchain_format 
        }
        print(f"DEBUG: Calling agent_executor.invoke with input: {agent_input['input'][:50]}...") # Keep this debug
        
        # Use asyncio.to_thread for synchronous LangChain invoke in async context
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, agent_executor.invoke, agent_input
        )
        print(f"DEBUG: Agent invocation finished. Response received.") 
        return ChatResponse(response=result.get('output', 'Error: No output received'))
    except Exception as e:
        print(f"Error during agent invocation: {e}") 
        raise HTTPException(status_code=500, detail=f"An error occurred during agent invocation: {e}")



@app.get("/stock/{ticker}")
async def get_stock_data(ticker: str):
    """
    Endpoint to get historical stock data, filtered by time range.
    Uses daily data for longer ranges (6M, 1Y, MAX) for performance.

    Args:
        ticker (str): Stock ticker symbol.
        range (str): Time range ('1W', '1M', '6M', '1Y', 'MAX'). Defaults to '1M'.

    Returns:
        dict: Contains a 'data' key with a list of chart data points [{time, Open, ...}].
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    filepath_daily = os.path.join(STOCK_DATA_PATH, f"{ticker}_10y_daily.csv")
    filepath_hourly = os.path.join(STOCK_DATA_PATH, f"{ticker}_2y_hourly.csv")

    if not os.path.exists(filepath_daily) or not os.path.exists(filepath_hourly):
        raise HTTPException(status_code=404, detail=f"Data files not found for {ticker}")

    try:
        df_daily = pd.read_csv(filepath_daily, parse_dates=['Date'])
        df_hourly = pd.read_csv(filepath_hourly, parse_dates=['Datetime'])

        # Select relevant columns and format for charting
        chart_data_daily = df_daily[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={'Date':'time'}).to_dict('records')
        chart_data_hourly = df_hourly[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={'Datetime':'time'}).to_dict('records')

        # Convert Timestamps to string format suitable for JSON/JavaScript
        for record in chart_data_daily:
            record['time'] = record['time'].isoformat()
        for record in chart_data_hourly:
            record['time'] = record['time'].isoformat()


        return {"daily": chart_data_daily, "hourly": chart_data_hourly}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading stock data: {e}")


@app.get("/news/{ticker}")
async def get_news_articles(ticker: str, limit: int = 5):
    """
    Endpoint to get recent news article details (title, desc, url, image, etc.).

    Args:
        ticker (str): Stock ticker symbol.
        limit (int): Max number of articles to return (1-20). Defaults to 5.

    Returns:
        dict: Contains an 'articles' key with a list of article dictionaries.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    filepath = os.path.join(NEWS_DATA_PATH, f"{ticker}_news_with_text.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"News file not found for {ticker}")

    articles_to_return = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        # Return only essential fields for the top 'limit' articles
        for article in news_data.get('articles', [])[:limit]:
             articles_to_return.append({
                "title": article.get('title'),
                "description": article.get('description'),
                "url": article.get('url'),
                "urlToImage": article.get('urlToImage'),
                "publishedAt": article.get('publishedAt'),
                "sourceName": article.get('source', {}).get('name')
             })
        return {"articles": articles_to_return}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading news data: {e}")


@app.get("/market-movers")
async def get_market_movers(count: int = 3):
    """
    Endpoint to get top daily market gainers and losers from the TICKERS list.

    Args:
        count (int): Number of gainers/losers to return (1-5). Defaults to 3.

    Returns:
        dict: Contains 'gainers' and 'losers' keys, each with a list of
              {ticker, latest_price, change_pct} dictionaries.
    """
    movers = []
    for ticker in TICKERS:
        filepath = os.path.join(STOCK_DATA_PATH, f"{ticker}_10y_daily.csv")
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'])
            df = df.sort_values(by='Date', ascending=False)
            if len(df) < 2:
                continue
            latest_close = df.iloc[0]['Close']
            prev_close = df.iloc[1]['Close']
            change_pct = ((latest_close - prev_close) / prev_close) * 100
            movers.append({
                "ticker": ticker,
                "latest_price": latest_close,
                "change_pct": change_pct
            })
        except Exception as e:
            print(f"Error calculating movers for {ticker}: {e}")

    # Sort by percentage change
    movers.sort(key=lambda x: x['change_pct'], reverse=True)

    top_gainers = movers[:count]
    top_losers = sorted(movers, key=lambda x: x['change_pct'])[:count] # Sort ascending for losers

    return {"gainers": top_gainers, "losers": top_losers}


@app.get("/predict/{ticker}")
async def get_prediction_endpoint(ticker: str):
    """
    Endpoint to generate the next hour's stock price prediction using the trained LSTM model.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        dict: Contains 'ticker' and 'predicted_price'.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported for prediction")

    model, feature_scaler, target_scaler = load_prediction_assets(ticker)
    if not model or not feature_scaler or not target_scaler:
        raise HTTPException(status_code=503, detail=f"Prediction model/scalers not available for {ticker}")

    try:
        # --- Fetch latest data needed for the sequence ---
        stock_file = os.path.join(STOCK_DATA_PATH, f"{ticker}_2y_hourly.csv")
        sentiment_file = os.path.join(SENTIMENT_DATA_PATH, f"{ticker}_sentiment.csv")
        stock_df = pd.read_csv(stock_file)
        sentiment_df = pd.read_csv(sentiment_file)
        stock_df['Datetime'] = pd.to_datetime(stock_df['Datetime'], utc=True).dt.tz_localize(None)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        stock_df.set_index('Datetime', inplace=True)
        stock_df['date'] = stock_df.index.date
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        merged_df = pd.merge_ordered(stock_df.reset_index(), sentiment_df, on='date', fill_method='ffill')
        merged_df.set_index('Datetime', inplace=True)
        merged_df.dropna(inplace=True)
        merged_df = merged_df.sort_index() # Ensure data is sorted chronologically

        if len(merged_df) < SEQUENCE_LENGTH:
             raise ValueError(f"Not enough data ({len(merged_df)} points) to form sequence of length {SEQUENCE_LENGTH} for {ticker}")

        # Get the last SEQUENCE_LENGTH points
        latest_sequence_df = merged_df[FEATURES].tail(SEQUENCE_LENGTH)

        # Scale the sequence using the loaded feature scaler
        scaled_sequence = feature_scaler.transform(latest_sequence_df.values)

        # Convert to tensor and add batch dimension
        sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device)

        # --- Predict ---
        with torch.no_grad():
            scaled_prediction = model(sequence_tensor)

        # Inverse transform the prediction using the TARGET scaler
        predicted_price = target_scaler.inverse_transform(scaled_prediction.cpu().numpy())[0][0]

        return {"ticker": ticker, "predicted_price": float(predicted_price)} # Convert numpy float

    except Exception as e:
        print(f"Error during prediction for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

#############################################
# --- Server Execution ---
#############################################

if __name__ == "__main__":
    import uvicorn
    # Check for API key and print warning if missing
    if not os.getenv("OPENAI_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: OPENAI_API_KEY environment variable not set. !!!")
        print("!!! Agent functionality will likely fail.                 !!!")
        print("!!! Create a .env file or export the variable.            !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    uvicorn.run(app, host="0.0.0.0", port=8000)