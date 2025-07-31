# data/data_fetch.py

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import holidays
import requests
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.utils import totp_now

logger = get_logger("DataFetch")

# Load credentials from environment or .env
load_dotenv()
api_key = os.getenv("ANGEL_API_KEY")
client_id = os.getenv("ANGEL_CLIENT_ID")
pwd = os.getenv("ANGEL_PASSWORD")
totp_key = os.getenv("ANGEL_TOTP_SECRET") or os.getenv("TOTP_SECRET") or os.getenv("TOTP")

if not totp_key:
    raise ValueError(
        "TOTP secret not set; please define ANGEL_TOTP_SECRET"
    )

otp = totp_now(totp_key)
api = SmartConnect(api_key=api_key)
resp = api.generateSession(client_id, pwd, otp)
if not resp.get("status"):
    logger.error(
        "Angel One login failed: %s (%s)",
        resp.get("message"),
        resp.get("errorcode"),
    )
    raise RuntimeError("Angel One login failed")

def relogin():
    """Generate a fresh OTP and refresh the Smart API session."""
    otp = totp_now(totp_key)
    resp = api.generateSession(client_id, pwd, otp)
    if not resp.get("status"):
        logger.error(
            "Angel One login failed: %s (%s)",
            resp.get("message"),
            resp.get("errorcode"),
        )
        raise RuntimeError("Angel One login failed")
    logger.info("Session refreshed via relogin().")

def fetch_ohlcv(symbol, exchange, token, start, end, interval):
    dfs = []
    # Precompute list of Indian market holidays for the given period
    ind_holidays = holidays.country_holidays('IN', years=range(start.year, end.year + 1))
    dt = start
    while dt <= end:
        # Skip weekends and official market holidays
        if dt.weekday() >= 5 or dt.date() in ind_holidays:
            dt += timedelta(days=1)
            continue

        dt_str = dt.strftime("%Y-%m-%d")
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": f"{dt_str} 09:15",
            "todate": f"{dt_str} 15:30"
        }
        success = False
        retry_count = 0
        max_retries = 5
        while not success and retry_count < max_retries:
            try:
                data = api.getCandleData(params)
                if isinstance(data, dict) and not data.get("status", True):
                    if data.get("errorcode") == "AB1004":
                        logger.info("Session expired (AB1004). Refreshing session.")
                        relogin()
                        retry_count += 1
                        continue
                    msg = data.get("message", "")
                    if "Something Went Wrong" in msg:
                        print(f"API error for {symbol} {dt_str}: {msg}. Retrying...")
                        time.sleep(30)
                        retry_count += 1
                        if retry_count >= 3:
                            relogin()
                        continue
                    else:
                        # API returns status False for non-trading days
                        print(f"No data for {symbol} {dt_str}: {msg}")
                elif "data" in data and data["data"]:
                    day_df = pd.DataFrame(data["data"], columns=["date", "open", "high", "low", "close", "volume"])
                    day_df["symbol"] = symbol
                    dfs.append(day_df)
                success = True
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                print(f"Network error fetching {symbol} {dt_str}: {e}. Retrying...")
                time.sleep(30)
                retry_count += 1
                if retry_count >= 3:
                    relogin()
                continue
            except Exception as e:
                err = str(e)
                if "exceeding access rate" in err or "Access denied" in err:
                    print(f"⚠️  Rate limit hit for {symbol} {dt_str}, waiting 60s and retrying...")
                    time.sleep(60)
                    retry_count += 1
                    if retry_count >= 3:
                        relogin()
                    continue
                else:
                    print(f"Error fetching {symbol} {dt_str}: {e}")
                    break
        time.sleep(2)
        dt += timedelta(days=1)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

symbol_tokens = {
    "NIFTY": "99926009",
    "BANKNIFTY": "99926037",
    "FINNIFTY": "99926044"
}

end_date = datetime.today()
YEARS_TO_FETCH = 10  # Number of years
all_df = []

for symbol, token in symbol_tokens.items():
    curr_end = end_date
    for year in range(YEARS_TO_FETCH):
        curr_start = curr_end - timedelta(days=365)
        print(f"Fetching {symbol} from {curr_start.date()} to {curr_end.date()} (year {year+1}/{YEARS_TO_FETCH})")
        df = fetch_ohlcv(symbol, "NSE", token, curr_start, curr_end, "ONE_MINUTE")
        if df.empty:
            print(f"No data for {symbol} from {curr_start.date()} to {curr_end.date()}. Stopping further fetch for this symbol.")
            break
        all_df.append(df)
        curr_end = curr_start - timedelta(days=1)  # Move window back by 1 day

if all_df:
    full_df = pd.concat(all_df, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    full_df.to_csv("data/historical_data.csv", index=False)
    print("✅ Done! Data saved to data/historical_data.csv")
else:
    print("❌ No data was fetched for any symbol!")
