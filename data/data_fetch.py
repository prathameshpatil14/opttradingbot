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
    logger.warning("Initial login failed, retrying with time sync.")
    otp = totp_now(totp_key, force_sync=True)
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
        logger.warning("Relogin failed, retrying with time sync.")
        otp = totp_now(totp_key, force_sync=True)
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

DATA_PATH = "data/historical_data.csv"
MONTHS_TO_FETCH = 6

existing_df = pd.DataFrame()
if os.path.exists(DATA_PATH):
    try:
        existing_df = pd.read_csv(DATA_PATH)
        existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
    except Exception as e:
        logger.warning(f"Failed to load existing data: {e}")

end_date = datetime.today()
min_start = end_date - timedelta(days=30 * MONTHS_TO_FETCH)

all_df = []

for symbol, token in symbol_tokens.items():
    start_dt = min_start
    if not existing_df.empty:
        sym_df = existing_df[existing_df["symbol"] == symbol]
        if not sym_df.empty:
            last_date = sym_df["date"].max().normalize() + timedelta(days=1)
            if last_date > start_dt:
                start_dt = last_date
    if start_dt > end_date:
        print(f"Data for {symbol} already up to date. Skipping fetch.")
        continue

    print(f"Fetching {symbol} from {start_dt.date()} to {end_date.date()}")
    df = fetch_ohlcv(symbol, "NSE", token, start_dt, end_date, "ONE_MINUTE")
    if not df.empty:
        all_df.append(df)

if all_df:
    new_df = pd.concat(all_df, ignore_index=True)
    full_df = pd.concat([existing_df, new_df], ignore_index=True)
    full_df.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    full_df.to_csv(DATA_PATH, index=False)
    print("✅ Done! Data saved to data/historical_data.csv")
else:
    print("❌ No data was fetched for any symbol!")
