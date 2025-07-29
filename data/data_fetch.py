# data/data_fetch.py

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import pyotp
import yaml
from tqdm import tqdm

# Read keys
with open("config/keys.yml", "r") as f:
    keys = yaml.safe_load(f)
api_key = keys["api_key"]
client_id = keys["client_id"]
pwd = keys["password"]
totp_key = keys.get("totp_secret") or keys.get("totp")  # support both keys

otp = pyotp.TOTP(totp_key).now()
api = SmartConnect(api_key=api_key)
api.generateSession(client_id, pwd, otp)

def fetch_ohlcv(symbol, exchange, token, start, end, interval):
    dfs = []
    dt = start
    while dt <= end:
        # Skip weekends as the API does not return data for these days
        if dt.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
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
        while not success and retry_count < 3:
            try:
                data = api.getCandleData(params)
                if isinstance(data, dict) and not data.get("status", True):
                    # API returns status False for non-trading days
                    print(f"No data for {symbol} {dt_str}: {data.get('message')}")
                elif "data" in data and data["data"]:
                    day_df = pd.DataFrame(data["data"], columns=["date", "open", "high", "low", "close", "volume"])
                    day_df["symbol"] = symbol
                    dfs.append(day_df)
                success = True
            except Exception as e:
                err = str(e)
                if "exceeding access rate" in err or "Access denied" in err:
                    print(f"⚠️  Rate limit hit for {symbol} {dt_str}, waiting 60s and retrying...")
                    time.sleep(60)
                    retry_count += 1
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
