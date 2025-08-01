# src/angel_one_api.py

import time
import os
import pandas as pd
from SmartApi import SmartConnect
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.utils import totp_now

logger = get_logger("AngelOneAPI")

class AngelOneAPI:
    """Wrapper around the Angel One SmartConnect API.

    Credentials are loaded from environment variables. Use a ``.env`` file or
    set ``ANGEL_API_KEY``, ``ANGEL_CLIENT_ID``, ``ANGEL_PASSWORD`` and
    ``ANGEL_TOTP_SECRET`` in your shell environment.
    """

    def __init__(self, token_map_path="data/symbol_token_map.csv"):
        load_dotenv()
        self.token_map_path = token_map_path
        self._load_credentials()
        self.api = self._login()
        self.token_cache = {}  # {(symbol, expiry, strike, option_type): (tradingsymbol, token)}
        self._load_token_map()

    def _load_credentials(self):
        """Read API credentials from environment variables."""
        self.api_key = os.getenv("ANGEL_API_KEY")
        self.client_id = os.getenv("ANGEL_CLIENT_ID")
        self.pin = os.getenv("ANGEL_PIN")
        # Support both a direct TOTP value and a secret key
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET")
        self.totp = os.getenv("ANGEL_TOTP")
        self.password = os.getenv("ANGEL_PASSWORD")

    def _login(self):
        api = SmartConnect(api_key=self.api_key)
        try:
            otp = totp_now(self.totp_secret) if self.totp_secret else self.totp
            data = api.generateSession(self.client_id, self.password, otp)
            if not data.get("status"):
                logger.warning("Initial AngelOne login failed; retrying with time sync.")
                otp = (
                    totp_now(self.totp_secret, force_sync=True)
                    if self.totp_secret
                    else self.totp
                )
                data = api.generateSession(self.client_id, self.password, otp)
                if not data.get("status"):
                    raise RuntimeError(
                        f"AngelOne login failed: {data.get('message')} ({data.get('errorcode')})"
                    )
            logger.info("AngelOne login successful.")
        except Exception as e:
            logger.error(f"AngelOne login failed: {e}")
            raise e
        return api
    
    def resolve_option_token(self, symbol, expiry, strike, option_type):
        
        expiry = pd.to_datetime(expiry).strftime("%Y-%m-%d") if expiry else None
        key = (symbol, expiry, float(strike), option_type)
        if key in self.token_cache:
            return self.token_cache[key]

        df = self.token_map
        match = df[
            (df["symbol"] == symbol) &
            (df["expiry"] == expiry) &
            (df["strike"].astype(float) == float(strike)) &
            (df["option_type"] == option_type)
        ]
        if not match.empty:
            tradingsymbol = match.iloc[0].get("tradingsymbol", "")
            token = str(match.iloc[0]["token"])
            self.token_cache[key] = (tradingsymbol, token)
            return tradingsymbol, token

        logger.warning(f"Token not found in local map for {symbol} {expiry} {strike} {option_type}. Update your symbol_token_map.csv!")
        return None, None
    
    def _load_token_map(self):
        if os.path.exists(self.token_map_path):
            self.token_map = pd.read_csv(self.token_map_path)
        else:
            self.token_map = pd.DataFrame(columns=["symbol", "expiry", "strike", "option_type", "tradingsymbol", "token"])

    def get_option_chain(self, symbol, expiry):
        # Not all brokers provide this; adjust as needed.
        try:
            params = {"exchange": "NFO", "symboltoken": symbol, "expirydate": expiry}
            resp = self.api.getOptionChain(params)
            df = pd.DataFrame(resp["data"])
            return df
        except Exception as e:
            logger.error(f"Failed to fetch option chain: {e}")
            return pd.DataFrame()

    def get_ltp(self, tradingsymbol, token, exchange="NFO"):
        try:
            ltp = self.api.ltpData(exchange, tradingsymbol, token)
            return ltp.get("data", {}).get("close", None)
        except Exception as e:
            logger.error(f"Failed to get LTP: {e}")
            return None

    def place_order(self, symbol, token, side, qty, expiry, strike, option_type, tradingsymbol=None, producttype="INTRADAY"):
        
        if not token or not tradingsymbol:
            # Try to resolve if tradingsymbol is missing
            ts, tk = self.resolve_option_token(symbol, expiry, strike, option_type)
            if not token:
                token = tk
            if not tradingsymbol:
                tradingsymbol = ts

        if not token or not tradingsymbol:
            logger.error("Cannot place order: token or tradingsymbol is None.")
            return "NO_TOKEN_OR_SYMBOL"

        try:
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": tradingsymbol,   # e.g. NIFTY24JUN23000CE
                "symboltoken": token,
                "transactiontype": side,          # "BUY" or "SELL"
                "exchange": "NFO",                # Options segment
                "ordertype": "MARKET",
                "producttype": producttype,
                "duration": "DAY",
                "quantity": qty,
            }
            response = self.api.placeOrder(orderparams)
            logger.info(f"Placed {side} order: {tradingsymbol} ({symbol} {expiry} {strike} {option_type}), resp: {response}")
            return response
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return str(e)
        
    def get_token_for_option(self, symbol, expiry, strike, option_type):
        ts, token = self.resolve_option_token(symbol, expiry, strike, option_type)
        return token
    
    
    def order_status(self, order_id):
        try:
            resp = self.api.orderBook()
            for order in resp["data"]:
                if order["orderid"] == order_id:
                    return order
            return None
        except Exception as e:
            logger.error(f"Order status check failed: {e}")
            return None

    def logout(self):
        try:
            self.api.terminateSession(self.client_id)
        except Exception as e:
            logger.warning(f"Logout error: {e}")
