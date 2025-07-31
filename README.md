Dev -
cd C:\myproject\bots\opttrbot
>> .\.venv\Scripts\activate

## Installation

Install required packages including the broker APIs (`smartapi-python`,
`angel-one-api` and `kiteconnect`):

```bash
pip install -r requirements.txt
```

### API credentials

Create a `.env` file in the project root or export the following environment
variables before running any scripts:

```bash
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

These values are used by `data_fetch.py` and the `AngelOneAPI` wrapper.
=============
For tranning the bot >>>>>>
Training the Bot Overnight :-
python data/data_fetch.py

python main.py --mode train
python main.py --autoloop


For Running th the bot in live market   >>>>>>>

Switching to Live Trading in the Morning :-
python main.py --mode live

