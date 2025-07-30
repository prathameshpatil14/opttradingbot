Dev -
cd C:\myproject\bots\opttrbot
>> .\.venv\Scripts\activate

## Installation

Install required packages including the broker APIs (`smartapi-python`,
`angel-one-api` and `kiteconnect`):

```bash
pip install -r requirements.txt
```
=============
For tranning the bot >>>>>>
Training the Bot Overnight :-
python data/data_fetch.py

python main.py --mode train
python main.py --autoloop


For Running th the bot in live market   >>>>>>>

Switching to Live Trading in the Morning :-
python main.py --mode live

