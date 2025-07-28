Dev -
cd C:\myproject\bots\opttrbot                                                                          
>> .\.venv\Scripts\activate    
=============
For tranning the bot >>>>>>
Training the Bot Overnight :-
python data/data_fetch.py

python main.py --mode train
python main.py --autoloop


For Running th the bot in live market   >>>>>>>

Switching to Live Trading in the Morning :-
python main.py --mode live

### Notes

`OptionTradingEnv.step()` returns a reward value and an `info` dictionary.
The dictionary now includes a `pnl` key containing the raw trade profit/loss
separate from the shaped reward.
