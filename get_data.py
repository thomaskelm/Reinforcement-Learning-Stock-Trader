import talib
import yfinance as yf
from fp.fp import FreeProxy

lookup = [
    item
        for key, sublist in talib.__dict__['get_function_groups']().items()
            for item in sublist
                if key not in ['Math Transform', 'Math Operators']
]

def dataset_loader(stock_name, start, end, interval, use_proxy=True):
    ticker = yf.Ticker(stock_name)
    temp = None
    for i in range(1,11):
        if temp is not None:
            break
        try:
            temp = ticker.history(period='max',start=start,end=end,interval=interval,proxy=FreeProxy(country_id=['US'],rand=True).get())
        except Exception as e:
            print(f'likely issue with the proxy on try {i}, trying {11-i} more times')
            print(e)
    temp.reset_index(inplace=True)
    return temp
    # df = add_indicators(temp)
    # df[~df.ADX.isna()][:]
    # return df

def add_indicators(data):
    df = data.copy(deep=True)
    for indicator in lookup:
        try:
            result = getattr(talib,indicator)(df.Close)
            if isinstance(result, tuple):
                for i, val in enumerate(result):
                    df[f'{indicator}_{i}'] = val
            else:
                df[f'{indicator}'] = val
        except (KeyboardInterrupt, SystemError):
            raise
        except:
            try:
                result = getattr(talib,indicator)(df.High, df.Low)
                if isinstance(result, tuple):
                    for i, val in enumerate(result):
                        df[f'{indicator}_{i}'] = val
                else:
                    df[f'{indicator}'] = val
            except (KeyboardInterrupt, SystemError):
                raise
            except:
                try:
                    result = getattr(talib,indicator)(df.High, df.Low, df.Close)
                    if isinstance(result, tuple):
                        for i, val in enumerate(result):
                            df[f'{indicator}_{i}'] = val
                    else:
                        df[f'{indicator}'] = val
                except (KeyboardInterrupt, SystemError):
                    raise
                except:
                    try:
                        result = getattr(talib,indicator)(df.Open, df.High, df.Low, df.Close)
                        if isinstance(result, tuple):
                            for i, val in enumerate(result):
                                df[f'{indicator}_{i}'] = val
                        else:
                            df[f'{indicator}'] = val
                    except (KeyboardInterrupt, SystemError):
                        raise
                    except:
                        print(f'issue with {indicator}')

    df.OBV = talib.OBV(df.Close, df.Volume)
    df.AD = talib.AD(df.High, df.Low, df.Close, df.Volume)
    df.ADOSC  = talib.ADOSC (df.High, df.Low, df.Close, df.Volume)
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df.Close)
    df['HT_DCPHASE']  = talib.HT_DCPHASE(df.Close)

    return df