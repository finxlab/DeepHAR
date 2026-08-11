import pickle
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
import requests
from dateutil.relativedelta import relativedelta



class AlphaVantageCrawler:
    """Crawler that fetches intraday (minute-bar) data from Alpha Vantage, month by month.
    
    NOTE: 
        written against Alpha Vantage's API as of Jan 2026. For the latest
        version, check https://www.alphavantage.co/documentation/
    
    NOTE: 
        crawl() has no retry/error handling -- it can fail partway through for
        various reasons (bad api_key, rate limit, network error, etc.), so if it
        stops, just call crawl() again manually.

    Example:
        crawler = AlphaVantageCrawler(api_key='YOUR_KEY', interval=5)
        dataset = crawler.crawl('2005-01', '2025-12', 'SPY')
        dataset = crawler.fillna(dataset)
    """

    def __init__(self, api_key, symbols, interval=5, sleep_sec=0.5):
        self.api_key = api_key
        self.symbols = symbols
        self.interval = interval
        self.sleep_sec = sleep_sec

    def _build_url(self, symbol, month):
        return (
            'https://www.alphavantage.co/query'
            f'?function=TIME_SERIES_INTRADAY&symbol={symbol}'
            f'&interval={self.interval}min&month={month}'
            '&outputsize=full&extended_hours=false'
            f'&apikey={self.api_key}'
        )

    def crawl(self, start_time, end_time, symbol):
        dataset = pd.DataFrame()
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        while True:
            if start_time > end_time:
                break
            month = start_time.strftime('%Y-%m')
            url = self._build_url(symbol, month)
            r = requests.get(url)

            data = r.json()
            data_n = pd.DataFrame(data[f'Time Series ({self.interval}min)']).T
            data_n.index = pd.to_datetime(data_n.index)
            dataset = pd.concat([data_n, dataset], axis=0)

            print(start_time)
            time.sleep(self.sleep_sec)
            start_time += relativedelta(months=1)

        dataset.columns = ['open', 'high', 'low', 'close', 'volume']
        return dataset

    
    @staticmethod
    def fillna(_dataset):
        dataset = _dataset.copy()
        datetime_idx = dataset.index.to_series()
        date = datetime_idx.dt.strftime('%Y-%m-%d')
        time_ = datetime_idx.dt.strftime('%H:%M')
        dataset.loc[:, 'date'] = date
        dataset.loc[:, 'time'] = time_

        datetime_grid = date.unique().reshape(-1, 1) + ' ' + time_.unique().reshape(1, -1)
        datetime_idx_full = pd.to_datetime(datetime_grid.reshape(-1))
        nan_idx = datetime_idx_full[~datetime_idx_full.isin(dataset.index)]

        df_nan = pd.DataFrame(index=nan_idx, columns=['open', 'high', 'low', 'close', 'volume'])
        df_nan.loc[:, 'date'] = df_nan.index.to_series().dt.strftime('%Y-%m-%d')
        df_nan.loc[:, 'time'] = df_nan.index.to_series().dt.strftime('%H:%M')
        df_fillna = pd.concat([dataset, df_nan], axis=0).sort_index()

        return df_fillna

    def run(self, start_date, end_date, save_dir='dataset_etf'):
        os.makedirs(save_dir, exist_ok=True)

        for symbol in self.symbols:
            try:
                dataset = self.crawl(start_date, end_date, symbol)
            except RuntimeError as e:
                print(f"[{symbol}] skipped -> {e}")
                continue
            dataset = self.fillna(dataset)
            save_path = os.path.join(save_dir, f"dataset_{symbol}.csv")
            dataset.to_csv(save_path)
            print(f"[{symbol}] saved -> {save_path}")





class ETFDataCleaner:
    """Fill in missing 5-min timestamps for raw OHLCV data and restrict to regular
    trading hours. Ported from dataset_cleaning.ipynb's dataFillna() + time filter.
    """

    def __init__(self, symbols, session_start='09:30', session_end='15:55', freq='5min'):
        self.symbols = symbols
        self.session_times = pd.date_range(session_start, session_end, freq=freq).strftime('%H:%M').tolist()

    @staticmethod
    def fillna(_dataset):
        """Fill missing 5-min bars (e.g. thin trading / data gaps), forward-filling
        open/high/low/close from the last known close, and setting volume to 0 for
        the filled bars. (Same logic as the original dataFillna.)
        """
        dataset = _dataset.copy()
        datetime_idx = dataset.index.to_series()
        date = datetime_idx.dt.strftime('%Y-%m-%d')
        time_ = datetime_idx.dt.strftime('%H:%M')
        dataset.loc[:, 'date'] = date
        dataset.loc[:, 'time'] = time_

        datetime_grid = date.unique().reshape(-1, 1) + ' ' + time_.unique().reshape(1, -1)
        datetime_idx_full = pd.to_datetime(datetime_grid.reshape(-1))
        nan_idx = datetime_idx_full[~datetime_idx_full.isin(dataset.index)]

        df_nan = pd.DataFrame(index=nan_idx, columns=['open', 'high', 'low', 'close', 'volume'], dtype='float64')
        df_nan.loc[:, 'date'] = df_nan.index.to_series().dt.strftime('%Y-%m-%d')
        df_nan.loc[:, 'time'] = df_nan.index.to_series().dt.strftime('%H:%M')

        df_fillna = pd.concat([dataset, df_nan], axis=0).sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_fillna[col] = df_fillna[col].astype('float64')

        close_ffill = df_fillna['close'].ffill()
        df_fillna['open'] = df_fillna['open'].fillna(close_ffill)
        df_fillna['high'] = df_fillna['high'].fillna(close_ffill)
        df_fillna['low'] = df_fillna['low'].fillna(close_ffill)
        df_fillna['close'] = df_fillna['close'].fillna(close_ffill)
        df_fillna['volume'] = df_fillna['volume'].fillna(0)

        return df_fillna

    def clean(self, raw_path, save_path=None):
        """raw 5-min csv -> fillna -> restrict to regular session times."""
        raw = pd.read_csv(raw_path, index_col=0)
        raw.index = pd.to_datetime(raw.index)

        df = self.fillna(raw)
        df = df.loc[df['time'].isin(self.session_times)]

        if save_path is not None:
            df.to_csv(save_path)
        return df

    def run(self, raw_dir='dataset_etf', save_dir='dataset'):
        """Run clean() for every symbol and save dataset_{symbol}_5m.csv."""
        os.makedirs(save_dir, exist_ok=True)

        cleaned = {}
        for symbol in self.symbols:
            raw_path = os.path.join(raw_dir, f'dataset_{symbol}.csv')
            save_path = os.path.join(save_dir, f'dataset_{symbol}_5m.csv')
            cleaned[symbol] = self.clean(raw_path, save_path)
            print(f"[{symbol}] cleaned -> {save_path} (shape={cleaned[symbol].shape})")

        return cleaned
    
    
    

class DailyFeatureMerger:
    """Aggregate cleaned 5-min OHLCV data into daily realized-volatility-style
    features: intraday 5-min log returns, daily realized volatility (RV, sum of
    squared 5-min returns), daily open-to-close return, and daily total volume.
    Also produces descriptive statistics (mean/std/skew/kurtosis/ADF) for the daily
    RV series. Ported from dataset_merge.ipynb.
    """

    def __init__(self, symbols, freq_label='5m'):
        self.symbols = symbols
        self.freq_label = freq_label

    def build_intraday_return_and_rv(self, clean_dir='dataset'):
        """5-min close price -> log return series (the first bar of each day uses
        the actual open->first-close return instead of a diff(), since a plain
        diff() at the first bar of a new day would pick up the overnight gap) ->
        daily RV (sum of squared 5-min log returns).
        """
        dataset_price = pd.DataFrame()
        dataset_return_open = pd.DataFrame()
        unique_date = None

        for symbol in self.symbols:
            data_5min = pd.read_csv(os.path.join(clean_dir, f'dataset_{symbol}_5m.csv'), index_col=0)
            data_5min.index = pd.to_datetime(data_5min.index)

            dataset_price.loc[:, symbol] = data_5min.loc[:, 'close']

            data_5min_open = data_5min[data_5min.index.time == pd.to_datetime('09:30:00').time()]
            dataset_return_open.loc[:, symbol] = (
                np.log(data_5min_open['close']) - np.log(data_5min_open['open'])
            )
            unique_date = data_5min['date'].unique()

        dataset_return = np.log(dataset_price).diff()
        dataset_return.loc[dataset_return_open.index] = dataset_return_open
        dataset_return.index = pd.to_datetime(dataset_return.index)

        dataset_vol = dataset_return.resample('24h').apply(lambda x: np.sum(x ** 2))
        dataset_vol = dataset_vol.loc[unique_date]

        return dataset_return, dataset_vol

    def build_daily_return_and_volume(self, clean_dir='dataset'):
        """groupby('date') -> daily open-to-close log return, daily total volume."""
        dataset_return_daily = pd.DataFrame()
        dataset_volume_daily = pd.DataFrame()

        for symbol in self.symbols:
            data_5min = pd.read_csv(os.path.join(clean_dir, f'dataset_{symbol}_5m.csv'), index_col=0)
            daily = data_5min.groupby('date').agg(
                open_price=('open', 'first'),
                close_price=('close', 'last'),
                total_volume=('volume', 'sum'),
            )
            dataset_return_daily.loc[:, symbol] = np.log(daily['close_price'] / daily['open_price'])
            dataset_volume_daily.loc[:, symbol] = daily['total_volume']

        return dataset_return_daily, dataset_volume_daily

    def run(self, clean_dir='dataset', save_dir='dataset'):
        os.makedirs(save_dir, exist_ok=True)

        dataset_return, dataset_vol = self.build_intraday_return_and_rv(clean_dir)
        dataset_return_daily, dataset_volume_daily = self.build_daily_return_and_volume(clean_dir)

        dataset_return.to_csv(os.path.join(save_dir, f'dataset_return_{self.freq_label}.csv'))
        dataset_vol.to_csv(os.path.join(save_dir, f'dataset_vol_{self.freq_label}.csv'))
        dataset_return_daily.to_csv(os.path.join(save_dir, f'dataset_return_daily_{self.freq_label}.csv'))
        dataset_volume_daily.to_csv(os.path.join(save_dir, f'dataset_volume_{self.freq_label}.csv'))

        result = {
            'dataset_return': dataset_return,
            'dataset_vol': dataset_vol,
            'dataset_return_daily': dataset_return_daily,
            'dataset_volume_daily': dataset_volume_daily,
        }
        return result
    
    
    
    
class HARFeatureBuilder:
    """Compute realized volatility (RV) and bipower variation (BV_t) per symbol per
    day from 5-min log returns. 
    """

    def __init__(self, symbols):
        self.symbols = symbols

    def _build_symbol(self, df_5min_return, dataset_volume_daily, dataset_return_daily, symbol):
        dataset = pd.DataFrame()
        unique_date = np.unique(df_5min_return.index.date)

        for _date in unique_date:
            data_5min = df_5min_return.loc[df_5min_return.index.date == _date]
            intra_return_t = data_5min.values

            dataset.loc[_date, 'volatility'] = np.sum(np.square(data_5min))
            dataset.loc[_date, 'BV_t'] = np.sum(np.abs(intra_return_t[:-1] * intra_return_t[1:])) / 2 * np.pi

        dataset['volume'] = dataset_volume_daily.loc[:, symbol]
        dataset['return'] = dataset_return_daily.loc[:, symbol]
        return dataset

    def run(self, dataset_return, dataset_return_daily, dataset_volume_daily, save_path=None):
        """dataset_return comes from DailyFeatureMerger.build_intraday_return_and_rv();
        dataset_return_daily/dataset_volume_daily come from build_daily_return_and_volume().
        """
        dataset_return = dataset_return.copy()
        dataset_return.index = pd.to_datetime(dataset_return.index, utc=True)

        dataset_return_daily = dataset_return_daily.copy()
        dataset_volume_daily = dataset_volume_daily.copy()
        dataset_return_daily.index = pd.to_datetime(dataset_return_daily.index)
        dataset_volume_daily.index = pd.to_datetime(dataset_volume_daily.index)

        dataset_dict = {}
        for symbol in self.symbols:
            df_5min_return = dataset_return.loc[:, symbol]
            dataset_dict[symbol] = self._build_symbol(
                df_5min_return, dataset_volume_daily, dataset_return_daily, symbol
            )
            print(f"[{symbol}] RV/BV_t built (shape={dataset_dict[symbol].shape})")

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(dataset_dict, f)
            print(f"saved -> {save_path}")

        return dataset_dict
    
    
    

class DeepHARDataset:
    """Build sliding-window (seq_len-day) HAR-style tensors (RV, BV_t, return,
    volume) for DeepHAR model training. A single StandardScaler is fit across all
    symbols on the train period (shared scale), split into train/val/test folds
    by calendar-year ranges, saved as .npy per symbol plus a merged "all" version.
    """

    FEATURE_COLS = ['RV', 'BV_t', 'return', 'volume']

    def __init__(self, symbols, seq_len=22,
                 start_base_year=2005, train_years=12, val_years=3, test_years=6, n_folds=1):
        self.symbols = symbols
        self.seq_len = seq_len
        self.start_base_year = start_base_year
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.n_folds = n_folds

    def make_fold_dates(self):
        folds = []
        for i in range(self.n_folds):
            t_start_y = self.start_base_year + i
            t_end_y = t_start_y + self.train_years - 1
            v_start_y = t_end_y + 1
            v_end_y = v_start_y + self.val_years - 1
            te_start_y = v_end_y + 1
            te_end_y = te_start_y + self.test_years - 1
            folds.append({
                'start_train': pd.to_datetime(f'{t_start_y}-01-01 00:00:00+00:00'),
                'end_train': pd.to_datetime(f'{t_end_y}-12-31 23:59:59+00:00'),
                'start_val': pd.to_datetime(f'{v_start_y}-01-01 00:00:00+00:00'),
                'end_val': pd.to_datetime(f'{v_end_y}-12-31 23:59:59+00:00'),
                'start_test': pd.to_datetime(f'{te_start_y}-01-01 00:00:00+00:00'),
                'end_test': pd.to_datetime(f'{te_end_y}-12-31 23:59:59+00:00'),
            })
        return folds

    def _build_daily_df(self, symbol, dataset_dict):
        d = dataset_dict[symbol]
        df_daily = pd.DataFrame(index=d.index)
        df_daily['RV'] = d['volatility']
        df_daily['BV_t'] = d['BV_t']
        df_daily['return'] = d['return']
        df_daily['volume'] = d['volume']
        df_daily.index = pd.to_datetime(df_daily.index, utc=True)
        return df_daily

    def _fit_scaler(self, daily_by_symbol, start_train, end_train):
        stacks = []
        for symbol in self.symbols:
            df_daily = daily_by_symbol[symbol]
            mask = (df_daily.index >= start_train) & (df_daily.index <= end_train)
            stacks.append(df_daily.loc[mask, self.FEATURE_COLS].values)
        combined = np.vstack(stacks)
        scaler = preprocessing.StandardScaler()
        scaler.fit(combined)
        return scaler

    def _make_windows(self, df_daily, start, end, scaler, datatype='train'):
        start_index_n = df_daily.index.get_loc(df_daily.index[df_daily.index >= start][0])
        if datatype == 'train':
            start_index_n += self.seq_len
        end_index_n = df_daily.index.get_loc(df_daily.index[df_daily.index <= end][-1])

        N = end_index_n - start_index_n + 1
        input_x = np.zeros((N, self.seq_len, len(self.FEATURE_COLS)), dtype='float32')
        label = np.zeros((N, 1), dtype='float32')

        for idx, n in enumerate(range(start_index_n, end_index_n + 1)):
            window_start_n = n - self.seq_len
            target_n = n
            input_x[idx] = scaler.transform(df_daily.iloc[window_start_n:target_n].values)
            label[idx] = df_daily.iloc[target_n].loc['RV'] * 100 ** 2

        return input_x, label

    def build_fold(self, dataset_dict, fold_dates, save_dir='dataset'):
        daily_by_symbol = {sym: self._build_daily_df(sym, dataset_dict) for sym in self.symbols}
        scaler = self._fit_scaler(daily_by_symbol, fold_dates['start_train'], fold_dates['end_train'])

        per_symbol = {}
        for symbol in self.symbols:
            df_daily = daily_by_symbol[symbol]
            train_x, train_label = self._make_windows(df_daily, fold_dates['start_train'], fold_dates['end_train'], scaler, 'train')
            val_x, val_label = self._make_windows(df_daily, fold_dates['start_val'], fold_dates['end_val'], scaler, 'val')
            test_x, test_label = self._make_windows(df_daily, fold_dates['start_test'], fold_dates['end_test'], scaler, 'test')

            save_path = os.path.join(save_dir, symbol)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, 'train_data.npy'), train_x)
            np.save(os.path.join(save_path, 'train_label.npy'), train_label)
            np.save(os.path.join(save_path, 'validation_data.npy'), val_x)
            np.save(os.path.join(save_path, 'validation_label.npy'), val_label)
            np.save(os.path.join(save_path, 'test_data.npy'), test_x)
            np.save(os.path.join(save_path, 'test_label.npy'), test_label)

            per_symbol[symbol] = {'train': (train_x, train_label), 'val': (val_x, val_label), 'test': (test_x, test_label)}
            print(f"[{symbol}] shapes -> train {train_x.shape}, val {val_x.shape}, test {test_x.shape}")

        merged = self._merge_symbols(per_symbol, save_dir)
        return per_symbol, merged, scaler

    def _merge_symbols(self, per_symbol, save_dir):
        save_path = os.path.join(save_dir, 'all')
        os.makedirs(save_path, exist_ok=True)
        merged = {}
        for split in ['train', 'val', 'test']:
            xs = [per_symbol[sym][split][0] for sym in self.symbols]
            ys = [per_symbol[sym][split][1] for sym in self.symbols]
            merged[split] = (np.concatenate(xs, axis=0), np.concatenate(ys, axis=0))
        np.save(os.path.join(save_path, 'train_data.npy'), merged['train'][0])
        np.save(os.path.join(save_path, 'train_label.npy'), merged['train'][1])
        np.save(os.path.join(save_path, 'validation_data.npy'), merged['val'][0])
        np.save(os.path.join(save_path, 'validation_label.npy'), merged['val'][1])
        np.save(os.path.join(save_path, 'test_data.npy'), merged['test'][0])
        np.save(os.path.join(save_path, 'test_label.npy'), merged['test'][1])
        print(f"[all] shapes -> train {merged['train'][0].shape}, val {merged['val'][0].shape}, test {merged['test'][0].shape}")
        return merged

    def save_sample(self, merged, save_dir='data_sample', n=100, seed=None):
        rng = np.random.default_rng(seed)
        save_path = os.path.join(save_dir, 'all')
        os.makedirs(save_path, exist_ok=True)
        for split in ['train', 'val', 'test']:
            x, y = merged[split]
            idx = rng.choice(x.shape[0], size=min(n, x.shape[0]), replace=False)
            np.save(os.path.join(save_path, f'{"validation" if split == "val" else split}_data.npy'), x[idx])
            np.save(os.path.join(save_path, f'{"validation" if split == "val" else split}_label.npy'), y[idx])
        print(f"sample saved -> {save_path} (n={n} per split)")

    def run(self, dataset_dict, save_dir='dataset'):
        folds = self.make_fold_dates()
        results = []
        for fold_dates in folds:
            results.append(self.build_fold(dataset_dict, fold_dates, save_dir))
        return results