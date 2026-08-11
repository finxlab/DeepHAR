import os
import argparse

from lib.dataset_module import (
    AlphaVantageCrawler, ETFDataCleaner, DailyFeatureMerger, HARFeatureBuilder, DeepHARDataset,
)

parser = argparse.ArgumentParser()

# crawling
parser.add_argument('--api_key', type=str, default='INPUT YOUR API KEY')
parser.add_argument('--symbols', type=str, nargs='+', default=['SPY', 'DIA', 'QQQ'])
parser.add_argument('--start_date', type=str, default='2005-01', help='crawl start (YYYY-MM)')
parser.add_argument('--end_date', type=str, default='2025-12', help='crawl end (YYYY-MM)')
parser.add_argument('--interval', type=int, default=5, help='bar interval in minutes')
parser.add_argument('--sleep_sec', type=float, default=0.5)
parser.add_argument('--skip_crawl', action='store_true',
                     help='skip crawling (e.g. raw_dir/dataset_{symbol}.csv already exist)')

# paths
parser.add_argument('--raw_dir', type=str, default='dataset_etf', help='where crawl() output csvs live')
parser.add_argument('--save_dir', type=str, default='dataset', help='cleaned + feature + windowed outputs')

# DeepHARDataset windowing / folds
parser.add_argument('--seq_len', type=int, default=22)
parser.add_argument('--start_base_year', type=int, default=2005)
parser.add_argument('--train_years', type=int, default=12)
parser.add_argument('--val_years', type=int, default=3)
parser.add_argument('--test_years', type=int, default=6)
parser.add_argument('--n_folds', type=int, default=1)


if __name__ == '__main__':
    configs = parser.parse_args()

    if not configs.skip_crawl:
        crawler = AlphaVantageCrawler(
            api_key=configs.api_key, symbols=configs.symbols,
            interval=configs.interval, sleep_sec=configs.sleep_sec,
        )
        crawler.run(configs.start_date, configs.end_date, save_dir=configs.raw_dir)

    cleaner = ETFDataCleaner(configs.symbols)
    cleaner.run(raw_dir=configs.raw_dir, save_dir=configs.save_dir)

    merger = DailyFeatureMerger(configs.symbols)
    merge_result = merger.run(clean_dir=configs.save_dir, save_dir=configs.save_dir)

    os.makedirs(configs.save_dir, exist_ok=True)
    har_builder = HARFeatureBuilder(configs.symbols)
    dataset_dict = har_builder.run(
        merge_result['dataset_return'], merge_result['dataset_return_daily'],
        merge_result['dataset_volume_daily'],
        save_path=os.path.join(configs.save_dir, 'dataset_dict_5m.pickle'),
    )

    dataset_builder = DeepHARDataset(
        configs.symbols, seq_len=configs.seq_len, start_base_year=configs.start_base_year,
        train_years=configs.train_years, val_years=configs.val_years,
        test_years=configs.test_years, n_folds=configs.n_folds,
    )
    dataset_builder.run(dataset_dict, save_dir=configs.save_dir)