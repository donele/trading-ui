import pandas as pd
import numpy as np
import json
from datetime import timedelta
import pyarrow.parquet as pq

def get_sim_sdates(head_dir):
    from pathlib import Path

    log_dir = Path(head_dir) / 'log'
    sdates = []
    for path in log_dir.glob('order.*.parquet'):
        parts = path.name.split('.')
        if len(parts) == 3 and parts[0] == 'order' and parts[2] == 'parquet':
            sdates.append(parts[1])
    return sorted(set(sdates))

def get_sim_symbols(head_dir):
    from pathlib import Path

    state_dir = Path(head_dir) / 'log' / 'state'
    symbols = []
    for path in state_dir.glob('*.parquet'):
        parts = path.name.split('.')
        if len(parts) >= 3 and parts[-1] == 'parquet':
            symbols.append(parts[0])
    return sorted(set(symbols))


def _read_parquet_frame(path, columns=None):
    table = pq.ParquetFile(path).read(columns=columns)
    df = table.to_pandas()
    if df.columns.has_duplicates:
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

class SimData:
    def __init__(self, head_dir):
        self.head_dir = head_dir
        self.sdates = get_sim_sdates(head_dir)
        self.symbols = get_sim_symbols(head_dir)
        self.dfo = None # Read from orders.<yyyymmdd>.parquet
        self.dfs = None # Read from <symbol>.0.<yyyymmdd>.parquet
    def load_order(self, sdate):
        exists = self.dfo is not None and ((self.dfo.index.get_level_values(1).normalize() == pd.to_datetime(sdate, format='%Y%m%d'))).any()
        if not exists:
            dfo = _read_parquet_frame(f'{self.head_dir}/log/order.{sdate}.parquet')
            dfo['create_datetime'] = pd.to_datetime(dfo.create_time, unit='us')
            dfo = dfo.set_index(['symbol', 'create_datetime'])
            self.dfo = dfo if self.dfo is None else pd.concat([self.dfo, dfo]).sort_index()
    def load_state(self, symbol, sdate):
        exists = self.dfs is not None and ((self.dfs.index.get_level_values(0) == symbol) & (self.dfs.index.get_level_values(1).normalize() == pd.to_datetime(sdate, format='%Y%m%d'))).any()
        if not exists:
            dfs = _read_parquet_frame(f'{self.head_dir}/log/state/{symbol}.0.{sdate}.parquet')
            dfs.time = pd.to_datetime(dfs.time, unit='us')
            dfs = dfs.set_index(['symbol', 'time']).rename(columns={'fees_pnl': 'fees', 'funding_pnl': 'funding_cost', 'notional': 'notional_pos'})
            dfs.loc[dfs.bid > 1e8, 'bid'] = np.nan
            dfs.loc[dfs.ask > 1e8, 'ask'] = np.nan
            self.dfs = dfs if self.dfs is None else pd.concat([self.dfs, dfs]).sort_index()
    def load_all(self):
        for sdate in self.sdates:
            self.load_order(sdate)
            for symbol in self.symbols:
                self.load_state(symbol, sdate)
    def get_timeline(self, symbol, sdate, freq='5min'):
        self.load_state(symbol, sdate)
        t1 = pd.to_datetime(sdate, format='%Y%m%d')
        t2 = t1 + timedelta(days=1)
        if self.dfs is None:
            return pd.DataFrame(columns=['size_traded', 'notional_traded', 'notional_pos', 'mid', 'pnl', 'fees'])
        try:
            df_symbol = self.dfs.loc[symbol]
        except KeyError:
            return pd.DataFrame(columns=['size_traded', 'notional_traded', 'notional_pos', 'mid', 'pnl', 'fees'])
        df_symbol = df_symbol[(df_symbol.index >= t1) & (df_symbol.index < t2)]
        if df_symbol.empty:
            return pd.DataFrame(columns=['size_traded', 'notional_traded', 'notional_pos', 'mid', 'pnl', 'fees'])
        df1 = df_symbol.resample(freq).ffill()
        df1['mid'] = df1[['bid', 'ask']].mean(axis=1)
        df1 = df1[['size_traded', 'notional_traded', 'notional_pos', 'mid', 'pnl', 'fees']]
        return df1
    def get_timelines(self, sdate, freq='5min'):
        for symbol in self.symbols:
            self.load_state(symbol, sdate)
        timelines = {symbol: self.get_timeline(symbol, sdate, freq) for symbol in self.symbols}
        return timelines
    def get_orders_bid_ask(self, symbol, sdate, hour):
        self.load_state(symbol, sdate)
        self.load_order(sdate)
        t1 = pd.to_datetime(sdate, format='%Y%m%d').replace(hour=hour)
        t2 = t1 + timedelta(hours=1)
        if self.dfo is None or self.dfs is None:
            return pd.DataFrame(), pd.DataFrame(columns=['bid', 'ask'])
        try:
            dfo = self.dfo.loc[symbol].loc[t1:t2]
        except KeyError:
            dfo = pd.DataFrame()
        try:
            dfbidask = self.dfs.loc[symbol].loc[t1:t2][['bid', 'ask']]
        except KeyError:
            dfbidask = pd.DataFrame(columns=['bid', 'ask'])
        return dfo, dfbidask

def plot_symbol_date(simdata, symbol, sdate):
    import matplotlib.pyplot as plt

    dftimeline = simdata.get_timeline(symbol, sdate)
    
    ny = 6
    nx = 1
    plt.figure(figsize=(16, ny*2))
    iplot = 0

    for col in dftimeline.columns:
        plt.subplot(ny, nx, iplot:=iplot+1)
        plt.step(dftimeline.index, dftimeline[col])
        plt.title(f'{col} {symbol} {sdate}')
        plt.grid()
    plt.tight_layout()

def plot_portfolio_date(simdata, sdate):
    import matplotlib.pyplot as plt

    timelines = simdata.get_timelines(sdate, freq='5min')

    ny = 4
    nx = 1
    plt.figure(figsize=(16, ny*2))
    iplot = 0

    symbols = list(timelines.keys())
    for col in ['notional_traded', 'notional_pos', 'pnl']:
        plt.subplot(ny, nx, iplot:=iplot+1)
        for symbol in symbols:
            plt.step(timelines[symbol].index, timelines[symbol][col], label=symbol)
        total = pd.concat([timelines[symbol][col] for symbol in symbols], axis=1).sum(axis=1)
        plt.step(total.index, total, '.', color='k', markersize=2, label='total')
        plt.title(f'{col} {sdate}')
        plt.legend()
        plt.grid()
    plt.tight_layout()

def plot_hlines(df, label, iplot):
    import matplotlib.pyplot as plt

    mask_b = df['price'].notna()
    plt.hlines(
        y=df.loc[mask_b, 'price'],
        xmin=pd.to_datetime(df.loc[mask_b, 'start_time'], unit='us'),
        xmax=pd.to_datetime(df.loc[mask_b, 'end_time'], unit='us'),
        color=f'C{iplot}',
        linewidth=1,
        label=label,
    )

def plot_hour(simdata, symbol, sdate, hour):
    import matplotlib.pyplot as plt

    dfo1, dfbidask = simdata.get_orders_bid_ask(symbol, sdate, hour)
    
    dfo1b = dfo1.query('side=="BID"')
    dfo1a = dfo1.query('side=="ASK"')
    
    timepoints_b = np.sort(pd.concat([dfo1b.create_time, dfo1b.last_update_time]).unique())
    timepoints_a = np.sort(pd.concat([dfo1a.create_time, dfo1a.last_update_time]).unique())
    
    dfob = pd.DataFrame({'start_time': timepoints_b[:-1], 'end_time': timepoints_b[1:]})
    dfoa = pd.DataFrame({'start_time': timepoints_a[:-1], 'end_time': timepoints_a[1:]})
    
    dfob['price'] = dfob.apply(lambda x: dfo1b[(dfo1b.create_time <= x.start_time) & (dfo1b.last_update_time >= x.end_time)].price.max(), axis=1)
    dfoa['price'] = dfoa.apply(lambda x: dfo1a[(dfo1a.create_time <= x.start_time) & (dfo1a.last_update_time >= x.end_time)].price.min(), axis=1)

    plt.figure(figsize=(20, 4))

    plt.plot(dfbidask.bid, color='C0', alpha=0.3, label='bid')
    plt.plot(dfbidask.ask, color='C1', alpha=0.3, label='ask')

    plot_hlines(dfob, 'buy order', 0)
    plot_hlines(dfoa, 'sell order', 1)

    dfo1bf = dfo1b.query('filled_qty > 0')
    dfo1af = dfo1a.query('filled_qty > 0')
    plt.plot(pd.to_datetime(dfo1bf.last_update_time, unit='us'), dfo1bf.price, '^', color='C2', label='buy fill')
    plt.plot(pd.to_datetime(dfo1af.last_update_time, unit='us'), dfo1af.price, 'v', color='C3', label='sell fill')

    plt.grid()
    plt.legend()
