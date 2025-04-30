#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date


def download_data(ticker: str, start: str) -> pd.DataFrame:
    """
    Скачивает исторические данные по символу ticker с Yahoo Finance.
    """
    df = yf.download(
        ticker,
        start=start,
        end=date.today().isoformat(),
        progress=True
    )
    if df.empty:
        raise RuntimeError(f"No data for ticker {ticker}")
    return df


def compute_annual_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает годовую реализованную волатильность (%)
    и возвращает DataFrame с двумя колонками: Year и Volatility (%).
    """
    # Лог-доходности по цене закрытия
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    # Реализованная волатильность: std * sqrt(252) * 100%
    annual_vol = log_returns.groupby(log_returns.index.year).std() * np.sqrt(252) * 100

    # Формируем чистую таблицу через reset_index
    vol_table = annual_vol.round(2).reset_index()
    vol_table.columns = ['Year', 'Volatility (%)']

    return vol_table


def main():
    ticker = "GC=F"         # фьючерс на золото (XAU/USD)
    start_date = "1975-01-01"

    print(f"⬇️  Downloading data for {ticker} since {start_date}...")
    df = download_data(ticker, start=start_date)

    print("🔢  Computing annual volatility...")
    vol_table = compute_annual_volatility(df)

    # Выводим на экран
    print("\n📊  Годовая реализованная волатильность XAU/USD:\n")
    print(vol_table.to_string(index=False))

    # Сохраняем в CSV
    output_csv = "annual_volatility.csv"
    vol_table.to_csv(output_csv, index=False)
    print(f"\n✅  Результаты сохранены в {output_csv}")


if __name__ == "__main__":
    main()
