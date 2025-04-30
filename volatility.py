#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt


def download_data(ticker: str, start: str) -> pd.DataFrame:
    """
    Скачивает исторические данные по символу ticker с Yahoo Finance.
    """
    df = yf.download(
        ticker,
        start=start,
        end=date.today().isoformat(),
        progress=True,
        auto_adjust=False  # чтобы не менять Close
    )
    if df.empty:
        raise RuntimeError(f"No data for ticker {ticker}")
    return df


def compute_annual_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает годовую реализованную волатильность (%) и
    возвращает DataFrame с колонками Year, Volatility (%).
    """
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    annual_vol = log_returns.groupby(log_returns.index.year).std() * np.sqrt(252) * 100
    vol_table = annual_vol.round(2).reset_index()
    vol_table.columns = ['Year', 'Volatility (%)']
    return vol_table


def plot_volatility(vol_table: pd.DataFrame, output_png: str = 'annual_volatility.png'):
    """
    Строит и сохраняет линейный график годовой волатильности.
    """
    plt.figure()
    plt.plot(vol_table['Year'], vol_table['Volatility (%)'])
    plt.title('Годовая реализованная волатильность XAU/USD')
    plt.xlabel('Год')
    plt.ylabel('Волатильность (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)  # сохранит файл в папке проекта
    plt.show()               # покажет график в интерактивном режиме


def main():
    ticker = "GC=F"         # фьючерс на золото (XAU/USD)
    start_date = "1975-01-01"

    print(f"⬇️  Downloading data for {ticker} since {start_date}...")
    df = download_data(ticker, start=start_date)

    print("🔢  Computing annual volatility...")
    vol_table = compute_annual_volatility(df)

    print("\n📊  Годовая реализованная волатильность XAU/USD:\n")
    print(vol_table.to_string(index=False))

    # Сохраняем CSV
    csv_file = "annual_volatility.csv"
    vol_table.to_csv(csv_file, index=False)
    print(f"\n✅  Результаты сохранены в {csv_file}")

    # Строим график
    print("📈  Строим график и сохраняем в PNG...")
    plot_volatility(vol_table)
    print("✅  График сохранён как annual_volatility.png")


if __name__ == "__main__":
    main()
