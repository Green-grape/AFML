import time
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

session = curl_requests.Session(impersonate="chrome")

# Download ETF data from ETFdb.com

# 1. Download the ETF name and URL mapping from ETFdb.com
etf_db_url = "https://etfdb.com/etfs/industry/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

response = requests.get(etf_db_url, headers=headers)
with open("etf_db_response.html", "wb") as file:
    file.write(response.text.encode("utf-8"))
soup = BeautifulSoup(response.content, "html.parser")
etf_tbody = soup.find("tbody")
etf_rows = etf_tbody.find_all("tr")
etf_link_dict = {}
for row in etf_rows:
    etf_industry_cell = row.find("td", {"data-th": "Industry"})
    etf_industry_cell_a = etf_industry_cell.find("a")
    industry_name = etf_industry_cell_a.text.strip()
    etf_table_url = etf_industry_cell_a["href"]
    query_url = "#etfs__expenses&sort_name=inception&sort_order=asc&page=1"
    etf_link_dict[industry_name] = urllib.parse.urljoin(
        etf_db_url, etf_table_url, query_url
    )

# 2. Download the best ETFs for each industry
etf_dict = {}
for industry, url in etf_link_dict.items():
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    table_body = soup.find("tbody")
    table_cells = table_body.find("tr").find("td", {"data-th": "Symbol"}).text.strip()
    etf_dict[industry] = table_cells
    # time.sleep(1)  # Sleep to avoid overwhelming the server
with open("etf_dict.json", "w") as file:
    import json

    json.dump(etf_dict, file, indent=4)

# 3. Download the GICS mapping price data from the yfinance API


import yfinance as yf


def download_gics_data(etf_dict):
    ret = pd.DataFrame()
    for industry, etf_symbol in etf_dict.items():
        print(f"Downloading for {industry} ETF: {etf_symbol}")
        try:
            etf_ticker = yf.Ticker(etf_symbol)
            etf_data = etf_ticker.history(period="max", interval="1d")
            if etf_data.empty:
                print(f"No data found for {etf_symbol}. Skipping.")
                continue
            etf_data.index = pd.to_datetime(pd.Series(etf_data.index.values)).dt.date
            etf_data_close = etf_data["Close"].rename(etf_symbol)
            ret = pd.concat([ret, etf_data_close], axis=1, join="outer")
            print(ret)
        except Exception as e:
            print(f"Error downloading data for {industry}: {e}")
        time.sleep(1)  # Sleep to avoid overwhelming the server
    return ret.sort_index()


gics_data = download_gics_data(etf_dict)
gics_data.to_csv("industry_etf_price.csv")
