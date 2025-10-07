# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:02:26 2025
@author: Neal

Given that the shareholder information of a given stock ticker (such as 000001,
 000002 and 000003) are provided in :
    https://q.stock.sohu.com/cn/000001/ltgd.shtml
    https://q.stock.sohu.com/cn/000002/ltgd.shtml
    https://q.stock.sohu.com/cn/000003/ltgd.shtml
    ...

Please collect the shareholder information tables for the stocks listed in
"selected_stocks," ensuring you include the following seven columns:


And you need to collect the tables of shareholder information for stocks in
 "selected_stocks", with following 7 columns,
    1. 'stock'-股票代码 / Stock code
    2. 'rank'-排名 / Shareholder rank
    3. 'org_name'-股东名称 / Shareholder name
    4. 'shares'-持股数量(万股) / Number of shares held (in units of 10,000 shares)
    5. 'percentage'-持股比例	 / Shareholding percentage
    6. 'changes'-持股变化(万股) / Change in shares held (in units of 10,000 shares)
    7. 'nature'-股本性质 / Nature of equity

Then perform the analysis on the collected shareholder information to answer
 the questions.

Note:
    1. Be mindful of the default data types for each column, particularly 'rank' and 'percentage.'
    2. Remember that the 'shares' column is measured in increments of 10,000 shares.
"""


from unittest import result
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

chrome_header = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, sdch",
    "Accept-Language": "zh-TW,zh;q=0.8,en-US;q=0.6,en;q=0.4,zh-CN;q=0.2",
}

data_file = "./data/stock_shareholders.csv"
selected_stocks = (
    "601398",
    "601857",
    "601728",
    "600276",
    "601166",
    "600887",
    "601816",
    "601328",
    "003816",
    "300750",
    "000333",
    "300999",
    "000651",
    "300760",
    "002415",
)


stock_share_prices = {
    "000333.CH": 73.41,
    "000651.CH": 40.11,
    "002415.CH": 30.92,
    "003816.CH": 3.68,
    "300750.CH": 365.0,
    "300760.CH": 238.5,
    "300999.CH": 32.15,
    "600276.CH": 71.1,
    "600887.CH": 27.73,
    "601166.CH": 20.28,
    "601328.CH": 6.89,
    "601398.CH": 7.28,
    "601728.CH": 6.78,
    "601816.CH": 5.23,
    "601857.CH": 8.19,
}


print("There are", len(selected_stocks), "stocks in selected_stocks")

# base_url = "https://q.stock.sohu.com/cn/{}/ltgd.shtml"
# row_count = 0
# # create a list to store the crawled share-holdoing records
# results = []

# # process stock one by one
# for stock in selected_stocks:
#     # prepare the request URL with desired parameters
#     url = base_url.format(stock)
#     print("Now we are scraping stock", stock)
#     # send http request with Chrome http header
#     response = requests.get(url, headers=chrome_header)
#     if response.status_code == 200:
#         response.encoding = "gbk"  # ++insert your code here++  look for charset in html
#         print("Encoding:", response.encoding)
#         root = BeautifulSoup(response.text, "html.parser")
#         # search the table storing the shareholder information
#         table = root.find("table", class_="tableG")  # ++insert your code here++
#         # list all rows the table, i.e., tr tags
#         if table is not None:
#             rows = table.find_all("tr")[1:]  # ++insert your code here++
#         else:
#             print(f"Warning: No table found for stock {stock}")
#             continue
#         cur_count = 0
#         for row in rows:  # iterate rows
#             # define a record with stock pre-filled and then store columns of the row/record
#             record = [stock + ".CH"]
#             # list all columns of the row , i.e., td tags
#             columns = row.find_all("td")  # ++insert your code here++
#             for col in columns:  # iterate colums
#                 record.append(col.get_text().strip())
#             # if has valid columns, save the record to list results
#             if len(record) == 7:
#                 results.append(record)  # ++insert your code here++
#                 cur_count += 1
#         row_count += cur_count
#         print(f"{cur_count} records found for stock {stock}")
#         time.sleep(1)

# sharehold_records_df = pd.DataFrame(
#     columns=["stock", "rank", "org_name", "shares", "percentage", "changes", "nature"],
#     data=results,
# )
# print("\n", "=" * 20)
# print(
#     "Crawled and saved {} records of shareholder information of selected_stocks ".format(
#         len(sharehold_records_df)
#     )
# )
# sharehold_records_df.to_csv(data_file, index=False, encoding="utf-8")
sharehold_records_df = pd.read_csv(data_file)

# Q2-1:
mask = sharehold_records_df["nature"].str.contains("境外可流")
unique_orgs = sharehold_records_df[mask]["org_name"].unique()
print(
    f"Q2-1: Number of unique organizations holding '境外可流' shares is {len(unique_orgs)}"
)

# Q2-2:
groups = sharehold_records_df.groupby("org_name").size().sort_values(ascending=False)
print(
    f"Q2-2: The most frequent organization is {groups.index[0]} which appears {groups.iloc[0]} times"
)

# Q2-3:
sharehold_records_df["percentage"] = (
    sharehold_records_df["percentage"].str.rstrip("%").astype(float)
)
top5_sum = (
    sharehold_records_df[sharehold_records_df["rank"] <= 5]
    .groupby("stock")["percentage"]
    .sum()
)
max_stock = top5_sum.idxmax()
max_sum = top5_sum.max()
print(
    f"Q2-3: The stock with highest total percentage held by top 5 shareholders is {max_stock} ({max_sum:.2f}%)"
)

# Q2-4: 哪个 org_name 持有 selected_stocks 中所有股票的总市值最大？
sharehold_records_df["stock_price"] = sharehold_records_df["stock"].map(
    stock_share_prices
)
sharehold_records_df["total_value"] = (
    sharehold_records_df["shares"] * sharehold_records_df["stock_price"]
)
org_total_value = sharehold_records_df.groupby("org_name")["total_value"].sum()
max_org = org_total_value.idxmax()
max_value = org_total_value.max()
print(
    f"Q2-4: The organization with highest total value of shares is {max_org} (¥{max_value:,.2f})"
)
