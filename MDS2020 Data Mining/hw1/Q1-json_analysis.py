# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 14:47:45 2025

@author: Neal

Analyze the 10 provided JSON files (as below) and answer the questions in the answer book accordingly.
    1. nvda_1.json
    2. nvda_2.json
    3. nvda_3.json
    4. nvda_4.json
    5. nvda_5.json
    6. tsla_1.json
    7. tsla_2.json
    8. tsla_3.json
    9. tsla_4.json
   10. tsla_5.json

Hints:
    1. Skip/remove any duplicate posts that share the same "id".
    2. Assume a "followers_count" of 0 for any user object lacking a valid "followers_count."
"""

import json

# with open("./data/nvda_1.json",'r', encoding = "utf-8") as rf:
#     nvdia_page_1 = json.load(rf)

# print('The 1st post object is: `nvdia_page_1["list"][0]`, and its "text" attribute is:\n {}'.format(
#     nvdia_page_1['list'][0]['text']) )

# print('\n\n The user object of the 2nd post is: `nvdia_page_1["list"][1]["user"]`, and its "screen_name" attribute is: \n {}'.format(
#     nvdia_page_1['list'][1]['user']['screen_name']) )

# %% Hints for Q1-1
# print("\n", "="*20)
# print("Hints for Q1-1")
# total_post = 0
# unique_posts = set()
# for post in nvdia_page_1["list"]:
#     total_post += 1
#     unique_posts.add(post["id"])

# print(f"There are {len(unique_posts)} unique posts out of {total_post} total posts for `nvda_1.json`")

# ++insert your code here++
# concat all json files
all_post = []
for i in range(1, 6):
    with open(f"./data/nvda_{i}.json", "r", encoding="utf-8") as rf:
        cur_post = json.load(rf)
        all_post.extend(cur_post["list"])

    with open(f"./data/tsla_{i}.json", "r", encoding="utf-8") as rf:
        cur_post = json.load(rf)
        all_post.extend(cur_post["list"])
# Q1-1
unique_posts_dict = {}
for post in all_post:
    unique_posts_dict[post["id"]] = post
print(
    f"Q1-1: There are {len(unique_posts_dict)} unique posts out of {len(all_post)} total posts for all json files"
)

# Q1-2
unique_posts = list(unique_posts_dict.values())
count_iphone_ai = 0
for post in unique_posts:
    source = post.get("source", "")
    text = post.get("text", "")
    if source == "iPhone" and ("AI" in text or "人工智能" in text):
        count_iphone_ai += 1
print(
    f"Q1-2: Number of unique posts with source 'iPhone' and text containing 'AI' or '人工智能' is {count_iphone_ai}"
)

# Q1-3
view_counts_gt100 = []
for post in unique_posts:
    user = post.get("user", {})
    followers_count = user.get("followers_count", 0)
    if followers_count is None:
        followers_count = 0
    if followers_count > 100:
        view_counts_gt100.append(post.get("view_count", 0))
avg_view_gt100 = (
    round(sum(view_counts_gt100) / len(view_counts_gt100), 2)
    if view_counts_gt100
    else 0.00
)
print(
    f"Q1-3: Average view_count for users with followers_count > 100 is {avg_view_gt100:.2f}"
)

# Q1-4
view_counts_le100 = []
for post in unique_posts:
    user = post.get("user", {})
    followers_count = user.get("followers_count", 0)
    if followers_count is None:
        followers_count = 0
    if followers_count <= 100:
        view_counts_le100.append(post.get("view_count", 0))
avg_view_le100 = (
    round(sum(view_counts_le100) / len(view_counts_le100), 2)
    if view_counts_le100
    else 0.00
)
print(
    f"Q1-4: Average view_count for users with followers_count <= 100 is {avg_view_le100:.2f}"
)
