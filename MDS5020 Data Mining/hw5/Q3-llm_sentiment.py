# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:15:28 2025

@author: Neal
Please use ZhipuAI model-"glm-4-flash" and appropriate prompt to analyze the 
sentiment of news titles following steps as below.
    1)	Install the ZhipuAI SDK package as "pip install zhipuai" if necessary.
    2)	Register your ZhipuAI account and get your API token, and fill in it 
        in the code indicated as "++insert your API token ++".
    3)	Complete the definition of function "analyze_sentiment()" with 
        appropriate prompt.
    4)	Fine-tune your prompt and code in function "analyze_sentiment()" based 
        on its performance (in terms of accuracy) on (all or selected) labelled
        news sentiment samples in the file "benchmark_news.xlsx" 
    5)	If the performance is satisfactory (e.g., accuracy >0.9), then 
        apply the function "analyze_sentiment()" to analyze the 50 "news_title" 
        in "test_news.xlsx", and saved the returned "score","reason" as two 
        new columns in the name of "pred_sentiment" and "sentiment_reason", respectively.
    6)	Save the results in a new Excel file named as 
        "test_news_with_predictions.xlsx".This file is expected to 
        have 50 rows for 50 "news_title" with 4 columns as below
        ('news_id', 'news_title', 'pred_sentiment', 'sentiment_reason'). 

Note:
    1. Improve your prompt as in lecture notes of Week-12, especially the 
        slide-“Prompt Engineering in General”,
    2. Try to restrict the output format (such as json) in prompt, such that  
       you could parse it reliably and easily
    3. Use "try ... except ..." clause to deal with unexpected situations
    4. Refer to the "./data/[Example]test_news_with_predictions.xlsx" as 
       an example of file format(/columns) to be submitted 
"""

import json
import pandas as pd
from zhipuai import ZhipuAI
from sklearn.metrics import accuracy_score
import os
expected_score = {1, -1}


def analyze_sentiment(glm_model, text): 
    """
    Analyze the sentiment of given `text` with glm_model ("glm-4-flash")
    Parameters
    ----------
    glm_model : zhipuai._client.ZhipuAI
        The authorized client to use ZhipuAI
    text : str
        The text to be analyzed

    Returns
    -------
    score : int
        The sentiment score, should be either 1 or -1
    reason : str
        The reason for the provided sentiment score

    """
    #++insert your code below ++ to compute `score` and `reason`
    # for `text` based on `glm_model`
    prompt = (
        "You are an expert financial news analyst."
        " Given a headline, label its sentiment toward the market as"
        " strictly positive (1) or negative (-1)."
        " Respond ONLY with valid JSON in the format\n"
        "{{\"sentiment\": <1 or -1>, \"reason\": \"<short explanation>\"}}."
        " Make sure the sentiment is 1 for optimistic tone and -1 for pessimistic"
        " tone. Headline: {headline}"
    ).format(headline=text.strip())

    score, reason = -1, ""
    try:
        response = glm_model.responses.create(
            model="glm-4-flash",
            input=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]
        )

        response_text = None
        if hasattr(response, 'output'):
            # Newer SDK response shape
            content_blocks = response.output or []
            for block in content_blocks:
                block_content = block.get('content') if isinstance(block, dict) else None
                if block_content:
                    response_text = block_content[0].get('text')
                    break
        if response_text is None and hasattr(response, 'choices'):
            # Fallback to legacy chat completions style
            for choice in response.choices:
                message = getattr(choice, 'message', None)
                if message and getattr(message, 'content', None):
                    response_text = message.content[0].get('text') if isinstance(message.content, list) else message.content
                    break
        if response_text is None:
            response_text = str(response)

        parsed = json.loads(response_text)
        sentiment_raw = parsed.get('sentiment')
        if isinstance(sentiment_raw, str):
            sentiment_raw = sentiment_raw.strip().lower()
            if sentiment_raw in {'positive', 'pos', 'bullish', '+1', '1'}:
                sentiment_raw = 1
            elif sentiment_raw in {'negative', 'neg', 'bearish', '-1'}:
                sentiment_raw = -1
        score = int(sentiment_raw)
        reason = parsed.get('reason', '').strip()
    except Exception as exc:
        reason = f"Fallback due to error: {exc}"
        score = -1
    
    assert score in expected_score
    return score, reason



if __name__ == "__main__":
    
    #++insert your API token ++ of  ZhipuAI after registration
    client = ZhipuAI(api_key="0d2b5ac60e244ec6bee5c104e39ccaf7.PMx1VbtJwx9MckVK") 
    
    # %% Evaluate and improve the performance of your solution on "benchmark_news.xlsx" and
    benchmark_file = os.path.join('./data', "benchmark_news.xlsx")
    df_benchmark = pd.read_excel(benchmark_file)
    print(f"\n========Processing {benchmark_file}==========")
    pred_sentiments = []
    true_sentiments = []
    count = 0
    for row in df_benchmark.itertuples(index=False):
        count+=1
        print(f"Process {count} rows with {row.news_title}")
        score, reason = analyze_sentiment(client, row.news_title)
        pred_sentiments.append(score)
        true_sentiments.append(row.news_sentiment)
    print(f"Accuracy_score amomg {len(true_sentiments)} news in {benchmark_file} is", 
          accuracy_score(true_sentiments, pred_sentiments))

    # %% Predict the `pred_sentiment` and `sentiment_reason` for `news_title` 
    # in "test_news.xlsx" and save to Excel (to be submit)
    test_file = os.path.join('./data', "test_news.xlsx")
    df_test = pd.read_excel(test_file)
    print(f"\n========Processing {test_file}==========")
    results = []
    count = 0
    for row in df_test.itertuples(index=False):
        count+=1
        if count > 5:
            break
        print(f"Process {count} rows with {row.news_title}")
        score, reason = analyze_sentiment(client, row.news_title)
        results.append((row.news_id, row.news_title, score, reason))
    df_results = pd.DataFrame(results,
                          columns =['news_id', 'news_title', 'pred_sentiment', 'sentiment_reason'])
    out_file = os.path.join('./data', "test_news_with_predictions.xlsx")
    df_results.to_excel(out_file, index = False)
    print(f'Save {len(results)} results to {out_file}')

        
