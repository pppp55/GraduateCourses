import requests
import logging
from requests.exceptions import JSONDecodeError, RequestException
import argparse
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def search(retrieve_path, payload):
    """调用本地检索服务"""

    # payload = {"queries": [query], "topk": 3, "return_scores": True}
    try:
        response = requests.post(
            retrieve_path,
            json=payload,
            proxies={"http": None, "https": None},
            timeout=10
        )
        response.raise_for_status()
        json_data = response.json()
        results = json_data.get("result", [])
    except requests.exceptions.Timeout:
        print("[ERROR] Search request timed out.")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return ""
    except ValueError as e:
        print(f"[ERROR] Failed to decode JSON: {e}")
        return ""

    if not results:
        print("[INFO] No results returned from search.")
        return ""

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['content']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            
            score=doc_item['score']
            score=(round(float(score), 4))
            
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n score={score}\n"
        return format_reference

    return _passages2string(results[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "test requests.")

    parser.add_argument('--queries', default= '抢劫罪',type=str)
    parser.add_argument('--test_url', default= "http://127.0.0.1:8006/retrieve",type=str)
    parser.add_argument('--topk', default= 3,type=int)
    # parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    args = parser.parse_args()

    queries=args.queries
    topk=args.topk
    
    test_url=args.test_url
    
    payload = {
            "queries": [queries],
            "topk": topk,
            "return_scores": True
        }

    print(f'正在检索{queries}')
    try:
        
        print(search(test_url, payload))

    except Exception as err:
        print("Search service verification failed:", err)
        # 根据实际情况决定：exit(1) 终止训练，或继续但不执行检索逻辑
        import sys; sys.exit(1)
