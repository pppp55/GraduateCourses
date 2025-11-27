# test_api.py
import requests
import json


def test_api():
    base_url = "http://localhost:5724"

    # 测试健康检查
    print("1. 测试健康检查...")
    response = requests.get(f"{base_url}/health")
    print(f"状态: {response.status_code}")
    print(f"响应: {response.json()}\n")

    # 测试单条预测 - 使用明确的测试文本
    print("2. 测试单条预测...")
    test_data = {"text": "太极实业:2022年度独立董事述职报告"}
    # test_data = "太极实业:2022年度独立董事述职报告"

    response = requests.post(f"{base_url}/predict_topic", json=test_data)
    if response.status_code == 200:
        result = response.json()
        print(f"预测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        print(f"错误: {response.text}")


if __name__ == "__main__":
    test_api()
