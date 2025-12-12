import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import aiohttp

PROMPT_TEMPLATE = (
    "你是一名法律条文判定助手。请阅读下方问题，只需判断是否必须引用具体法律条文，并列出精确的法条名称。"
    "输出需满足：\n"
    "1. 仅返回一行JSON；\n"
    "2. law_appearance：若需要引用法律条文为1，否则为0；\n"
    "3. legal_text：当law_appearance为1时，必须是一个JSON数组，数组中每个元素均为严谨的法条定位字符串，格式为“法规全称（含修订年份等） 空格 章节 空格 条次”，例如：\n"
    "   - “营业性演出管理条例(2020修订) 第五章  第四十七条”\n"
    "   - “国际收支统计申报办法(2013修订)  第八条”\n"
    "   - “中华人民共和国行政许可法(2019修正) 第一章  第二条”\n"
    "   数组内禁止出现多余解释、冒号、引号或原文摘录；\n"
    "4. 当law_appearance为0时，legal_text必须是一个空数组。\n"
    "JSON示例：{{\"law_appearance\":1,\"legal_text\":[\"中华人民共和国刑法(2023修正) 第二章  第二百三十四条\",\"中华人民共和国刑法(2023修正) 第二章  第六十七条\"]}}。\n"
    "严禁输出任何JSON外的文字或代码块。\n"
    "以下是问题内容：\n{question}\n"
    "评估要求（逐条满足，缺一不可）：\n{hints}"
)

SAMPLE_PER_FILE = 100
SAMPLE_SEED = 20251202


def load_questions(dataset_path: str, max_items: int) -> List[Dict[str, Any]]:
    dataset_file = Path(dataset_path)
    with dataset_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return _load_ucl_questions(data, os.path.basename(dataset_path), max_items)

    if isinstance(data, list) and data and isinstance(data[0], str):
        return _load_instruction_questions(dataset_file, data, max_items)

    if isinstance(data, list):
        # Dataset already contains instruction/question entries.
        return _serialize_instruction_entries(
            data,
            os.path.basename(dataset_path),
            max_items,
            sample_size=SAMPLE_PER_FILE,
            seed_key=dataset_path,
        )

    raise ValueError("Unsupported dataset format; expected dict or list")


def _load_ucl_questions(data: Dict[str, Any], dataset_name: str, max_items: int) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    for task_name, entries in data.items():
        for entry in entries:
            question_text = entry.get("model_prompt") or ""
            needs = (entry.get("needs") or "").strip()

            if question_text:
                # Dataset prompts often以“你的问题是：”结尾，追加needs保证问句完整。
                if needs and needs not in question_text:
                    suffix = "\n" if not question_text.endswith("\n") else ""
                    question_text = f"{question_text}{suffix}{needs}"
            else:
                info = entry.get("information") or ""
                question_text = f"任务类型：{task_name}\n案件信息：{info}\n需求：{needs}".strip()
            hints_text = (entry.get("evaluation_hints") or "").strip() or "无"
            question_id = f"{dataset_name}-{task_name}-{entry.get('id')}"
            questions.append(
                {
                    "id": entry.get("id"),
                    "task_name": task_name,
                    "question": question_text,
                    "hints": hints_text,
                    "question_id": question_id,
                }
            )
            if len(questions) >= max_items:
                return questions
    return questions


def _load_instruction_questions(
    list_file: Path, relative_paths: Iterable[str], max_items: int
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    base_dir = list_file.parent
    for rel_path in relative_paths:
        dataset_path = (base_dir / rel_path).resolve()
        with dataset_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        questions.extend(
            _serialize_instruction_entries(
                entries,
                os.path.basename(str(dataset_path)),
                max_items,
                current=len(questions),
                sample_size=SAMPLE_PER_FILE,
                seed_key=str(dataset_path),
            )
        )
        if len(questions) >= max_items:
            break
    return questions[:max_items]


def _serialize_instruction_entries(
    entries: List[Dict[str, Any]],
    dataset_name: str,
    max_items: int,
    current: int = 0,
    sample_size: int | None = None,
    seed_key: str | None = None,
) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for original_idx, entry in _select_entries(entries, sample_size, seed_key or dataset_name):
        if current + len(serialized) >= max_items:
            break
        instruction = (entry.get("instruction") or "").strip()
        question_body = (entry.get("question") or "").strip()
        merged_question = f"{instruction}\n{question_body}".strip()
        hints_text = (entry.get("evaluation_hints") or "").strip() or "无"
        serialized.append(
            {
                "id": entry.get("id", original_idx),
                "task_name": entry.get("task_name", dataset_name),
                "question": merged_question,
                "hints": hints_text,
                "question_id": entry.get("question_id") or f"{dataset_name}-{original_idx}",
            }
        )
    return serialized


def _select_entries(
    entries: List[Dict[str, Any]], sample_size: int | None, seed_key: str
) -> List[tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(entries))
    if not sample_size or len(entries) <= sample_size:
        return indexed
    rng = random.Random()
    rng.seed(f"{SAMPLE_SEED}-{seed_key}")
    selected_indices = sorted(rng.sample(range(len(entries)), sample_size))
    return [(idx, entries[idx]) for idx in selected_indices]


async def call_deepseek(
    session: aiohttp.ClientSession,
    api_key: str,
    question: str,
    hints: str,
    model: str,
    retries: int = 3,
    timeout: int = 60,
) -> Dict[str, Any]:
    escaped_question = question.replace("{", "{{").replace("}", "}}")
    escaped_hints = hints.replace("{", "{{").replace("}", "}}")
    prompt = PROMPT_TEMPLATE.format(question=escaped_question, hints=escaped_hints)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful Chinese legal assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: str | None = None

    for attempt in range(1, retries + 1):
        try:
            async with session.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                parsed = extract_json(content)
                if parsed is None:
                    raise ValueError(f"Model output is not valid JSON: {content}")
                if "law_appearance" not in parsed:
                    parsed["law_appearance"] = 1 if parsed.get("legal_text") else 0
                return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = f"Attempt {attempt} failed: {exc}"
            await asyncio.sleep(2 * attempt)
    raise RuntimeError(last_error or "Unknown error when calling DeepSeek")


def extract_json(text: str) -> Dict[str, Any] | None:
    """Extract the first JSON object from the text."""
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


async def process_questions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DeepSeek API key not provided. Use --api_key or set DEEPSEEK_API_KEY.")

    questions = load_questions(args.dataset, args.limit)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in questions:
            question = item["question"]

            async def runner(entry=item, q=question):
                async with semaphore:
                    parsed = await call_deepseek(session, api_key, q, entry.get("hints", "无"), args.model)
                    law_flag = parsed.get("law_appearance")
                    if isinstance(law_flag, str) and law_flag.isdigit():
                        law_flag = int(law_flag)
                    if law_flag not in (0, 1):
                        law_flag = 1 if parsed.get("legal_text") else 0
                    legal_text = parsed.get("legal_text") or []
                    if isinstance(legal_text, str):
                        legal_text = [legal_text.strip()] if legal_text.strip() else []
                    elif isinstance(legal_text, list):
                        legal_text = [str(item).strip() for item in legal_text if str(item).strip()]
                    else:
                        legal_text = []
                    if law_flag == 0:
                        legal_text = []
                    return {
                        "question_id": entry.get("question_id"),
                        "question": entry.get("question"),
                        "law_appearance": law_flag,
                        "legal_text": legal_text,
                    }

            tasks.append(asyncio.create_task(runner()))

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            print(f"[INFO] Completed {len(results)}/{len(questions)}")

    return results


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(results)} records to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch query DeepSeek for legal QA")
    parser.add_argument(
        "--dataset",
        default="UCL_bench_legal_data_sample.json",
        help="Path to the UCL bench JSON file",
    )
    parser.add_argument(
        "--output",
        default="deepseek_pipeline/sample_responses.json",
        help="Output JSON path",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of questions to send")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--api_key", default=None, help="DeepSeek API key")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of parallel requests (default 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = asyncio.run(process_questions(args))
    save_results(results, args.output)


if __name__ == "__main__":
    main()
