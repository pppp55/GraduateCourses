"""Batch generate DeepSeek keyword queries for legal articles."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

PROMPT_TEMPLATE_GENERAL = (
    "你是一名面向法律咨询场景的检索 Query 生成助手。请阅读下列法条原文，提炼便于检索法律条文的核心关键信息。要求：\n"
    "1. 仅输出一行 JSON；\n"
    "2. keywords：长度恰为 {keyword_count} 的字符串数组，使用简体中文名词或短语，按重要性排序；\n"
    "3. query：将 keywords 按单个空格连接得到的字符串；\n"
    "4. 关键词需覆盖主体、行为/义务、法律后果或责任等要素，突出该法条最具区分度的特征；\n"
    "5. 不得添加案例、情节、编号等额外信息，也不要描述模型、角色或输出说明；\n"
    "6. 输出内容仅限 JSON（含 keywords 与 query 字段）。\n"
    "法条原文：\n{law_content}\n"
    "请严格遵守格式，仅输出符合要求的 JSON。"
)

PROMPT_TEMPLATE_SPECIFIC = (
    "你是一名法律条文定位助手。根据提供的【法条名称】与【条款内容】生成可以直接检索到该条款的 Query。要求：\n"
    "1. 仅输出一行 JSON；\n"
    "2. keywords：长度恰为 {keyword_count} 的字符串数组，使用简体中文，并按重要性排序；\n"
    "   - 至少包含能唯一定位该条款的法条名称和条款序号（如《刑法》第十条）；\n"
    "   - 其他关键词覆盖主体、行为/义务、法律后果等核心要素；\n"
    "3. query：按 keywords 的顺序使用单个空格拼接；\n"
    "4. 禁止加入案例、背景说明或输出格式提示，仅描述条款本身；\n"
    "5. 输出字段仅限 keywords 与 query。\n"
    "法条名称：{law_title}\n"
    "条款原文：\n{law_content}\n"
    "请严格按上述 JSON 结构返回。"
)

GENERAL_PROMPT_WEIGHT = 2
SPECIFIC_PROMPT_WEIGHT = 1

DEFAULT_SAMPLE_SEED = 20251203


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek queries for legal provisions"
    )
    parser.add_argument(
        "--dataset",
        default="法律法规.jsonl",
        help="Path to the JSONL file containing legal articles",
    )
    parser.add_argument(
        "--output",
        default="../data/law2query_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of query-id pairs to generate",
    )
    parser.add_argument(
        "--keyword_count",
        type=int,
        default=5,
        help="Number of keywords to request from the model each run",
    )
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--api_key", default=None, help="DeepSeek API key")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API calls",
    )
    parser.add_argument(
        "--http_timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for each DeepSeek request",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Random seed for sampling provisions",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of times to query the model per law article",
    )
    parser.add_argument(
        "--min_keywords",
        type=int,
        default=2,
        help="Discard entries whose intersection has this many or fewer keywords",
    )
    return parser.parse_args()


def load_law_entries(dataset_path: str) -> List[Dict[str, Any]]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    entries: List[Dict[str, Any]] = []
    with dataset_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            content = (data.get("content") or "").strip()
            law_id = data.get("id")
            if not content or law_id is None:
                continue
            entries.append(
                {
                    "id": law_id,
                    "content": content,
                    "law_name": data.get("law_name"),
                    "law_duration": data.get("law_duration"),
                }
            )
    if not entries:
        raise ValueError("Dataset is empty or missing required fields")
    return entries


def sample_entries(
    entries: List[Dict[str, Any]], count: int, seed: int
) -> List[Dict[str, Any]]:
    if count > len(entries):
        raise ValueError(
            f"Requested {count} samples but dataset only has {len(entries)} records"
        )
    rng = random.Random(seed)
    return rng.sample(entries, count)


def build_prompt(entry: Dict[str, Any], keyword_count: int, use_specific: bool) -> str:
    content = entry.get("content", "")
    escaped_content = content.replace("{", "{{").replace("}", "}}").strip()
    law_name = (entry.get("law_name") or "").strip()
    clause_label = (entry.get("law_duration") or "").strip()
    if law_name and clause_label:
        law_title = f"{law_name}{clause_label}"
    else:
        law_title = law_name or clause_label or "该法条"
    law_title = law_title.replace("{", "{{").replace("}", "}}")

    if use_specific:
        return PROMPT_TEMPLATE_SPECIFIC.format(
            law_content=escaped_content,
            keyword_count=keyword_count,
            law_title=law_title,
        )
    return PROMPT_TEMPLATE_GENERAL.format(
        law_content=escaped_content, keyword_count=keyword_count
    )


async def call_deepseek(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    model: str,
    retries: int = 3,
    timeout: aiohttp.ClientTimeout | None = None,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful Chinese legal assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
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
                return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = f"Attempt {attempt} failed: {exc}"
            await asyncio.sleep(2 * attempt)
    raise RuntimeError(last_error or "Unknown error when calling DeepSeek")


def extract_json(text: str) -> Dict[str, Any] | None:
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def sanitize_keywords(
    raw_keywords: Any,
    fallback_query: str,
    keyword_count: int,
) -> List[str]:
    keywords: List[str] = []
    if isinstance(raw_keywords, list):
        candidates = raw_keywords
    elif isinstance(raw_keywords, str):
        candidates = re.split(r"[，,;；\s]+", raw_keywords)
    else:
        candidates = []

    for item in candidates:
        token = str(item).strip()
        if token and token not in keywords:
            keywords.append(token)

    if len(keywords) < keyword_count and fallback_query:
        for token in fallback_query.split():
            cleaned = token.strip()
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)
            if len(keywords) >= keyword_count:
                break

    return keywords[:keyword_count]


def intersect_keyword_lists(keyword_lists: List[List[str]]) -> List[str]:
    if not keyword_lists:
        return []
    intersection = set(keyword_lists[0])
    for kw_list in keyword_lists[1:]:
        intersection &= set(kw_list)
    if not intersection:
        return []
    ordered: List[str] = []
    for kw_list in keyword_lists:
        for token in kw_list:
            if token in intersection and token not in ordered:
                ordered.append(token)
    return ordered


async def process_entries(args: argparse.Namespace) -> List[Dict[str, Any]]:
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DeepSeek API key not provided. Use --api_key or set DEEPSEEK_API_KEY."
        )

    entries = load_law_entries(args.dataset)
    oversample = max(args.limit, args.limit * max(2, args.generations))
    oversample = min(len(entries), oversample)
    selected = sample_entries(entries, oversample, args.seed)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    request_timeout = aiohttp.ClientTimeout(total=args.http_timeout)
    results: List[Dict[str, Any]] = []
    prompt_rng = random.Random(args.seed + 99991)

    async with aiohttp.ClientSession(timeout=request_timeout) as session:
        tasks: List[asyncio.Task[Dict[str, Any] | None]] = []

        async def runner(entry: Dict[str, Any]) -> Dict[str, Any] | None:
            keyword_runs: List[List[str]] = []
            has_law_title = bool(
                (entry.get("law_name") or "").strip()
                or (entry.get("law_duration") or "").strip()
            )
            specific_threshold = SPECIFIC_PROMPT_WEIGHT / (
                GENERAL_PROMPT_WEIGHT + SPECIFIC_PROMPT_WEIGHT
            )
            use_specific = has_law_title and (prompt_rng.random() < specific_threshold)

            content_prompt = build_prompt(entry, args.keyword_count, use_specific)
            for _ in range(args.generations):
                async with semaphore:
                    raw = await call_deepseek(
                        session,
                        api_key,
                        content_prompt,
                        args.model,
                        timeout=request_timeout,
                    )
                raw_keywords = (
                    raw.get("keywords") or raw.get("keyword_list") or raw.get("keyword")
                    if isinstance(raw, dict)
                    else []
                )
                query = raw.get("query") if isinstance(raw, dict) else ""
                if isinstance(query, str):
                    query = " ".join(query.split())
                else:
                    query = ""
                keywords = sanitize_keywords(raw_keywords, query, args.keyword_count)
                if not keywords and query:
                    keywords = sanitize_keywords(
                        query.split(), query, args.keyword_count
                    )
                if keywords:
                    keyword_runs.append(keywords)

            final_keywords = intersect_keyword_lists(keyword_runs)
            if len(final_keywords) <= args.min_keywords:
                return None
            final_query = " ".join(final_keywords)
            return {
                "id": entry["id"],
                "query": final_query,
                # "keywords": final_keywords,
            }

        for entry in selected:
            tasks.append(asyncio.create_task(runner(entry)))

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                continue
            results.append(result)
            print(f"[INFO] Accepted {len(results)}/{args.limit} entries")
            if len(results) >= args.limit:
                break

        if len(results) >= args.limit:
            for task in tasks:
                if not task.done():
                    task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

    return results


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(results)} records to {output_path}")


def main() -> None:
    args = parse_args()
    results = asyncio.run(process_entries(args))
    save_results(results, args.output)


if __name__ == "__main__":
    main()
