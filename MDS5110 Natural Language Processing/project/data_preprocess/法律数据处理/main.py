import re
import os
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path


def chinese_num_to_int(s: str) -> Optional[int]:
    """中文数字转阿拉伯数字（简化版）"""
    num_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
    }
    unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}

    # 如果已经是数字
    if s.isdigit():
        return int(s)

    # 简单处理：一、二、三...九
    if len(s) == 1 and s in num_map:
        return num_map[s]

    # 处理十、二十、三十...
    if s == '十':
        return 10

    result = 0
    temp_num = 0
    temp_unit = 1

    i = 0
    while i < len(s):
        char = s[i]

        if char in num_map:
            temp_num = num_map[char]
        elif char in unit_map:
            unit = unit_map[char]
            if temp_num == 0:
                temp_num = 1  # 处理"十五"这种情况

            if unit == 10000:  # 万
                result = (result + temp_num) * unit
                temp_num = 0
                temp_unit = 1
            elif unit >= temp_unit:  # 十、百、千
                result += temp_num * unit
                temp_num = 0
                temp_unit = unit
            else:
                result += temp_num * unit
                temp_num = 0

        i += 1

    result += temp_num
    return result if result > 0 else None


def int_to_chinese_num(n: int) -> str:
    """阿拉伯数字转中文数字（简化版）"""
    if n <= 0:
        return ''
    if n >= 10000:
        return ''  # 暂不支持万以上

    nums = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']

    # 1-9
    if n < 10:
        return nums[n]

    # 10-19: 十、十一...十九
    if n < 20:
        return '十' + (nums[n - 10] if n > 10 else '')

    # 20-99: 二十、二十一...九十九
    if n < 100:
        tens = n // 10
        ones = n % 10
        return nums[tens] + '十' + (nums[ones] if ones > 0 else '')

    # 100-999
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        result = nums[hundreds] + '百'

        if remainder == 0:
            return result
        elif remainder < 10:
            return result + '零' + nums[remainder]
        else:
            tens = remainder // 10
            ones = remainder % 10
            if tens == 1:
                result += '一十'
            else:
                result += nums[tens] + '十'
            if ones > 0:
                result += nums[ones]
            return result

    # 1000-9999
    thousands = n // 1000
    remainder = n % 1000
    result = nums[thousands] + '千'

    if remainder == 0:
        return result
    elif remainder < 10:
        return result + '零' + nums[remainder]
    elif remainder < 100:
        return result + '零' + int_to_chinese_num(remainder)
    else:
        return result + int_to_chinese_num(remainder)


def extract_law_data_from_file(file_path: str) -> List[Dict]:
    p = Path(file_path)
    if not p.exists() or not p.is_file() or p.suffix != '.txt':
        return []

    filename = p.name

    m_dur = re.search(r'【.+?】', filename)
    if not m_dur:
        return []
    law_duration = m_dur.group()

    m_name = re.search(r'】(.+?)\.txt$', filename)
    if not m_name:
        return []
    law_name = m_name.group(1)

    content = p.read_text(encoding='utf-8')

    last_part = None
    for part_iter in re.finditer(r'(?<!\S)第[一1]编', content):
        last_part = part_iter
    if last_part:
        content = content[last_part.start():]
    else:
        last_chapter = None
        for chap_iter in re.finditer(r'(?<!\S)第[一1]章', content):
            last_chapter = chap_iter
        if last_chapter:
            content = content[last_chapter.start():]

    part_pattern = re.compile(r'(?<!\S)第[一二三四五六七八九十零百千分\d]+编', re.UNICODE)
    chap_pattern = re.compile(r'(?<!\S)第[一二三四五六七八九十零百千\d]+章', re.UNICODE)
    section_pattern = re.compile(r'(?<!\S)第[一二三四五六七八九十零百千\d]+节', re.UNICODE)
    article_pattern = re.compile(r'(?<!\S)第([一二三四五六七八九十零百千\d]+)条', re.UNICODE)

    part_matches = list(part_pattern.finditer(content))
    results = []

    def extract_articles_in_block(block: str, prefix: str) -> List[Dict]:
        articles = []
        remaining = block

        while remaining:
            # 找到第一个条
            match = article_pattern.search(remaining)
            if not match:
                break

            art_start = match.start()
            art_heading = match.group(0)
            cur_num_str = match.group(1)
            cur_num = chinese_num_to_int(cur_num_str)

            # 计算下一条号
            next_num = cur_num + 1
            next_num_chinese = int_to_chinese_num(next_num)
            next_num_arabic = str(next_num)

            # 查找下一条（中文或阿拉伯数字）
            next_pattern = re.compile(
                rf'(?<!\S)第(?:{re.escape(next_num_chinese)}|{re.escape(next_num_arabic)})条',
                re.UNICODE
            )
            next_remaining = remaining[match.end():]
            next_match = next_pattern.search(next_remaining)

            if next_match:
                art_end = match.end() + next_match.start()
            else:
                art_end = len(remaining)

            ablock = remaining[art_start:art_end]
            art_text = ablock[len(art_heading):]
            multi_iters = list(re.finditer(rf'(?<!\S){art_heading}之[一二三四五六七八九十零百千\d]+', art_text))
            if multi_iters:
                abstract_text = art_text[:multi_iters[0].start()]
                articles.append({
                    'law_name': law_name,
                    'law_duration': law_duration,
                    'content': f'{law_name} {prefix}\n  {art_heading}\n{abstract_text}'
                })
                for i, m in enumerate(multi_iters):
                    start = m.start()
                    end = multi_iters[i + 1].start() if i + 1 < len(multi_iters) else len(art_text)
                    multi_heading = m.group(0)
                    multi_text = art_text[start + len(multi_heading):end]

                    combined = f'{law_name} {prefix}\n  {multi_heading}\n{multi_text}'

                    articles.append({
                        'law_name': law_name,
                        'law_duration': law_duration,
                        'content': combined
                    })
            else:
                combined = f'{law_name} {prefix}\n  {art_heading}\n{art_text}'

                articles.append({
                    'law_name': law_name,
                    'law_duration': law_duration,
                    'content': combined
                })

            remaining = remaining[art_end:]

        return articles

    def process_chapters_and_sections(block: str, prefix: str):
        """处理章和节"""
        chap_matches = list(chap_pattern.finditer(block))

        if chap_matches:
            starts = [m.start() for m in chap_matches] + [len(block)]
            for i, m in enumerate(chap_matches):
                start = m.start()
                end = starts[i + 1]
                chap_block = block[start:end]
                chap_heading = m.group(0)

                section_matches = list(section_pattern.finditer(chap_block))

                if section_matches:
                    sec_starts = [s.start() for s in section_matches] + [len(chap_block)]
                    for j, sec in enumerate(section_matches):
                        sec_start = sec.start()
                        sec_end = sec_starts[j + 1]
                        sec_block = chap_block[sec_start:sec_end]
                        sec_heading = sec.group(0)

                        sec_prefix = f'{prefix}  {chap_heading}  {sec_heading}' if prefix else f'{chap_heading}  {sec_heading}'
                        results.extend(extract_articles_in_block(sec_block, sec_prefix))
                else:
                    chap_prefix = f"{prefix}  {chap_heading}" if prefix else chap_heading
                    results.extend(extract_articles_in_block(chap_block, chap_prefix))
        else:
            results.extend(extract_articles_in_block(block, prefix))

    if part_matches:
        # 有编：按编分块
        starts = [m.start() for m in part_matches] + [len(content)]
        for i, m in enumerate(part_matches):
            start = m.start()
            end = starts[i + 1]
            part_block = content[start:end]
            part_heading = m.group(0)

            # 在每个编内处理章和节
            process_chapters_and_sections(part_block, part_heading)
    else:
        # 无编：直接处理章和节
        process_chapters_and_sections(content, "")

    return results


def extract_regulation_data_from_file(file_path: str):
    if not os.path.exists(file_path) or not os.path.isfile(file_path) or not file_path.endswith(".txt"):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()


def init_logging(log_file: str = "logs/app.log",
                 level: int = logging.DEBUG):

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

if __name__ == "__main__":
    init_logging()
    processing_dirs = ['法律条文（已处理）']
    out_file = "collected_data.jsonl"
    start_idx = 1
    with open(out_file, "wb") as f:
        for proc_dir in processing_dirs:
            is_law = (proc_dir == '法律条文（已处理）')
            sub_entries = os.listdir(proc_dir)
            for sub_entry in sub_entries:
                sub_entry_path = os.path.join(proc_dir, sub_entry)
                if not os.path.isdir(sub_entry_path) or not sub_entry.endswith('clean'):
                    continue
                files = os.listdir(sub_entry_path)
                for file in files:
                    file_path = os.path.join(sub_entry_path, file)
                    if is_law:
                        data = extract_law_data_from_file(file_path)
                    else:
                        data = extract_regulation_data_from_file(file_path)
                    if data:
                        logging.debug(f"Processing file: {file_path}, Extracted items: {len(data)}")
                        for item in data[:-1]:
                            item['id'] = start_idx
                            start_idx += 1
                            f.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
    logging.debug(f"Data collection completed. Total items: {start_idx}")