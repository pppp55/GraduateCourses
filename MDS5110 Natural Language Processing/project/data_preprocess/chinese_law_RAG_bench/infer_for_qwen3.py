
import os
import json
import torch
import logging
import argparse
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
import transformers
from transformers import set_seed
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel 
import re
import requests
from transformers import TextStreamer
import torch


#从本地模型输出各评测集的回答，支持retriever
os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def list_to_dict(data_list):
    classified_dict = {}
    for item in data_list:
        task_name = item["task_name"]
        if task_name not in classified_dict:
            classified_dict[task_name] = [item]
        else:
            classified_dict[task_name].append(item)
    return classified_dict







import torch.nn.functional as F

class StreamStopper:
    """
    流式生成检测 </search> 或 </answer>，停止生成。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = ""  # 累积本次生成的新文本
        self.done = False
        self.action = None  # "search" 或 "answer"
        self.output_text = ""

    
    def process_token(self, token_id):
        """
        接收新生成的 token_id，累积到 generated_ids，并 decode 新增部分到 buffer。
        同时检测 </search> 或 </answer> 停止。
        """
        # 保存 token_id 到生成列表
        if not hasattr(self, "generated_ids"):
            self.generated_ids = []

        self.generated_ids.append(token_id.item())

        # decode 当前所有 token
        text = self.tokenizer.decode(self.generated_ids, skip_special_tokens=False)

        # 取新生成的部分
        new_text = text[len(self.buffer):]  # 只取新增文本
        self.buffer += new_text

        # 检测 </search> 或 </answer>
        if "</search>" in self.buffer and not self.done:
            self.done = True
            self.action = "search"
            # 输出从生成开始到 </search> 的内容
            # self.output_text = self.buffer.split("</search>")[0] + "</search>"
            self.output_text = self.tokenizer.decode(self.generated_ids, skip_special_tokens=False).split("</search>")[0] + "</search>"

        elif "</answer>" in self.buffer and not self.done:
            self.done = True
            self.action = "answer"
            # self.output_text = self.buffer.split("</answer>")[0] + "</answer>"
            self.output_text = self.tokenizer.decode(self.generated_ids, skip_special_tokens=False).split("</answer>")[0] + "</answer>"

        return new_text  # 返回新增的可显示文本



@torch.no_grad()
def stream_until_search(model, tokenizer, input_ids, 
                        max_new_tokens=1500, 
                        temperature=0.3,
                        repetition_penalty = 1.2):
    """
    流式生成，检测 </search> 和 </answer>。
    返回:
        action: "search" 或 "answer"
        output_text: 本轮生成内容，decode为文本
    """
    stopper = StreamStopper(tokenizer)

    generated = input_ids
    past_key_values = None

    for _ in range(max_new_tokens):
        # 模型前向
        outputs = model(
            generated if past_key_values is None else generated[:, -1:],
            use_cache=True,
            past_key_values=past_key_values
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # 对每个 token 进行惩罚
        for token_id in set(generated[0].tolist()):  # 用 set 避免重复遍历
            logits[0, token_id] /= repetition_penalty  # 或者 logits[0, token_id] /= repetition_penalty


        # 采样下一个 token
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 处理 token
        stopper.process_token(next_token[0])
        
        # 追加 token
        generated = torch.cat((generated, next_token), dim=1)

        # 检查是否停止
        if stopper.done:
            break

    return stopper.action, stopper.output_text



class LLM_retriever:
    """
    实现一个“思考-检索-再思考-回答”的闭环生成系统。
    """
    def __init__(self, model_path, api_key=None, api_url=None, max_turn=5,retrieve_path="http://127.0.0.1:8006/retrieve"):
        self.model_path = model_path
        
        self.api_key = api_key
        self.api_url = api_url
        self.retrieve_path = retrieve_path
        self.max_turn=max_turn

        print("--------------加载模型路径为：---------------\n", model_path)

        
        # Initialize the tokenizer and model
    
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")


        # 特殊token设置
        if 'Qwen' in model_path:
            self.tokenizer.pad_token_id = 151643
            self.tokenizer.eos_token_id = 151643
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '</s>'

        # 停止条件定义
        # self.target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_eos = [151645, 151643]
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    def _extract_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def _search(self, query):
        """调用本地检索服务"""
        if not query or not query.strip():
            print("[WARNING] Empty query passed to search function.")
            return ""

        payload = {"queries": [query], "topk": 3, "return_scores": True}
        try:
            response = requests.post(
                self.retrieve_path,
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
                            
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        return _passages2string(results[0])

    def gen(self, query ,
            #  history = [], model_prompt=""
            ):
        """执行完整的思考-检索-再思考-回答流程"""
        

        question = query.strip()

        system_prompt = (
            "根据要求，回答问题。你必须遵守思考-检索-思考-回答的推理模式。\n"
            "思考：对问题进行推理，尝试解答。推理过程中，如果你发现涉及某些法律条文，则进入检索步骤。\n"
            "检索：请把需要检索的关键词放在 <search> 和 </search> 标签之间，调用搜索引擎。例如：<search> 民法典 盗窃罪 </search>。\n"
            "系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。根据返回的结果，继续下一步思考。\n"
            "再次思考：基于检索结果，继续对问题进行推理。如果没有帮助，则修改关键词重新检索,不要重复检索已经检索过的关键词；如果有把握得到最终答案，则进入回答。\n"
            "回答：注意！在 <answer> 和 </answer> 标签内提供最终答案。例如：<answer> 北京 </answer>\n"
            f"以下是需要回答的问题：{question}\n"
        )

        
        
        prompt = system_prompt
        cnt = 0
        print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
        print(f'**[prompt]:{prompt}')

        search_word_before=[]

        while True:#多轮检索
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            

            action, output_text = stream_until_search(
                                                    self.model,
                                                    self.tokenizer,
                                                    input_ids,
                                                    max_new_tokens=1500,
                                                    temperature=0.4,
                                                    repetition_penalty = 1.2
                                                    )

            instruct=''
            search_results=''

            print(f'[debug] output_text="{output_text}"')

            if action=="answer" or cnt > self.max_turn:
                response = output_text
                print(f'[debug]search turn:{cnt}')
                # print(f'[debug]final answer:{response}')
                break

            elif action=="search":
                tmp_query = self._extract_query(output_text)
                print(f'[debug] search query="{tmp_query}"')
                if search_word_before==tmp_query:#重复检索
                    instruct="如果我想给出最终回答，应该把答案放在 <answer> 和 </answer>之间。 \
                        如果需要继续搜索，应该把新的关键词放在<search> 和 </search>之间。重新思考。"
                
                elif tmp_query and( cnt < self.max_turn):
                    search_word_before=tmp_query
                    search_results = self._search(tmp_query)
                
                elif cnt==self.max_turn:
                    instruct = "跳过检索阶段。注意：接下来总结以上思考，必须给出最终回答！把最终答案放在 <answer> 和 </answer>之间。"

                else:
                    instruct="检索失败。重新检索或直接回答。"

            else:
                instruct="\n我先前的操作有问题。 \
如果我想搜索，应该把关键词放在<search> 和 </search>之间。 \
如果我想给出最终回答，应该把答案放在 <answer> 和 </answer>之间。让我重新思考。\n"

            if len(search_results):
                print(f'**[debug]searching result :\n"{search_results}"')
            if len(instruct):
                print(f'**[debug]instruct :"{instruct}"')

            search_text = self.search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            prompt += instruct

            cnt += 1

        # # 更新历史记录
        # history.append({"role": "user", "content": question})
        # history.append({"role": "assistant", "content": response})

        return response
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Model Args
    parser.add_argument('--question', default="Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?", type=str)
    parser.add_argument('--model_path', default="/caizhenyang/panghuaiwen/legal_LLM/RL_ckp/legal_exam-search-r1-ppo-qwen3-8b-em/global_step_100/actor", type=str)
    parser.add_argument('--retrieve_path', default="http://127.0.0.1:8006/retrieve", type=str)
    parser.add_argument('--retriever', default=False, type=bool)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_turn', default=3, type=int)
    
    args = parser.parse_args()

    set_seed(args.seed)
    model_path=args.model_path


    llm=LLM_retriever(model_path,retrieve_path=args.retrieve_path,max_turn=args.max_turn)
    
    

    res = llm.gen(args.question)
    
    

