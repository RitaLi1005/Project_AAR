import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
import random
# from util.globals import *

# REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"

class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """
    def __init__(self, data_dir: str, tok: AutoTokenizer, *args, **kwargs):
        data_dir="/mnt/data"
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"
        # if not zsre_loc.exists():
        #     print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
        #     data_dir.mkdir(exist_ok=True, parents=True)
        #     torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        k=0
        for i, record in enumerate(raw):
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            if record["pred"]==record["answers"][0]:
                continue
            else:
                data.append(
                {
                    "case_id": k,
                    "requested_rewrite": {
                        "prompt": record["src"],
                        "subject": record["subject"],
                        "target_new": record["answers"][0],
                        "ground_truth": record["pred"],
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
                )
                k=k+1
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_random_prompts(data, k, num_samples=2):
    random.seed(k)
    selected_items = random.sample(data, num_samples)
    return [{"prompt": item["prompt"],"response": item["response"]} for item in selected_items]

class SEVALDataset:
    """
    https://github.com/IS2Lab/S-Eval
    """
    def __init__(self, data_dir: str, tok: AutoTokenizer, *args, **kwargs):
        # data_dir="/mnt/data"
        data_dir = Path(data_dir)
        seval_loc_en = data_dir / "s-eval_en_model_llama-3-8b-it.jsonl"

        # with open(seval_loc_en, "r") as f:
        #     raw = json.load(f)
        raw=load_jsonl(seval_loc_en)

        data = []
        k=0
        for i, record in enumerate(raw):
            # assert (
            #     "nq question: " in record["loc"]
            # ), f"Neighborhood prompt missing `nq question:`. Check for errors?"

            data.append(
                {
                    "case_id": k,
                    "traceid":record["traceid"],
                    "requested_rewrite": {
                        "prompt": record["prompt"],
                        "target_new": record["response"],
                        # "ground_truth": record["pred"],
                    },
                }
                )
            k = k + 1
        seval_loc_ori = data_dir / "new_s-eval_safetyc_ori_model_llama-3-8b-it.jsonl"

        raw=load_jsonl(seval_loc_ori)
        data1=[]
        for i, record in enumerate(raw):
            data1.append(
                {
                    "traceid": record["traceid"],
                    "ground_truth": record["response"],
                }
                )
        for update in data1:
            for record in data:
                if record["traceid"] == update["traceid"]:
                    record["requested_rewrite"]["ground_truth"] = update["ground_truth"]

        seval_loc_nei = data_dir / "new_s-eval_safetyc_nei_model_llama-3-8b-it.jsonl"
        raw=load_jsonl(seval_loc_nei)
        for record in data:
            record["neighborhood_prompts"]=get_random_prompts(raw,record["case_id"])

        seval_loc_re = data_dir / "s-eval_en_model_llama-3-8b-it_rephrase.jsonl"
        raw = load_jsonl(seval_loc_re)
        data2=[]
        for i, record in enumerate(raw):
            data2.append(
                {
                    "traceid": record["traceid"],
                    "paraphrase_prompts": [record["prompt"]],
                }
                )
        for update in data2:
            for record in data:
                if record["traceid"] == update["traceid"]:
                    record["paraphrase_prompts"] = update["paraphrase_prompts"]
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

class TEMDataset:
    """
    """
    def __init__(self, data_dir: str, tok: AutoTokenizer, *args, **kwargs):
        # data_dir="/mnt/data"
        data_dir = Path(data_dir)
        seval_loc_en = data_dir / "temporal_edit_origin.jsonl"

        raw=load_jsonl(seval_loc_en)

        data = []
        k=0
        for i, record in enumerate(raw):
            # assert (
            #     "nq question: " in record["loc"]
            # ), f"Neighborhood prompt missing `nq question:`. Check for errors?"

            target = record["entity_string"].split('/')[-1].replace("_", " ").lower()
            # print(target)
            # print(record["document"].lower().split(target, 1))
            data.append(
                {
                    "case_id": k,
                    "requested_rewrite": {
                        "prompt": record["prefix"],
                        "target_new": record["suffix"],
                        "ground_truth": record["origin"].split(record["prefix"], 1)[1],
                    },
                    "paraphrase_prompts": record["document"].lower().split(target, 1)[1],
                }
                )
            k = k + 1

        j=0
        seval_loc_nei = data_dir / "temporal_loc_origin.jsonl"
        raw=load_jsonl(seval_loc_nei)
        data2 = []
        for i, record in enumerate(raw):
            data2.append(
                {
                    "case_id": j,
                    "neighborhood_prompts": [{"prompt": record["prefix"],"response": record["origin"].split(record["prefix"], 1)[1]}],
                }
                )
            j=j+1
        for update in data2:
            for record in data:
                if record["case_id"] == update["case_id"]:
                    record["neighborhood_prompts"] = update["neighborhood_prompts"]

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)