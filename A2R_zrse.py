from copy import deepcopy
from typing import Any, Dict, List, Tuple
from pathlib import Path
import torch,os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Optional, Union, List, Tuple, Dict
import json,random
import numpy as np
import seaborn as sns
from collections import Counter

from util import nethook
from hparams import HyperParams
from trainer import kl_loss, masked_log_probs, compute_param_importance, apply_grad_mask, compute_contrastive_loss
from location import locate_repair_layer
from evaluate.eval_utils_zrse import compute_rewrite_quality_zsre,compute_safety_repair_quality
from dset import MENDQADataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

def apply_a2r_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        normal_requests: List[Dict],
        hparams: HyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    # deltas = execute_a2r(model, tok, requests, hparams)
    #
    # with torch.no_grad():
    #     for w_name, upd_matrix in deltas.items():
    #         w = nethook.get_parameter(model, w_name)
    #         if return_orig_weights and w_name not in weights_copy:
    #             weights_copy[w_name] = w.detach().clone()
    #
    #         w[...] += upd_matrix
    #
    # print(f"New weights successfully inserted into {list(deltas.keys())}")
    teacher_logits_map = {}
    for request in normal_requests:
        prompt_text = request["requested_rewrite"]["prompt"]
        with torch.no_grad():
            input_tokens = tok(prompt_text, return_tensors="pt", padding=True, truncation=True).to(hparams.device)
            teacher_logits = model(**input_tokens).logits
        teacher_logits_map[prompt_text] = teacher_logits
    for request in requests:
        print(f"Processing request: {request}")
        # Execute A2R for the single request
        deltas = execute_a2r(model, tok, [request], normal_requests,teacher_logits_map, hparams)  # Wrap request in a list

        # Apply the weight deltas
        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                w = nethook.get_parameter(model, w_name)

                # Save original weights if required
                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()

                # Update model weights
                w[...] += upd_matrix

        print(f"Updated weights for request: {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def get_repair_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -100)


def execute_a2r(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        normal_requests: List[Dict],
        teacher_logits_map,
        hparams: HyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes A2R to repair specific model while preserving overall performance.

    Steps:
    1. Process requests and format target text.
    2. Select and backup model weights for modification.
    3. Configure optimizer and enable gradients for target layers.
    4. Perform fine-tuning:
       - Compute target loss (NLL).
       - Apply knowledge distillation (KL divergence).
       - Use contrastive loss for enhanced learning.
       - Mask gradients to limit updates.
    5. Compute and return weight deltas while restoring original parameters.

    Args:
        model: Pretrained language model.
        tok: Tokenizer for text processing.
        requests: List of knowledge modification requests.
        normal_requests: Additional requests for distillation.
        teacher_logits_map: Teacher model outputs for knowledge distillation.
        hparams: Hyperparameters for optimization.

    Returns:
        Dict of weight deltas after refinement.
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["requested_rewrite"]["target_new"] != " ":
            request["requested_rewrite"]["target_new"] = " " + request["requested_rewrite"]["target_new"]

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers  # specific layer for each instance
        if hparams.rewrite_module_tmp.format(layer) in n
    }

    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    ############repair regions#############################


    # Prepare for knowledge distillation
    ft_input = [request["requested_rewrite"]["prompt"] + " " + request["requested_rewrite"]["target_new"] for request in requests]
    out_ids = dict(tok(request["requested_rewrite"]["target_new"], return_tensors="pt", padding=True).to(device))  # torch.Size([1, 69])
    out_labels = get_repair_labels(tok, out_ids["input_ids"])
    # # Update loop: intervene at layers simultaneously
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        inputs = tok(ft_input, return_tensors="pt", padding=True).to(device)
        opt.zero_grad()
        output = model(**inputs).logits  # torch.Size([1, 321, 32000])
        loss_dict = masked_log_probs(hparams, output, out_labels, shift=True)
        l_target = loss_dict["nll"]
        # Knowledge distillation loss
        distill_loss = 0
        for request in normal_requests:
            prompt_text = request["requested_rewrite"]["prompt"]
            input_tokens = tok(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)
            student_logits = model(**input_tokens).logits
            teacher_logits_batch = teacher_logits_map[prompt_text]
            distill_loss += kl_loss(teacher_logits_batch.unsqueeze(0), student_logits, mask=input_tokens["attention_mask"])


        # Computing contrastive loss
        contrastive_loss_value = 0
        for request in requests:
            prompt_text = request["requested_rewrite"]["prompt"]
            input_tokens = tok(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)
            student_output = model(**input_tokens).logits

            # Positive and negative output
            target_new_tokens = tok(request["requested_rewrite"]["prompt"] + " " + request["requested_rewrite"]["target_new"], return_tensors="pt", padding=True,
                                    truncation=True).to(device)
            target_new_logits = model(**target_new_tokens).logits

            ground_truth_tokens = tok(request["requested_rewrite"]["prompt"] + " " + request["requested_rewrite"]["ground_truth"], return_tensors="pt",
                                      padding=True, truncation=True).to(device)
            ground_truth_logits = model(**ground_truth_tokens).logits

            # Computing contrastive loss
            contrastive_loss_value += compute_contrastive_loss(student_output, target_new_logits, ground_truth_logits,
                                                               temperature=0.07)
        loss = hparams.kl_factor * l_target + 0.01*distill_loss+0.005*contrastive_loss_value #0.01 0.001
        print(f"Batch loss {loss.item()}, loss_target*0.01:{ l_target}, loss_distill:{0.01*distill_loss}, loss_contrastive*0.005:{0.005*contrastive_loss_value}")

        loss.backward()
        # the mask for each parameter is calculated and applied
        for name, param in model.named_parameters():
            if name in weights:  # update weights of target layer only
                # Get the gradient mask of the current parameter
                grad_mask = compute_param_importance(param.grad, mask_threshold=0.01)
                # Apply masking to update only important parameters
                param.grad = apply_grad_mask(param.grad, grad_mask)

        opt.step()

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def process_requests(ds, run_dir, repaired_model, tok, hparams, state):
    """
    Processes a dataset of requests, computes repair rate, and saves the results to JSON files.

    Args:
        ds (iterable): A collection of request dictionaries.
        run_dir (Path): The directory where results will be saved.
        repaired_model (Model): The repaired language model.
        tok (Tokenizer): The tokenizer for processing text.
        hparams (Namespace): Hyperparameters containing device and token limits.
    """
    for request in ds:
        case_id = request["case_id"]
        case_result_path = run_dir / f"case_{case_id}_{state}.json"

        ret1 = [request["requested_rewrite"]["prompt"]] + request["paraphrase_prompts"] + [
            item['prompt'] for item in request['neighborhood_prompts']
        ]
        ret2 = compute_safety_repair_quality(
            repaired_model, tok, request, hparams.device,
            max_tokens=hparams.max_length, max_output_tokens=hparams.max_output_length
        )[0]

        result = [ret1, ret2]

        with open(case_result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":

    tok = transformers.AutoTokenizer.from_pretrained("model/Llama-3-8b-instruct")
    tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained('model/Llama-3-8b-instruct').to("cuda")

    ds_class = MENDQADataset
    ds_eval_method = compute_rewrite_quality_zsre
    DATA_DIR = "/mnt/data"
    dset = ds_class(DATA_DIR, tok=tok)

    random.seed(500)
    random_indices = random.sample(range(len(dset)), 3)
    ds1 = [dset[i] for i in random_indices]
    time123=[]

    for seed_num in range(200,250):
        set_seed(seed_num)
        random_indices = random.sample(range(len(dset)), 100)
        ds = [dset[i] for i in random_indices]

        # for request in ds:
        #     # hparams.layers = self._locate_toxic_layer(self.model, self.tok, [request, ])
        #     print(request["case_id"])
        #     print(request["requested_rewrite"]["target_new"])
        #     print(request["requested_rewrite"]["ground_truth"])

        run_dir = f"/mnt/project_a2r/result/seed_{seed_num}"
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        hparams = HyperParams.from_hparams("hyperparams/Llama-3-8b-zrse.yaml")

        process_requests(ds, run_dir, model, tok, hparams,state="origin")
        start = time.time()
        # locate the most influential layer
        if len(hparams.layers) == 0:
            for request in ds:
                print(request["requested_rewrite"]["prompt"] + " " + request["requested_rewrite"]["target_new"])
                input_ids1 = tok(
                    request["requested_rewrite"]["prompt"] + " " + request["requested_rewrite"]["target_new"],
                    return_tensors="pt",
                    padding=True).to(hparams.device)
                input_ids1 = input_ids1["input_ids"]
                print(input_ids1)
                target_ids1 = tok(" " + request["requested_rewrite"]["target_new"], return_tensors="pt",
                                  padding=True).to(hparams.device)
                target_ids1 = target_ids1["input_ids"]
                print(target_ids1)
                layers = locate_repair_layer(model,type=hparams.model_name, input_ids=input_ids1, start_ix=0, end_ix=45,
                                             target_ids=target_ids1, delta=0.3)
                if layers != 0:
                    hparams.layers.append(layers)  # logits

        hparams.layers=[item for item, _ in Counter(hparams.layers).most_common(4)]
        # if len(hparams.layers)==0:
        #     continue

        repaired_model, weights_copy = apply_a2r_to_model(
            model,
            tok,
            ds,
            ds1,
            hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=True)
        exec_time = time.time() - start
        print("Execution took", exec_time)
        time123.append(exec_time)
        process_requests(ds, run_dir, repaired_model, tok, hparams,state="repair")
        print(time123)
        hparams.layers=[]
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams.device}")

