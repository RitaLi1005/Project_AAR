from functools import partial
import torch.nn.functional as F
import torch
import numpy as np
import scipy.special
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet  # noqa
import json
from util.python_utils import make_print_if_verbose

from .hooks import make_lens_hooks
from .layer_names import make_layer_names
from scipy.stats import entropy


def collect_logits(model, input_ids, type,layer_names, decoder_layer_names):
    model._last_resid = None
    if type=="gpt2-xl" :
        with torch.no_grad():
            out = model(input_ids)
        del out
        model._last_resid = None
        layer_logits = np.concatenate(
            [model._layer_logits[name] for name in layer_names],
            axis=0,
        )
    elif type=="mistral-7B-v0.1" or "Llama-3-8b-instruct":
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # List of hidden states for all layers
        # Create a dictionary to store the layer logits
        layer_logits = []
        # Ensure that the layers specified are within range
        for name in layer_names[1:]:
            layer_idx = int(name.split(".")[-1])  # Assuming 'layer_0', 'layer_1', ... format
            if layer_idx < len(hidden_states):
                layer_logits.append(hidden_states[layer_idx])  # Get the hidden state for the specific layer
            else:
                print(f"Warning: Layer {name} does not exist in the model's output.")
        # Concatenate the selected layer logits
        layer_logits = np.concatenate([logits.cpu().numpy() for logits in layer_logits], axis=0)

    return layer_logits, layer_names


def attn_scores(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
        attention_scores = out.attentions

    # print(attention_scores)
    attention_scores = torch.stack(attention_scores)
    # print(attention_scores.shape)

    return attention_scores


def find_subsequence(input_ids, target_ids):
    for i in range(input_ids.size(1) - target_ids.size(1) + 1):
        if (input_ids[0, i:i + target_ids.size(1)] == target_ids).all():
            return i
    raise ValueError("Target tokens are not found as a contiguous subsequence in input_ids")


def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark


def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)


def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)


def kl_div(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)


def locate_repair_layer_(
        layer_logits,
        attention_scores,
        input_ids,
        target_ids,
        delta,
):
    ####################################
    try:

        start_idx = find_subsequence(input_ids, target_ids)
        end_idx = start_idx + target_ids.size(1)

    except ValueError as e:
        print(e)
        start_idx, end_idx = None, None

    attention_score_sum = []

    for layer in range(layer_logits.shape[0] - 1):
        # Retrieve the attention scores for the token subsequence of this layer
        attention_submatrix = attention_scores[layer, :, :, start_idx:end_idx, start_idx:end_idx]

        if layer == 0:
            attention_score_sum.append(0)
        if layer > 0:
            kl_div_init = F.kl_div(
                torch.log(attention_submatrix + 1e-9),
                attention_scores[0, :, :, start_idx:end_idx, start_idx:end_idx],
                reduction="batchmean"
            ).item()
            attention_score_sum.append(kl_div_init)

    logits_increment = []
    initial_logits = layer_logits[0, start_idx:end_idx, :]  # Use logits of layer 0 as baseline
    for layer in range(layer_logits.shape[0] - 1):
        layer_logit_diff = layer_logits[layer, start_idx:end_idx, :] - initial_logits
        # Aggregate as needed, e.g., calculate the L2 norm or simply sum
        logits_increment.append(torch.norm(torch.tensor(layer_logit_diff), p=2, dim=-1).mean().item())

    max_attention_layer = np.argmax(attention_score_sum) + 1  # +1 because layer numbering starts at 1
    max_logit_layer = np.argmax(logits_increment) + 1
    print(f"Layer with greatest influence on subsequence (Attention scores): Layer {max_attention_layer}")
    print(f"Layer with greatest influence on subsequence (Logits increment): Layer {max_logit_layer}")
    attention_score_sum = np.array(attention_score_sum)
    logits_increment = np.array(logits_increment)

    attention_scores_std = attention_score_sum.std()
    logits_increments_std = logits_increment.std()
    attention_scores_std = attention_scores_std if attention_scores_std > 0 else 1
    logits_increments_std = logits_increments_std if logits_increments_std > 0 else 1
    attention_scores_normalized = (attention_score_sum - attention_score_sum.mean()) / attention_scores_std
    logits_increments_normalized = (logits_increment - logits_increment.mean()) / logits_increments_std
    combined_scores = (1 - delta) * attention_scores_normalized + (
        delta) * logits_increments_normalized  # According to experimental records, the order is reversed

    max_combined_layer = np.argsort(combined_scores)[-1:][::-1][0]
    print(
        f"Layer with greatest influence on subsequence: Layer {max_combined_layer + 1}")
    return max_combined_layer

    ####################################

def locate_repair_layer(
        model,type,
        input_ids,
        target_ids,
        start_ix: int,
        end_ix: int,
        delta,
        block_step=1,
        include_input=True,
        force_include_output=True,
        include_subblocks=False,
        decoder_layer_names: list = ['final_layernorm', 'lm_head'],
        top_down=False,
        verbose=False,

):

    layer_names = make_layer_names(
        model,type,
        block_step=block_step,
        include_input=include_input,
        force_include_output=force_include_output,
        include_subblocks=include_subblocks,
        decoder_layer_names=decoder_layer_names
    )

    if type=="gpt2-xl" :
        make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names,
                    decoder_layer_names=decoder_layer_names,
                    verbose=verbose)

    layer_logits, layer_names = collect_logits(
        model, input_ids,type, layer_names=layer_names, decoder_layer_names=decoder_layer_names,
    )
    attention_scores = attn_scores(model, input_ids)
    # print(attention_scores.sum(dim=-1))


    max_combined_layer = locate_repair_layer_(
        layer_logits=layer_logits,
        attention_scores=attention_scores,
        input_ids=input_ids,
        target_ids=target_ids,
        delta=delta
    )

    return max_combined_layer
