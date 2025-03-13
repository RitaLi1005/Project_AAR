import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0],
    }


def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()


def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels


def multiclass_log_probs(config, pred, targ, shift=False, exact_match=False):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        pred = pred[:, -targ.size(1):]
        # targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

    # debug
    # print(pred.shape, targ.shape)
    # if pred.size(1) > targ.size(1):
    #     pred = pred[:, :targ.size(1)]

    if exact_match:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        if pred.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding


    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens

    nll = -log_prob
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
    }
def masked_log_probs(config, pred, targ, shift=False, exact_match=False):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(config, pred, targ, shift=shift, exact_match=exact_match)


def compute_param_importance(grad, mask_threshold=0.01):
    grad_norm = grad.abs()

    mask = grad_norm >= grad_norm.max() * mask_threshold
    return mask


def apply_grad_mask(grad, mask):
    grad = grad * mask
    return grad

def compute_contrastive_loss(student_output, target_new_logits, ground_truth_logits, temperature=0.07):
    # Normalize outputs
    student_output = F.normalize(student_output, p=2, dim=-1)
    target_new_logits = F.normalize(target_new_logits, p=2, dim=-1)
    ground_truth_logits = F.normalize(ground_truth_logits, p=2, dim=-1)

    # Flatten the sequence_length dimension for easier computation
    student_output_flat = student_output.view(-1, student_output.size(-1))  # [batch_size * sequence_length, vocab_size]
    target_new_logits_flat = target_new_logits.view(-1, target_new_logits.size(
        -1))  # [batch_size * sequence_length, vocab_size]
    ground_truth_logits_flat = ground_truth_logits.view(-1, ground_truth_logits.size(
        -1))  # [batch_size * sequence_length, vocab_size]

    positive_similarity = torch.matmul(student_output_flat,
                                       target_new_logits_flat.T)  # [batch_size * sequence_length, batch_size * sequence_length]
    negative_similarity = torch.matmul(student_output_flat,
                                       ground_truth_logits_flat.T)  # [batch_size * sequence_length, batch_size * sequence_length]

    logits = torch.cat([positive_similarity, negative_similarity],
                       dim=1)  # [batch_size * sequence_length, 2 * batch_size * sequence_length]

    labels = torch.zeros(logits.size(0), dtype=torch.long).to(student_output.device)

    loss = F.cross_entropy(logits / temperature, labels)
    return loss

def kl_loss(teacher_logits, student_logits, mask=None):

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)

    kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction='batchmean')

    if mask is not None:
        mask = mask.float()
        kl_loss = torch.sum(kl_loss * mask) / torch.sum(mask)

    return kl_loss