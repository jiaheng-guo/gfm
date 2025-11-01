from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import random
import numpy as np
import time

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

##############################################################
# ---------------------------------------------------------- #
# SCORE COMPUTATION
# ---------------------------------------------------------- #
##############################################################

def compute_loss_and_perplexity(seq, model, tokenizer, device):
    """
    Returns
    -------
        loss.item(): Model's loss on input sequence
        perplexity.item(): Model's perplexity on input sequence
    """
    model.eval()

    inputs = tokenizer(seq, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        # perplexity = torch.exp(loss)

    return loss.item(), None
    return loss.item(), perplexity.item()


def compute_snp_loss_and_perplexity(seq, snp_index, model, tokenizer, device):
    """
    Compute the loss and perplexity of a variant sequence under the given model

    Parameters
    ----------
    seq : str
        The input DNA sequence (variant)
    snp_index : list
        List of SNP-affected token indices (0-based)

    Returns
    -------
        snp_loss_mean.item(): Average loss on SNP-affected tokens
        snp_perplexity.item(): Perplexity computed from SNP-affected tokens
    """

    model.eval()

    inputs = tokenizer(seq, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)
        loss_all = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction='none'
        )

        seq_len = input_ids.size(1)
        snp_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for idx in snp_index:
            if 0 <= idx < seq_len:
                snp_mask[idx] = True

        snp_loss = loss_all[snp_mask]

        if len(snp_loss) == 0:
            raise ValueError("No valid SNP indices found within sequence range.")

        snp_loss_mean = snp_loss.mean()
        snp_perplexity = torch.exp(snp_loss_mean)

    return snp_loss_mean.item(), snp_perplexity.item()

###############################################################
# ----------------------------------------------------------- #
# VISUALIZATION
# ----------------------------------------------------------- #
###############################################################

def store_results_to_txt(results, filepath):
    with open(filepath, 'w') as f:
        for item in results:
            f.write(f"{item}\n")


def visualize_loss_diff(A: list, B: list, C: list[list], mes: str):
    plt.figure(figsize=(12, 6))
    x = range(len(A))

    for c in C:
        plt.plot(x, c, color='gray', alpha=0.4, linewidth=1)

    plt.plot(x, A, color='red', linewidth=2.5, label='Original Sequence')
    plt.plot(x, B, color='blue', linewidth=2.5, label='Neighbor Sequence')

    plt.xlabel("Sequence Index")
    plt.ylabel("Average Loss")
    plt.title(f"Model Loss on Original & Neighbor Sequences ({mes})")
    plt.legend()
    plt.savefig(f"./static/lc_{mes}_{int(time.time())}.png")
    plt.show()


##############################################################
# ---------------------------------------------------------- #
# DEPRECATED FUNCTIONS (MIGHT BE REUSED LATER)
# ---------------------------------------------------------- #
##############################################################

def compute_filtered_perplexity(sequence, model, tokenizer, device, threshold=0.2):
    model.eval()
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["input_ids"]

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = torch.exp(log_probs)

        true_probs = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        mask = true_probs < threshold

        nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        if mask.sum() > 0:
            filtered_loss = (nll * mask).sum() / mask.sum()
            filtered_perplexity = torch.exp(filtered_loss)
        else:
            filtered_loss = nll.mean()
            filtered_perplexity = torch.exp(filtered_loss)

    return filtered_perplexity.item()


def generate_neighbors(sequence, mlm, tokenizer, device, k=0.1):
    """
    Randomly mask k% of tokens in sequence and replace with top-1 MLM predictions.

    [NOTE] This function is DEPRECATED due to unsatisfactory results in experiments.
    The reason is that the randomly masked tokens may not correspond to actual SNP locations,
    and may be just random, non-genetic regions with no functionality. Thus, we have converted
    to SNP-based neighbor generation in our main experiments.

    Args
    ----
        sequence: Input DNA sequence string
        mlm: Masked language model
        tokenizer: Tokenizer for the model
        device: CUDA/CPU device
        k: Fraction of tokens to mask (default 0.1 = 10%)

    Returns
    -------
        neighbor_seq: Modified sequence with replaced tokens
    """
    mlm.eval()
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].clone()

    seq_len = input_ids.size(1)
    num_mask = max(1, int(seq_len * k))

    mask_positions = random.sample(range(1, seq_len - 1), num_mask)

    mask_token_id = tokenizer.mask_token_id
    input_ids[0, mask_positions] = mask_token_id

    with torch.no_grad():
        outputs = mlm(input_ids)
        logits = outputs.logits

    vocab_size = tokenizer.vocab_size

    for pos in mask_positions:
        logits_at_pos = logits[0, pos, :vocab_size]
        pred_id = torch.argmax(logits_at_pos).item()

        if pred_id >= vocab_size:
            pred_id = mask_token_id

        input_ids[0, pos] = pred_id

    neighbor_seq = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return neighbor_seq


def compute_filtered_loglikelihood(sequence, model, tokenizer, device, threshold=0.2):
    """
    Compute the filtered log-likelihood of a sequence under the given model.

    [NOTE] Similar to generate_neighbors, this function is DEPRECATED due to unsatisfactory
    results in experiments.
    """
    model.eval()
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["input_ids"]

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = torch.exp(log_probs)

        true_probs = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        mask = true_probs < threshold

        nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        if mask.sum() > 0:
            filtered_nll = (nll * mask).sum() / mask.sum()
        else:
            filtered_nll = nll.mean()

    return filtered_nll.item()