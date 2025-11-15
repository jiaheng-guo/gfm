from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import random
import numpy as np
import time
import os

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###############################################################
# ----------------------------------------------------------- #
# TOKENIZATION UNIFYING
# ----------------------------------------------------------- #
###############################################################

def _infer_kmer_and_stride(tokenizer):
    kmer = getattr(tokenizer, "kmer", None)
    if kmer is None:
        kmer = getattr(tokenizer, "kmer_size", None)
    if kmer is None:
        kmer = getattr(tokenizer, "n_mer", None)
    if kmer is None:
        kmer = 1
    stride = getattr(tokenizer, "stride", None)
    if stride is None:
        stride = getattr(tokenizer, "kmer_stride", None)
    if stride is None or stride <= 0:
        stride = 1
    return int(kmer), int(stride)


def _build_offsets_fallback(sequence, tokenizer, token_length):
    """
    Approximate token offsets when tokenizer cannot return offsets.
    Assumes contiguous k-mer tokenization with stride (default 1).
    """
    kmer, stride = _infer_kmer_and_stride(tokenizer)
    seq_len = len(sequence)
    offsets = []

    prefix_specials = 1 if getattr(tokenizer, "bos_token_id", None) is not None else 0
    suffix_specials = 1 if getattr(tokenizer, "eos_token_id", None) is not None else 0

    for _ in range(prefix_specials):
        offsets.append((0, 0))

    pos = 0
    while len(offsets) < token_length - suffix_specials and pos < seq_len:
        end = min(seq_len, pos + kmer)
        offsets.append((pos, end))
        if pos == end:
            pos += 1
        else:
            pos += stride

    while len(offsets) < token_length - suffix_specials:
        offsets.append((seq_len, seq_len))

    for _ in range(suffix_specials):
        offsets.append((0, 0))

    if len(offsets) < token_length:
        offsets.extend([(0, 0)] * (token_length - len(offsets)))

    return offsets


def _tokenize_with_optional_offsets(sequence, tokenizer, device, need_offsets=False):
    encode_kwargs = dict(return_tensors="pt", truncation=True)
    request_offsets = need_offsets and getattr(tokenizer, "is_fast", False)
    if request_offsets:
        encode_kwargs["return_offsets_mapping"] = True

    try:
        encoded = tokenizer(sequence, **encode_kwargs)
    except NotImplementedError:
        if request_offsets:
            encode_kwargs.pop("return_offsets_mapping", None)
            encoded = tokenizer(sequence, **encode_kwargs)
            request_offsets = False
        else:
            raise

    offsets = None
    if request_offsets:
        offsets = encoded.pop("offset_mapping", None)
        if offsets is not None:
            offsets = offsets[0].tolist()

    inputs = {k: v.to(device) for k, v in encoded.items()}

    if need_offsets and offsets is None:
        token_length = inputs["input_ids"].size(1)
        offsets = _build_offsets_fallback(sequence, tokenizer, token_length)

    return inputs, offsets


def _map_bases_to_token_indices(offsets, base_positions, shifted=False):
    if offsets is None:
        raise ValueError("Token offsets unavailable for position-based computation.")

    base_positions = sorted(set(int(p) for p in base_positions if p is not None))
    if not base_positions:
        return []

    indices = []
    offset_slice = offsets[1:] if shifted else offsets
    for idx, (start, end) in enumerate(offset_slice):
        if end <= start:
            continue
        for pos in base_positions:
            if start <= pos < end:
                indices.append(idx)
                break
    return sorted(set(indices))


def sample_sequences_from_model(
    model,
    tokenizer,
    device,
    num_samples=32,
    max_new_tokens=512,
    top_k=40,
    temperature=1.0,
    seed=None,
):
    """
    Sample sequences from a causal LM using top-k sampling.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model.eval()

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None and getattr(tokenizer, "bos_token", None) is not None:
        bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    if bos_id is None:
        bos_id = getattr(tokenizer, "cls_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is None:
        raise ValueError("Tokenizer must provide a BOS/CLS/EOS token id for sampling.")

    eos_id = getattr(tokenizer, "eos_token_id", None)

    samples = []
    for _ in range(num_samples):
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        generated_ids = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                values, indices = torch.topk(next_token_logits, k, dim=-1)
                probs = torch.softmax(values, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                next_token = indices.gather(-1, next_idx)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if eos_id is not None and token_id == eos_id:
                break

            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if len(generated_ids) >= max_new_tokens:
                break

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        samples.append(text)

    return samples

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

    inputs, _ = _tokenize_with_optional_offsets(seq, tokenizer, device, need_offsets=False)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        # perplexity = torch.exp(loss)

    return loss.item(), None
    return loss.item(), perplexity.item()


def compute_conditional_suffix_loss(
    prefix_text: str,
    target_seq: str,
    model,
    tokenizer,
    device,
    position_indices=None,
):
    """
    Compute the mean loss of the target sequence tokens when
    the model is conditioned on a prefix (non-member context).
    """
    if not isinstance(prefix_text, str) or not isinstance(target_seq, str):
        raise ValueError("Prefix and target sequences must be strings.")

    combined = prefix_text + target_seq
    if not combined:
        raise ValueError("Combined sequence is empty.")

    model.eval()
    inputs, offsets = _tokenize_with_optional_offsets(
        combined,
        tokenizer,
        device,
        need_offsets=True,
    )
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    if logits.size(1) < 2:
        raise ValueError("Sequence too short after tokenization.")

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)

    if offsets is None:
        raise ValueError("Offsets required to isolate suffix tokens.")

    label_offsets = offsets[1:]
    prefix_len = len(prefix_text)
    base_mask_values = [
        (start is not None and start >= prefix_len)
        for start, _ in label_offsets
    ]
    if not any(base_mask_values):
        raise ValueError("No suffix tokens remain after applying prefix.")

    mask = torch.tensor(base_mask_values, dtype=torch.bool, device=per_token_loss.device)

    if position_indices is not None:
        token_indices = _map_bases_to_token_indices(label_offsets, position_indices, shifted=False)
        if not token_indices:
            raise ValueError("Provided SNP positions do not map to tokenizer offsets.")
        snp_mask = torch.zeros_like(mask)
        for idx in token_indices:
            if 0 <= idx < snp_mask.numel():
                snp_mask[idx] = True
        mask = mask & snp_mask

    suffix_losses = per_token_loss[0][mask]
    if suffix_losses.numel() == 0:
        raise ValueError("Suffix token mask produced no elements after SNP filtering.")

    return suffix_losses.mean().item()


def compute_dropout_loss(
    seq,
    model,
    tokenizer,
    device,
    num_masks=5,
):
    """
    Estimate the expected loss when randomly masking layers (via dropout)
    by enabling train mode and running multiple stochastic forward passes.
    """
    if num_masks <= 0:
        raise ValueError("num_masks must be positive.")

    was_training = model.training
    model.train()

    inputs, _ = _tokenize_with_optional_offsets(seq, tokenizer, device, need_offsets=False)
    losses = []

    with torch.no_grad():
        for _ in range(num_masks):
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    if not was_training:
        model.eval()

    return float(np.mean(losses)), float(np.var(losses))


def compute_snp_loss_and_perplexity(seq, snp_index, model, tokenizer, device):
    """
    Compute the mean loss and perplexity of SNP-affected tokens under a causal LM.
    """
    model.eval()

    inputs, offsets = _tokenize_with_optional_offsets(seq, tokenizer, device, need_offsets=True)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_all = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )

        seq_len = shift_labels.size(1)
        token_indices = _map_bases_to_token_indices(offsets, snp_index, shifted=True)
        if not token_indices:
            return 0.0, 0.0

        snp_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        snp_mask[token_indices] = True

        snp_loss = loss_all[snp_mask]

        snp_loss_mean = snp_loss.mean()
        snp_perplexity = torch.exp(snp_loss_mean)

    return snp_loss_mean.item(), snp_perplexity.item()


def compute_min_kpp_score(
    sequence,
    model,
    tokenizer,
    device,
    k_percent=0.2,
    eps=1e-8,
    positions=None,
):
    """
    Compute the Min-K%++ score (Equation 3 in the Min-K%++ paper) for a DNA sequence.

    Parameters
    ----------
    sequence : str
        Input DNA sequence.
    model : torch.nn.Module
        Target genomic foundation model (decoder-only).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the model.
    device : str
        Device identifier ("cpu" or "cuda").
    k_percent : float, optional
        Fraction (0, 1] of token positions with the minimum scores to average over.
    eps : float, optional
        Numerical stabilizer to avoid division by zero when the variance is tiny.

    positions : list[int], optional
        Token indices (0-based w.r.t. sequence) to be considered. If None, use all.

    Returns
    -------
    float
        Sentence-level Min-K%++ score.
    """
    if not 0 < k_percent <= 1:
        raise ValueError("k_percent must be within (0, 1].")

    model.eval()
    inputs, offsets = _tokenize_with_optional_offsets(
        sequence,
        tokenizer,
        device,
        need_offsets=positions is not None,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, seq_len, vocab_size)
    if logits.size(1) < 2:
        raise ValueError("Sequence too short to compute Min-K%++ score.")

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    probs = torch.exp(log_probs)

    mu = torch.sum(probs * log_probs, dim=-1)
    diff = log_probs - mu.unsqueeze(-1)
    var = torch.sum(probs * diff.pow(2), dim=-1)
    sigma = torch.sqrt(torch.clamp(var, min=eps))

    target_ids = inputs["input_ids"][:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_scores = (token_log_probs - mu) / sigma

    token_scores = token_scores.squeeze(0)
    if positions is not None:
        token_indices = _map_bases_to_token_indices(offsets, positions, shifted=True)
        if not token_indices:
            raise ValueError("No valid positions provided for Min-K%++ score.")
        index_tensor = torch.tensor(token_indices, device=token_scores.device, dtype=torch.long)
        token_scores = token_scores.index_select(0, index_tensor)

    num_tokens = token_scores.size(0)
    k = max(1, int(np.ceil(num_tokens * k_percent)))

    bottomk_values, _ = torch.topk(token_scores, k, largest=False)
    return bottomk_values.mean().item()


def find_diff_positions(seq_a: str, seq_b: str):
    """
    Return 0-based indices where two sequences differ.
    Length mismatch is handled by comparing up to the shorter length.
    """
    limit = min(len(seq_a), len(seq_b))
    return [idx for idx in range(limit) if seq_a[idx] != seq_b[idx]]

###############################################################
# ----------------------------------------------------------- #
# VISUALIZATION
# ----------------------------------------------------------- #
###############################################################

def store_results_to_txt(results, filepath):
    with open(filepath, 'w') as f:
        for item in results:
            f.write(f"{item}\n")


def visualize_loss_diff(A: list, B: list, C: list[list], mes, count):
    folder_path = f"./static/results/run_{count}"
    file_name = f"{mes}_{int(time.time())}.png"
    file_path = os.path.join(folder_path, file_name)
    os.makedirs(folder_path, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    x = range(len(A))

    for c in C:
        plt.plot(x, c, color='gray', alpha=0.4, linewidth=1)

    plt.plot(x, A, color='red', linewidth=2.5, label=f'Original Sequence  (mean:{np.mean(A):.4f}')
    plt.plot(x, B, color='blue', linewidth=2.5, label=f'Neighbor Sequence  (mean:{np.mean(B):.4f})')

    plt.xlabel("Sequence Index")
    plt.ylabel("Average Loss")
    plt.title(f"Model Loss on Original & Neighbor Sequences ({mes})")
    plt.legend()
    plt.savefig(file_path)
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
