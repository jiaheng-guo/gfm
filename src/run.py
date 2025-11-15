import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from eval import (
    roc_auc_score
)

from preprocess import (
    load_gfm_to_device, 
    create_variant_sequences,
    fetch_grch38_region,
    reconstruct_sequence,
    fetch_snps_in_region,
)

from utils import (
    fix_seed,
    compute_loss_and_perplexity, 
    compute_filtered_perplexity, 
    compute_snp_loss_and_perplexity,
    compute_min_kpp_score,
    find_diff_positions,
    compute_conditional_suffix_loss,
    visualize_loss_diff,
    _tokenize_with_optional_offsets,
)

# Configs, temporarily defined here. Pay attention to SEQ_LENGTH
SEQ_START_IDX = 10000000
SEQ_END_IDX   = 10250000
SEQ_LENGTH    = 10000
SEQ_STRIDE    = 5000
CHROMOSOME_ID = ['1']
REPLACE_PROB  = [0.1, 0.2, 0.3]
REPLACE_PROB_2 = [0.3, 0.4, 0.5]
VCF_SAMPLES = ["HG00096", "HG00097"]
VCF_PATH = "./data/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
REF_PATH = "./data/Homo_sapiens.GRCh38.dna.chromosome.1.fa"

##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 0] Illustration of model output logits
# ---------------------------------------------------------- #
##############################################################

def exp0(model, tokenizer, device):
    """
    In this experiment, we want to illustrate the distribution of genomic foundation model output logits, and what they look like after softmax, to give the reader an initial impression of this special domain and how it differs from natural language models.
    """
    model.eval()
    with torch.no_grad():
        # Generate a dummy input sequence
        input_seq = "ACGT" * 2500  # 10,000 tokens
        inputs, _ = _tokenize_with_optional_offsets(input_seq, tokenizer, device)

        outputs = model(**inputs)
        logits = outputs.logits

        # Compute softmax probabilities
        probs = logits.softmax(dim=-1)

    # Visualize the distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Logits Distribution")
    plt.hist(logits.cpu().numpy().flatten(), bins=100, alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.title("Softmax Probabilities Distribution")
    plt.hist(probs.cpu().numpy().flatten(), bins=100, alpha=0.7)
    plt.show()

##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 1] Pure Loss-Based MIA
# ---------------------------------------------------------- #
##############################################################

def exp1(model, tokenizer, device):
    """
    Target sequences:

    - ref_seq: human reference DNA sequence fetched from GRCh38, with length of 5,000,000
    - ind_seq: individual DNA sequence reconstructed from VCF, with length of 5,000,000

    we calculated the losses of GFM HyenaDNA on them, and found that

    - `seq` has noticably lower loss (e.g., 2.32) than `seq_rev` (e.g., 3.85,
    which is 1.6X times larger).

    - This situation applies to all the 100 sequences we fetched.

    [NOTE] This is a toy example. A major flaw is that when we reverse the sequence to simulate a non-member, we actually change the underlying distribution of the sequence, which makes 
    the comparison not entirely fair, since the reverse sequence might not belong
    to the same distribution as the original human sequence. We use this experiment
    as a toy example to demonstrate that pure loss-based MIA can work in genomic
    foundation models, but the setting is not ideal.
    """
    loss_ref, loss_ind = [], []
    for i in tqdm(range(100)):
        start_idx = 1000000 + 10000 * i # NOTE: make sure idx lie within valid boundary

        ref_seq = fetch_grch38_region('1', start_idx, start_idx + 10000)
        ind_seq = reconstruct_sequence(
            chrom='1',
            start=start_idx,
            end=start_idx + 10000,
            sample='HG00096',
            vcf_path=VCF_PATH,
            ref_path=REF_PATH
        )
        diff_indices = [i for i, (a, b) in enumerate(zip(ref_seq, ind_seq)) if a != b]
        if diff_indices == []:
            continue 
        loss_ref.append(compute_snp_loss_and_perplexity(ref_seq, diff_indices, model, tokenizer, device)[0])
        loss_ind.append(compute_snp_loss_and_perplexity(ind_seq, diff_indices, model, tokenizer, device)[0])

    print("\n")
    print(f"Average SNP-only losses on reference sequences: {np.mean(loss_ref):.2f}")
    print(f"Average SNP-only losses on individual sequences: {np.mean(loss_ind):.2f}")
    print(f"Average ratio: {np.mean(loss_ind) / np.mean(loss_ref):.2f}")
    print(f"Average difference: {np.mean(np.array(loss_ind) - np.array(loss_ref)):.2f}")


##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 2] Min-k% (ICLR 2024)
# ---------------------------------------------------------- #
##############################################################

def exp2(model, tokenizer, device):
    """
    Drop predictable tokens then compute perplexity.
    """

    mink_seq, mink_seq_rev = [], []
    for i in tqdm(range(100)):
        start_idx = 1000000 + 10000 * i # NOTE: make sure idx lie within boundary

        seq = fetch_grch38_region('X', start_idx, start_idx + 10**(3 + i % 2))
        seq_rev = seq[::-1]
        # blast_search_grch38(seq) # temporarily comment out cuz it costs too much time

        mink_seq.append(compute_filtered_perplexity(seq, model, tokenizer, device))
        mink_seq_rev.append(compute_filtered_perplexity(seq_rev, model, tokenizer, device))

    print("")
    print([f"{p:.2f}" for p in mink_seq[:10]])
    print([f"{p:.2f}" for p in mink_seq_rev[:10]])
    print(f"Average ratio: {np.mean(mink_seq_rev) / np.mean(mink_seq):.2f}")


##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 3] Neighborhood Comparison (ACL 2023)
# ---------------------------------------------------------- #
##############################################################

def exp3(model, tokenizer, device):
    """
    - Target model: HyenaDNA
    - Training data: Human reference genome (GRCh38)

    Generate neighbors using SNP annotations. For each sequence fetched from GRCh38,
    we create different kinds of neighboring sequences by randomly replacing a certain percentage
    of the nucleotides on SNP locations with their alternate alleles.

    Previously, we used MLM-based generation to create neighbors, but the results were not satisfactory. Therefore, we switched to SNP-based neighbor generation, which turns out to be much more effective.

    For evaluation, we currently focus on two metrics:
    1) Loss/Perplexity on SNP-affected tokens only
    2) Loss/Perplexity on all tokens
    """

    SEQ_NUM = (SEQ_END_IDX - SEQ_START_IDX) // SEQ_STRIDE

    def _process(tqdm_mes, ds, sn, rp, v_bool=True):
        # loss lists, postfix 'o' represents original sequence, 'n' represents neighbor sequences
        # prefix 'snp' represents loss on SNP-affected tokens only, 'all' represents loss on all
        # tokens, 'n_a' represents average over all neighbor sequences
        snp_o, snp_n_a, snp_n = [], [], [[] for _ in rp]
        all_o, all_n_a, all_n = [], [], [[] for _ in rp]

        for i in tqdm(range(SEQ_NUM), desc=tqdm_mes):
            for cid in CHROMOSOME_ID:
                start_idx = SEQ_START_IDX + i * SEQ_STRIDE
                end_idx   = start_idx + SEQ_LENGTH

                seqs = create_variant_sequences(cid, start_idx, end_idx, rp, dataset=ds, sample=sn)
                if seqs is None:
                    continue

                ref_seq = seqs['ref_seq']
                snp_pos = seqs['snp_pos']

                snp_o.append(compute_snp_loss_and_perplexity(ref_seq, snp_pos, model, tokenizer, device)[0])
                all_o.append(compute_loss_and_perplexity(ref_seq, model, tokenizer, device)[0])
                for p, s, a in zip(rp, snp_n, all_n):
                    var_seq = seqs['var_seq'][f'var_{p}']
                    s.append(compute_snp_loss_and_perplexity(var_seq, snp_pos, model, tokenizer, device)[0])
                    a.append(compute_loss_and_perplexity(var_seq, model, tokenizer, device)[0])
        
        snp_n_a = np.mean(np.array(snp_n), axis=0)
        all_n_a = np.mean(np.array(all_n), axis=0)

        if v_bool:
            with open(f"./static/results/count.txt", "r") as f:
                count = int(f.read().strip())
            visualize_loss_diff(snp_o, snp_n_a, snp_n, count=count, mes="SNP_Only")
            visualize_loss_diff(all_o, all_n_a, all_n, count=count, mes="All_Tokens")
            count += 1
            with open(f"./static/results/count.txt", "w") as f:
                f.write(str(count))

        return snp_o, snp_n_a, snp_n, all_o, all_n_a, all_n

    _, _, _, mall_o, mall_n_a, _ = _process("Processing members", 'grch38', None, REPLACE_PROB)
    _, _, _, nall_o1, nall_n_a1, _ = _process("Evaluating non-members", 'vcf', "HG00096", REPLACE_PROB)
    _, _, _, nall_o2, nall_n_a2, _ = _process("Evaluating non-members", 'vcf', "HG00097", REPLACE_PROB)

    # _, _, _, small_o, small_n_a, _ = _process("Processing members", 'grch38', None, REPLACE_PROB_2)
    # _, _, _, snpall_o1, snpall_n_a1, _ = _process("Evaluating non-members", 'vcf', "HG00096", REPLACE_PROB_2)
    # _, _, _, snpall_o2, snpall_n_a2, _ = _process("Evaluating non-members", 'vcf', "HG00097", REPLACE_PROB_2)

    # precision = []
    # all_diff = mall_n_a - mall_o
    # THRESHOLD_RANGE = np.arange(np.percentile(all_diff, 5), np.percentile(all_diff, 95), 0.01)
    # for t in THRESHOLD_RANGE:
    #     precision.append(np.mean(all_diff < t))
    # df = pd.DataFrame({
    #     "Threshold": THRESHOLD_RANGE,
    #     "Precision": precision
    # })
    # print(df)


##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 4] Min-K%++
# ---------------------------------------------------------- #
##############################################################

def exp4(model, tokenizer, device, sample_ids=None):
    """
    Min-K%++ MIA per Zhang et al. (ICLR 2025).

    Members: sequences fetched from GRCh38 (reference genome).
    Non-members: individual genomes reconstructed from 1kG VCF samples.
    """
    chrom = CHROMOSOME_ID[0]
    if chrom != '1':
        raise ValueError("VCF-based non-members currently only support chromosome '1'.")
    samples = sample_ids or VCF_SAMPLES

    mk_members = []
    mk_non_members = {sid: [] for sid in samples}

    for i in tqdm(range(100), desc="Min-K%++"):
        seq_len = 10 ** (3 + i % 2)
        start_idx = SEQ_START_IDX + 10000 * i
        end_idx = start_idx + seq_len

        seq = fetch_grch38_region(chrom, start_idx, end_idx)
        if seq is None:
            continue

        try:
            mk_members.append(compute_min_kpp_score(seq, model, tokenizer, device))
        except ValueError as err:
            print(f"[WARN] Skipping member seq {i}: {err}")
            continue

        for sid in samples:
            ind_seq = reconstruct_sequence(
                chrom,
                start_idx,
                end_idx,
                sid,
                vcf_path=VCF_PATH,
                ref_path=REF_PATH
            )
            if ind_seq is None:
                continue
            try:
                mk_non_members[sid].append(
                    compute_min_kpp_score(ind_seq, model, tokenizer, device)
                )
            except ValueError:
                continue

    non_member_scores = [s for scores in mk_non_members.values() for s in scores]
    if not mk_members or not non_member_scores:
        print("[ERROR] No valid sequences processed for Min-K%++.")
        return

    print("")
    print([f"{p:.4f}" for p in mk_members[:10]])
    print([f"{p:.4f}" for p in non_member_scores[:10]])
    print(f"Average member score: {np.mean(mk_members):.4f}")
    print(f"Average non-member score: {np.mean(non_member_scores):.4f}")
    print(f"Avg gap (member - non-member): {np.mean(mk_members) - np.mean(non_member_scores):.4f}")

    for sid, scores in mk_non_members.items():
        if scores:
            print(f"[{sid}] mean={np.mean(scores):.4f}, n={len(scores)}")


##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 5] Min-K%++ on SNP tokens (GRCh38 vs 1kG)
# ---------------------------------------------------------- #
##############################################################

EXP5_TARGET_REGIONS = 100
EXP5_MAX_ATTEMPTS = 2000

def exp5(
    model,
    tokenizer,
    device,
    sample_ids=None,
    k_percent=0.2,
    target_regions=None,
    threshold_count=7,
):
    """
    Focus Min-K%++ scores on annotated SNP positions only.
    Members are GRCh38 regions; non-members come from 1kG samples.
    """
    chrom = CHROMOSOME_ID[0]
    if chrom != '1':
        raise ValueError("VCF-based non-members currently only support chromosome '1'.")
    samples = sample_ids or VCF_SAMPLES

    total_regions = target_regions or EXP5_TARGET_REGIONS
    max_attempts = max(EXP5_MAX_ATTEMPTS, total_regions * 5)
    max_offset = max(0, SEQ_END_IDX - SEQ_START_IDX - SEQ_LENGTH)

    member_scores = {sid: [] for sid in samples}
    non_member_scores = {sid: [] for sid in samples}

    windows_used = 0
    attempts = 0
    pbar = tqdm(total=total_regions, desc="Min-K%++ SNP-only (diff loci)")
    while windows_used < total_regions and attempts < max_attempts:
        offset = (attempts * SEQ_STRIDE) % (max_offset + 1) if max_offset > 0 else 0
        start_idx = SEQ_START_IDX + offset
        end_idx = start_idx + SEQ_LENGTH
        attempts += 1

        if end_idx > SEQ_END_IDX or end_idx - start_idx < 2:
            continue

        ref_seq = fetch_grch38_region(chrom, start_idx, end_idx)
        if ref_seq is None:
            continue

        snps = fetch_snps_in_region(chrom, start_idx, end_idx)
        snp_pos = []
        for snp in snps:
            if snp.get("start") != snp.get("end"):
                continue
            pos = snp["start"] - start_idx
            if 0 <= pos < len(ref_seq) - 1:
                snp_pos.append(pos)
        if not snp_pos:
            continue
        snp_pos_set = set(snp_pos)

        window_has_scores = False
        for sid in samples:
            seq_ind = reconstruct_sequence(
                chrom,
                start_idx,
                end_idx,
                sid,
                vcf_path=VCF_PATH,
                ref_path=REF_PATH,
            )
            if seq_ind is None:
                continue
            try:
                diff_positions = [
                    pos for pos in find_diff_positions(ref_seq, seq_ind) if pos in snp_pos_set
                ]
                if not diff_positions:
                    continue
                member_scores[sid].append(
                    compute_min_kpp_score(
                        ref_seq,
                        model,
                        tokenizer,
                        device,
                        k_percent=k_percent,
                        positions=diff_positions,
                    )
                )
                non_member_scores[sid].append(
                    compute_min_kpp_score(
                        seq_ind,
                        model,
                        tokenizer,
                        device,
                        k_percent=k_percent,
                        positions=diff_positions,
                    )
                )
                window_has_scores = True
            except ValueError:
                continue

        if window_has_scores:
            windows_used += 1
            pbar.update(1)

    pbar.close()

    flat_members = [s for scores in member_scores.values() for s in scores]
    flat_non_members = [s for scores in non_member_scores.values() for s in scores]
    if not flat_members or not flat_non_members:
        print("[ERROR] Min-K%++ SNP-only produced no usable scores.")
        return

    print("\n[SNP-only]")
    print(f"Windows contributing scores: {windows_used} (attempted {attempts})")
    print(f"Members mean={np.mean(flat_members):.4f}, n={len(flat_members)}")
    print(f"Non-members mean={np.mean(flat_non_members):.4f}, n={len(flat_non_members)}")
    print(f"Gap (member - non-member): {np.mean(flat_members) - np.mean(flat_non_members):.4f}")
    for sid, scores in non_member_scores.items():
        if scores:
            print(
                f"[{sid}] member_mean={np.mean(member_scores[sid]):.4f}, "
                f"non_member_mean={np.mean(scores):.4f}, "
                f"n={len(scores)}"
            )

    labels = np.array([1] * len(flat_members) + [0] * len(flat_non_members))
    scores = np.array(flat_members + flat_non_members)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    print(f"AUC={auc:.4f}")

    if threshold_count and len(scores) > 0:
        unique_scores = np.unique(scores)
        if threshold_count > len(unique_scores):
            threshold_values = unique_scores
        else:
            threshold_values = np.linspace(scores.min(), scores.max(), num=threshold_count)

        print("Threshold-based metrics (positive = member, higher score => member)")
        for thr in threshold_values:
            preds = (scores >= thr).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            total = len(labels)
            acc = (tp + tn) / total if total else float("nan")
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            print(
                f"  thresh={thr:.4f} | acc={acc:.4f} | TPR={tpr:.4f} | FPR={fpr:.4f} "
                f"| TP={tp} FP={fp} TN={tn} FN={fn}"
            )



##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 6] Min-K%++ on SNP neighbors
# ---------------------------------------------------------- #
##############################################################

def exp6(model, tokenizer, device, probs=None, k_percent=0.2):
    """
    Compare GRCh38 reference sequences with SNP-generated neighbors
    using Min-K%++ restricted to SNP positions.
    """
    chrom = CHROMOSOME_ID[0]
    probs = probs or REPLACE_PROB

    seq_num = max(1, (SEQ_END_IDX - SEQ_START_IDX) // SEQ_STRIDE)
    ref_scores = []
    neighbor_scores = {p: [] for p in probs}

    for i in tqdm(range(seq_num), desc="Min-K%++ SNP neighbors"):
        start_idx = SEQ_START_IDX + i * SEQ_STRIDE
        end_idx = min(start_idx + SEQ_LENGTH, SEQ_END_IDX)
        if end_idx - start_idx < 2:
            continue

        seqs = create_variant_sequences(
            chrom,
            start_idx,
            end_idx,
            probs,
            dataset='grch38',
        )
        if seqs is None:
            continue

        ref_seq = seqs["ref_seq"]
        snp_pos = [pos for pos in seqs["snp_pos"] if 0 <= pos < len(ref_seq) - 1]
        if not snp_pos:
            continue

        try:
            ref_scores.append(
                compute_min_kpp_score(
                    ref_seq,
                    model,
                    tokenizer,
                    device,
                    k_percent=k_percent,
                    positions=snp_pos,
                )
            )
        except ValueError:
            continue

        for p in probs:
            var_key = f"var_{p}"
            var_seq = seqs["var_seq"].get(var_key)
            if var_seq is None:
                continue
            try:
                neighbor_scores[p].append(
                    compute_min_kpp_score(
                        var_seq,
                        model,
                        tokenizer,
                        device,
                        k_percent=k_percent,
                        positions=snp_pos,
                    )
                )
            except ValueError:
                continue

    if not ref_scores:
        print("[ERROR] Min-K%++ SNP neighbor experiment produced no reference scores.")
        return

    print("\n[SNP neighbors]")
    print(f"Reference mean={np.mean(ref_scores):.4f}, n={len(ref_scores)}")
    for p, scores in neighbor_scores.items():
        if scores:
            print(f"Neighbor p={p}: mean={np.mean(scores):.4f}, n={len(scores)}, gap={np.mean(scores) - np.mean(ref_scores):.4f}")
        else:
            print(f"Neighbor p={p}: no scores collected.")


##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 7] Prefix-Postfix Extraction
# ---------------------------------------------------------- #
##############################################################

from difflib import SequenceMatcher

EXP7_NUM_SAMPLES = 50
EXP7_PREFIX_LEN = 8000
EXP7_POSTFIX_LEN = 512
EXP7_SIMILARITY_THRESHOLD = 0.9

def exp7(
    model,
    tokenizer,
    device,
    num_samples=EXP7_NUM_SAMPLES,
    prefix_len=EXP7_PREFIX_LEN,
    postfix_len=EXP7_POSTFIX_LEN,
    similarity_threshold=EXP7_SIMILARITY_THRESHOLD,
    temperature=1.0,
    print_top=10,
):
    """
    Fetch GRCh38 sequences, split each into long prefixes and short postfixes,
    prompt the target model with the prefix, and evaluate how similar the
    generated postfix is to the ground-truth postfix. Deterministic (greedy)
    decoding is used to avoid randomness.
    """

    def _generate_postfix(prefix_text):
        model.eval()
        encode_kwargs = dict(return_tensors="pt", truncation=True)
        max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(max_length, int) and max_length > 0:
            encode_kwargs["max_length"] = max_length
        inputs = tokenizer(prefix_text, **encode_kwargs).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        eos_id = getattr(tokenizer, "eos_token_id", None)

        generated_ids = []
        for _ in range(postfix_len):
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            if eos_id is not None and token_id == eos_id:
                break
            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if attention_mask is not None:
                ones = torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=device,
                )
                attention_mask = torch.cat([attention_mask, ones], dim=1)

        if not generated_ids:
            return ""
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _summarize_sequence(seq, max_chars=120):
        return seq[:max_chars] + ("..." if len(seq) > max_chars else "")

    total_len = prefix_len + postfix_len
    samples = []
    success_count = 0
    attempts = 0
    max_attempts = num_samples * 5
    chrom = CHROMOSOME_ID[0]

    progress = tqdm(total=num_samples, desc="Prefix/Postfix Generation")
    while len(samples) < num_samples and attempts < max_attempts:
        start_idx = SEQ_START_IDX + attempts * SEQ_STRIDE
        end_idx = start_idx + total_len
        attempts += 1

        if end_idx > SEQ_END_IDX:
            break

        seq = fetch_grch38_region(chrom, start_idx, end_idx)
        if seq is None or len(seq) < total_len:
            continue

        prefix = seq[:prefix_len]
        true_postfix = seq[prefix_len:prefix_len + postfix_len]

        generated = _generate_postfix(prefix)
        generated = generated[: len(true_postfix)]

        if not generated or not true_postfix:
            similarity = 0.0
        else:
            similarity = SequenceMatcher(None, true_postfix, generated).ratio()

        exact_match = generated == true_postfix and len(generated) == len(true_postfix)
        success = similarity >= similarity_threshold
        if success:
            success_count += 1

        samples.append({
            "start": start_idx,
            "end": end_idx,
            "prefix_preview": _summarize_sequence(prefix),
            "true_postfix": true_postfix,
            "generated_postfix": generated,
            "similarity": similarity,
            "exact_match": exact_match,
        })

        progress.update(1)
        progress.set_postfix(success=success_count, sim=f"{similarity:.3f}")

    progress.close()

    if not samples:
        print("[exp7] No valid GRCh38 sequences processed.")
        return []

    samples.sort(key=lambda item: item["similarity"], reverse=True)

    print(
        f"[exp7] Processed {len(samples)} sequences "
        f"(attempted {attempts}, chrom {chrom})"
    )
    print(
        f"[exp7] Successes (similarity >= {similarity_threshold:.2f}): "
        f"{success_count}/{len(samples)}"
    )

    for idx, entry in enumerate(samples[:print_top]):
        print(
            f"[Top {idx+1}] start={entry['start']} sim={entry['similarity']:.3f} "
            f"exact={entry['exact_match']}"
        )
        print(f"True postfix : { _summarize_sequence(entry['true_postfix']) }")
        print(f"Generated    : { _summarize_sequence(entry['generated_postfix']) }")
        print("-" * 60)

    return samples


##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 8] RECALL-Style Conditional Likelihood
# ---------------------------------------------------------- #
##############################################################

EXP8_NUM_WINDOWS = 40
EXP8_PREFIX_MIN_LEN = 1000
EXP8_PREFIX_MAX_LEN = 10000
EXP8_TARGET_LEN = 2048
EXP8_SCORE_EPS = 1e-8

def exp8(
    model,
    tokenizer,
    device,
    num_windows=EXP8_NUM_WINDOWS,
    prefix_min_len=EXP8_PREFIX_MIN_LEN,
    prefix_max_len=EXP8_PREFIX_MAX_LEN,
    target_len=EXP8_TARGET_LEN,
    prefix_sample_ids=None,
    eval_sample_ids=None,
):
    """
    Implement a ReCall-style MIA by comparing conditional losses with and without a non-member prefix.
    """
    chrom = CHROMOSOME_ID[0]
    prefix_samples = prefix_sample_ids or VCF_SAMPLES
    eval_samples = eval_sample_ids or VCF_SAMPLES

    if not prefix_samples:
        raise ValueError("No samples provided for prefix construction.")
    if not eval_samples:
        raise ValueError("No evaluation samples provided for exp8.")

    def _build_non_member_prefix():
        segments = []
        attempts = 0
        stride = max(1, SEQ_STRIDE)
        min_segments = max(1, prefix_min_len // max(1, stride))
        max_segments = max(min_segments, prefix_max_len // max(1, stride))
        target_segments = max_segments
        max_attempts = target_segments * 10

        while len(segments) < target_segments and attempts < max_attempts:
            start_idx = SEQ_START_IDX + attempts * stride
            end_idx = start_idx + stride
            if end_idx > SEQ_END_IDX:
                break
            sid = prefix_samples[attempts % len(prefix_samples)]
            seq = reconstruct_sequence(
                chrom=chrom,
                start=start_idx,
                end=end_idx,
                sample=sid,
                vcf_path=VCF_PATH,
                ref_path=REF_PATH,
            )
            attempts += 1
            if seq is None or len(seq) < stride:
                continue
            segments.append(seq[:stride])

        prefix = "".join(segments)
        if len(prefix) < prefix_min_len:
            return ""
        return prefix[: prefix_max_len]

    prefix_text = _build_non_member_prefix()
    if not prefix_text:
        print("[exp8] Failed to build non-member prefix; aborting.")
        return
    print(
        f"[exp8] Prefix constructed (length={len(prefix_text)} bases, "
        f"min={prefix_min_len}, max={prefix_max_len}) from samples {prefix_samples}"
    )

    def _conditional_loss_with_adaptive_prefix(sequence, snp_positions=None):
        nonlocal prefix_text
        working_prefix = prefix_text
        trimmed = False
        while True:
            try:
                cond_loss = compute_conditional_suffix_loss(
                    working_prefix,
                    sequence,
                    model,
                    tokenizer,
                    device,
                    position_indices=snp_positions,
                )
                if trimmed and working_prefix != prefix_text:
                    print(f"Trimmed prefix length to {len(working_prefix)} to fit context.")
                prefix_text = working_prefix
                return cond_loss
            except ValueError as err:
                err_msg = str(err)
                needs_trim = (
                    "No suffix tokens remain" in err_msg
                    or "Suffix token mask produced no elements" in err_msg
                )
                if needs_trim and working_prefix:
                    trim_chars = max(1, len(working_prefix) // 2)
                    working_prefix = working_prefix[trim_chars:]
                    trimmed = True
                    continue
                raise

    member_scores = []
    non_member_scores = {sid: [] for sid in eval_samples}

    windows_used = 0
    attempts = 0
    max_attempts = num_windows * 5
    pbar = tqdm(total=num_windows, desc="RECALL SNP-only", unit="window")

    while windows_used < num_windows and attempts < max_attempts:
        start_idx = SEQ_START_IDX + attempts * SEQ_STRIDE
        end_idx = start_idx + target_len
        attempts += 1

        if end_idx > SEQ_END_IDX:
            break

        ref_seq = fetch_grch38_region(chrom, start_idx, end_idx)
        if ref_seq is None or len(ref_seq) < target_len:
            continue

        snps = fetch_snps_in_region(chrom, start_idx, end_idx)
        snp_positions = []
        for snp in snps:
            if snp.get("start") != snp.get("end"):
                continue
            pos = snp["start"] - start_idx
            if 0 <= pos < len(ref_seq):
                snp_positions.append(pos)
        if not snp_positions:
            continue

        try:
            loss_member, _ = compute_snp_loss_and_perplexity(ref_seq, snp_positions, model, tokenizer, device)
            cond_member = _conditional_loss_with_adaptive_prefix(
                ref_seq,
                snp_positions=snp_positions,
            )
        except ValueError as err:
            print(f"[exp8][WARN] Skipping member window ({start_idx}-{end_idx}): {err}")
            continue

        diff_member = abs(cond_member - loss_member)
        member_scores.append(diff_member)

        for sid in eval_samples:
            seq_ind = reconstruct_sequence(
                chrom=chrom,
                start=start_idx,
                end=end_idx,
                sample=sid,
                vcf_path=VCF_PATH,
                ref_path=REF_PATH,
            )
            if seq_ind is None or len(seq_ind) < target_len:
                continue
            try:
                loss_nm, _ = compute_snp_loss_and_perplexity(seq_ind, snp_positions, model, tokenizer, device)
                cond_nm = _conditional_loss_with_adaptive_prefix(
                    seq_ind,
                    snp_positions=snp_positions,
                )
            except ValueError:
                continue
            diff_nm = abs(cond_nm - loss_nm)
            non_member_scores[sid].append(diff_nm)

        windows_used += 1
        pbar.update(1)

    pbar.close()

    flat_non_members = [score for scores in non_member_scores.values() for score in scores]
    if not member_scores or not flat_non_members:
        print("Not enough sequences processed for RECALL scoring.")
        return

    mean_member = np.mean(member_scores)
    mean_non_member = np.mean(flat_non_members)
    print(
        f"Windows processed: {windows_used} "
        f"(attempted {attempts}, chrom {chrom})"
    )
    print(
        f"Member mean score={mean_member:.4f} | "
        f"Non-member mean score={mean_non_member:.4f} | "
        f"Gap={mean_member - mean_non_member:.4f}"
    )
    for sid, scores in non_member_scores.items():
        if scores:
            print(f"  [Non-member {sid}] mean={np.mean(scores):.4f}, n={len(scores)}")

    labels = np.array([1] * len(member_scores) + [0] * len(flat_non_members))
    scores = np.array(member_scores + flat_non_members)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    print(f"AUC={auc:.4f}")

    return {
        "prefix_len": len(prefix_text),
        "member_scores": member_scores,
        "non_member_scores": non_member_scores,
        "auc": auc,
    }

##############################################################
# ---------------------------------------------------------- #
# EXECUTION
# ---------------------------------------------------------- #
##############################################################

def main():
    fix_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE STATUS] Using {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    # model_hyena, tk_hyena = load_gfm_to_device('hyenadna', device)
    model_generator, tk_generator = load_gfm_to_device('generator', device)

    # exp1(model_hyena, tk_hyena, device) # Pure Loss-Based
    # exp2(model_hyena, tk_hyena, device) # Min-k% MIA
    # exp3(model_hyena, tk_hyena, device) # Neighborhood Comparison
    # exp3(model_generator, tk_generator, device) # Neighborhood Comparison
    # exp4(model_hyena, tk_hyena, device) # Min-K%++
    # exp5(model_hyena, tk_hyena, device, k_percent=0.2) # Min-K%++ @ SNP tokens
    # exp5(model_generator, tk_generator, device, k_percent=0.2) # Min-K%++ @ SNP tokens
    # exp6(model_hyena, tk_hyena, device) # SNP neighbors with Min-K%++
    # exp7(
    #     model_generator,
    #     tk_generator,
    #     device,
    #     num_samples=EXP7_NUM_SAMPLES,
    #     prefix_len=EXP7_PREFIX_LEN,
    #     postfix_len=EXP7_POSTFIX_LEN,
    #     similarity_threshold=EXP7_SIMILARITY_THRESHOLD,
    #     temperature=1.0,
    # ) # Prefix-Postfix Extraction
    exp8(
        model_generator,
        tk_generator,
        device,
        num_windows=EXP8_NUM_WINDOWS,
        prefix_min_len=EXP8_PREFIX_MIN_LEN,
        prefix_max_len=EXP8_PREFIX_MAX_LEN,
        target_len=EXP8_TARGET_LEN,
    )

if __name__ == "__main__":
    main()
