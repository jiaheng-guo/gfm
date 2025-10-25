from datasets import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

from preprocess import (
    load_gfm, 
    gfm_gallery,
    create_variant_sequences,
    get_grch38_sequence,
)

from utils import (
    fix_seed,
    compute_loss_and_perplexity, 
    compute_filtered_perplexity, 
    compute_snp_loss_and_perplexity,
    visualize_loss_diff,
)

# Configuration, temporarily defined here. Pay attention to SEQ_LENGTH
SEQ_START_IDX = 1000000
SEQ_END_IDX   = 1001000
SEQ_LENGTH    = 2000
SEQ_STRIDE    = 1000
CHROMOSOME_ID = ['1', '2', '3', '4', 'X']
REPLACE_PROB  = [0.6, 0.8, 1.0]

##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 1] Pure Loss-Based MIA
# ---------------------------------------------------------- #
##############################################################

def exp1(model, tokenizer, device):
    """
    Target sequences:

    - `seq`: a piece of raw human DNA sequence fetched from GRCh38, with various
         length from 100 to 10000

    - `seq_rev`: the reverse version of `sequence`

    We query BLAST and confirmed that `seq_rev` is not in GRCh38, then
    we calculated the perplexities of GFM HyenaDNA on them, and found that

    - `seq` has noticably lower perplexity (e.g., 2.32) than `seq_rev` (e.g., 3.85,
    which is 1.6X times larger).

    - This situation applies to all the 100 sequences we fetched.

    [NOTE] This is a toy example. A major flaw is that when we reverse the sequence to simulate a non-member, we actually change the underlying distribution of the sequence, which makes 
    the comparison not entirely fair, since the reverse sequence might not belong
    to the same distribution as the original human sequence. We use this experiment
    as a toy example to demonstrate that pure loss-based MIA can work in genomic
    foundation models, but the setting is not ideal.
    """
    ppx_seq, ppx_seq_rev = [], []
    for i in tqdm(range(100)):
        start_idx = 1000000 + 10000 * i # NOTE: make sure idx lie within valid boundary

        seq = get_grch38_sequence('X', start_idx, start_idx + 10**(3 + i % 2))
        seq_rev = seq[::-1] # reverse to simulate non-member, but change underlying distribution
        # blast_search_grch38(seq) # temporarily comment out cuz it costs too much time

        ppx_seq.append(compute_loss_and_perplexity(seq, model, tokenizer, device))
        ppx_seq_rev.append(compute_loss_and_perplexity(seq_rev, model, tokenizer, device))

    print("\n")
    print([f"{p:.2f}" for p in ppx_seq[:10]])
    print([f"{p:.2f}" for p in ppx_seq_rev[:10]])
    print(f"Average ratio: {np.mean(ppx_seq_rev) / np.mean(ppx_seq):.2f}")


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

        seq = get_grch38_sequence('X', start_idx, start_idx + 10**(3 + i % 2))
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
    [EXPERIMENT 3] Neighborhood comparison (Mattern et al. 2023)

    Generate neighbors using SNP annotations. For each sequence fetched from GRCh38,
    we create different kinds of neighboring sequences by randomly replacing a certain percentage
    of the nucleotides on SNP locations with their alternate alleles.

    Previously, we used MLM-based generation to create neighbors, but the results were not satisfactory. Therefore, we switched to SNP-based neighbor generation, which turns out to be much more effective.

    For evaluation, we currently focus on two metrics:
    1) Loss/Perplexity on SNP-affected tokens only
    2) Loss/Perplexity on all tokens
    """

    # loss lists, postfix 'o' represents original sequence, 'n' represents neighbor sequences
    # prefix 'snp' represents loss on SNP-affected tokens only, 'all' represents loss on all tokens
    # 'n_a' represents average over all neighbor sequences
    snp_o, snp_n_a, snp_n = [], [], [[] for _ in REPLACE_PROB]
    all_o, all_n_a, all_n = [], [], [[] for _ in REPLACE_PROB]

    for i in tqdm(range((SEQ_END_IDX - SEQ_START_IDX) // SEQ_STRIDE)):
        for cid in CHROMOSOME_ID:
            start_idx = SEQ_START_IDX + i * SEQ_STRIDE
            end_idx   = start_idx + SEQ_LENGTH

            seqs = create_variant_sequences(cid, start_idx, end_idx, REPLACE_PROB)
            if seqs is None:
                continue

            ref_seq = seqs['ref_seq']
            snp_pos = seqs['snp_pos']

            snp_o.append(compute_snp_loss_and_perplexity(ref_seq, snp_pos, model, tokenizer, device)[0])
            all_o.append(compute_loss_and_perplexity(ref_seq, model, tokenizer, device)[0])
            for p, s, a in zip(REPLACE_PROB, snp_n, all_n):
                var_seq = seqs['var_seq'][f'var_{p}']
                s.append(compute_snp_loss_and_perplexity(var_seq, snp_pos, model, tokenizer, device)[0])
                a.append(compute_loss_and_perplexity(var_seq, model, tokenizer, device)[0])

    snp_n_a = np.mean(np.array(snp_n), axis=0)
    all_n_a = np.mean(np.array(all_n), axis=0)
    
    visualize_loss_diff(snp_o, snp_n_a, snp_n, mes = "SNP_Only")
    visualize_loss_diff(all_o, all_n_a, all_n, mes = "All_Tokens")
    
    precision = []
    all_diff = all_n_a - all_o
    THRESHOLD_RANGE = np.arange(0, 0.2, 0.01)
    for t in THRESHOLD_RANGE:
        precision.append(np.mean(all_diff < t))
    df = pd.DataFrame({
        "Threshold": THRESHOLD_RANGE,
        "Precision": precision
    })
    print(df)

    return snp_o, snp_n_a, snp_n, all_o, all_n_a, all_n

##############################################################
# ---------------------------------------------------------- #
# [EXPERIMENT 4] ReCall (EMNLP 2024)
# ---------------------------------------------------------- #
##############################################################

def exp4(model, tokenizer, device):
    """
    [EXPERIMENT 4] ReCall (EMNLP 2024)
    To be implemented.
    """
    pass

##############################################################
# ---------------------------------------------------------- #
# EXPERIMENT EXECUTION
# ---------------------------------------------------------- #
##############################################################

def main():
    fix_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print("[STATUS] Using device:", _device_name)
    
    model_hyena, tk_hyena = load_gfm(gfm_gallery['hyenadna'])
    model_hyena = model_hyena.to(device)

    # exp1(model_hyena, tk_hyena, device) # Pure Loss-Based
    # exp2(model_hyena, tk_hyena, device) # Min-k% MIA
    exp3(model_hyena, tk_hyena, device) # Neighborhood Comparison
    # exp4(model_hyena, tk_hyena, device) # ReCall

if __name__ == "__main__":
    main()