# Membership Inference Attacks on Genomic Foundation Models

Benchmarking membership inference attacks on genomic foundation models. The following shows the organization of our benchmark.

![](./static/gfm_v1.png)

## Experiments

We conduct experiments to evaluate the privacy leakage of genomic foundation models (GFMs) under grey-box settings.

### Setup

Target Model: HyenaDNA

GPU: NVIDIA A100-SXM4-40GB  

### Numerical Results

#### Neighborhood Comparison

Given a sequence of confirmed training-set membership, we construct its neighbors by changing the nucleotides at SNP positions (we obtain these locations from dbSNP database and align them with the reference genome), of a certain replacement probability (we used 60%, 80%, and 100% in our experiments). We then compute the average loss of the original sequence and its neighbors under the target GFM, the results are shown below:

- number of original sequences: 500
- number of neighbor sequences per original sequence: 3 (60%, 80%, 100% SNP replacement)
- length of each sequence: 2000 base pairs
- grey curves: loss on individual neighbor; blue curve: average loss on neighbors; red curve: loss on original sequence.
- average ratio of SNP positions in each sequence: 41.33%

![](./static/loss_comparison_All_Tokens_huge.png)

![](./static/loss_comparison_SNP_Only_huge.png)

If we consider only model's loss on SNP positions, the difference between original sequences and their neighbors becomes very obscure, while if we consider loss on all tokens, the difference is much more obvious. By setting a proper threshold on the loss difference on all tokens, we can achieve a good membership inference performance. The following figure shows the precision score under different thresholds: