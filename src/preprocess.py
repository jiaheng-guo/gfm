from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import torch
import requests
import random
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

###############################################################
# ----------------------------------------------------------- #
# GFM LOADER
# ----------------------------------------------------------- #
###############################################################

"""
[GFM loader]

At the early stage, we only consider 3 GFMs with best performances:
HyenaDNA, GENERator, and Nucleotide Transformer (NT). A detailed introduction
can be found on the 9/28 Progress Report. Two things worth mentioning are that

(1) HyenaDNA and GENERator are decoder-only, while NT is encoder-only;

(2) The pre-training dataset of
      (i) HyenaDNA is GRCh38,
     (ii) GENERator is RefSeq, and
    (iii) Nucleotide Transformer is 1kG.

Later we will define functions that processes these datasets.
"""

def load_gfm(check_point: tuple) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        check_point[0],
        trust_remote_code=True
    )

    if check_point[1] == 'NTP':
        model = AutoModelForCausalLM.from_pretrained(
            check_point[0],
            trust_remote_code=True
        )
    else: # assuming 'MLM'
        model = AutoModelForMaskedLM.from_pretrained(
            check_point[0],
            trust_remote_code=True
        )

    return model, tokenizer

gfm_gallery = {
    'hyenadna':    ('LongSafari/hyenadna-medium-450k-seqlen-hf', 'NTP'),
    'generator':   ('GenerTeam/GENERator-eukaryote-1.2b-base', 'NTP'),
    'nucleotideT': ('InstaDeepAI/nucleotide-transformer-500m-human-ref', 'MLM'),
    # 'dnabert2':    ('zhihan1996/DNABERT-2-117M', 'MLM'), # Temporarily commented out
}

###############################################################
# ----------------------------------------------------------- #
# REF SEQ FETCHER & VERIFIER
# ----------------------------------------------------------- #
###############################################################

"""
[GRCh38 Sequence Fetcher] and [GRCh38 Sequence Verifier]

[GRCh38 sequence fetcher]
We use the Ensembl website, https://rest.ensembl.org/, to fetch genonic
sequences in mainstream genomic databases (used by the GFMs) including
  - GRCh38 (for HyenaDNA),
  - RefSeq (for GENERator), and
  - 1kG (for nucleotide transformer).

[GRCh38 sequence verifier]
We use the BLAST module in the Biopython package for verifying if a given
sequence is in the GRCh38 dataset, or if there is any similar sequences
(with more than 80% similar subsequences). The Doc of the BLAST module and
biopython can be found at: https://biopython.org/docs/latest/Tutorial/.
"""

def get_grch38_sequence(chromosome, start, end) -> str:
    """
    Ensembl human reference gene (GRCh38) fetcher.
    HyenaDNA was trained on this dataset.

    Output: <str> raw nucleotide sequence.
    """

    server = "https://rest.ensembl.org/"

    ext = f"/sequence/region/human/{chromosome}:{start}..{end}:1?"

    ver = "coord_system_version=GRCh38"

    headers = {"Content-Type": "text/plain"}

    try:
        r = requests.get(server + ext + ver, headers=headers)
        if r.ok:
            return r.text.strip()
        else:
            print(f"Error: {r.status_code}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def verify_grch38_membership(sequence) -> bool:
    raise NotImplementedError("BLAST service too time-consuming")

def is_clean_dna(seq):
    """
    Verify if a sequence only contains A, T, C, G.
    """
    valid_nucleotides = set("ATGC")
    return set(seq.upper()) <= valid_nucleotides


###############################################################
# ----------------------------------------------------------- #
# SNP INFO FETCHER & VARIANT SEQUENCE CREATOR
# ----------------------------------------------------------- #
###############################################################

def fetch_snps_in_region(chrom, start, end):
    """
    Fetch SNPs in a given genomic region from Ensembl REST API.
    """

    url = f"https://rest.ensembl.org/overlap/region/human/{chrom}:{start}-{end}?feature=variation"

    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)

    snps = response.json()

    return snps


def create_variant_sequences(chrom, start, end, probs):
    """
    Given a reference DNA sequence and SNP annotations (from fetch_snps_in_region),
    manually create variant sequences by substituting alternate alleles.

    Parameters
    ----------
    ref_seq : str
        Reference DNA sequence in GRCh38
    ref_start : int
        The genomic coordinate of the first base of ref_seq (1-based)
    snps : list
        List of SNP records returned by fetch_snps_in_region()
    precents : iterable
        List of fractions of SNPs to replace (e.g., [0.3, 0.4, 0.5])

    Returns
    -------
    variants : dict
        Dictionary mapping {variant_name: variant_sequence}
    """

    ref_seq = get_grch38_sequence(chrom, start, end)
    if ref_seq is None or not is_clean_dna(ref_seq):
        return None

    snps = fetch_snps_in_region(chrom, start, end)
    res = {
        "ref_seq": ref_seq,
        "snp_pos": [],
        "var_seq": {
            f"var_{p}": list(ref_seq) for p in probs
        }
    }
    for snp in snps:
        if snp["start"] != snp["end"]:
            continue  # skip non-single-nucleotide SNPs, though this rarely happens

        pos = snp["start"] - start

        alleles = snp.get("alleles", [])
        if len(alleles) == 0 or len(alleles) == 1:
            continue  # malformed record
        res["snp_pos"].append(pos)

        alts = alleles[1:]
        for p in probs:
            if random.random() <= p:
                res["var_seq"][f"var_{p}"][pos] = random.choice(alts)

    for k in res["var_seq"].keys():
        res["var_seq"][k] = ''.join(res["var_seq"][k])
    return res