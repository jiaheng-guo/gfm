from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
import requests
import random
import pysam

###############################################################
# ----------------------------------------------------------- #
# [GFM] MODEL LOADER
# ----------------------------------------------------------- #
###############################################################

def load_gfm_to_device(model_name, device) -> tuple:
    """
    At the early stage, we only consider 3 GFMs with best performances:
    HyenaDNA, GENERator, and Nucleotide Transformer (NT). A detailed introduction
    can be found on the 9/28 Progress Report. Two things worth mentioning are that
    (1) HyenaDNA and GENERator are decoder-only, while NT is encoder-only;
    (2) The pre-training dataset of
          (i) HyenaDNA is GRCh38,
         (ii) GENERator is RefSeq, and
        (iii) Nucleotide Transformer is 1kG.

    Parameters:
    ----------
    model_name : str
        The name of the model to load (e.g., "hyenadna").
    device : str
        The device to load the model onto.

    Returns:
    -------
    A tuple containing the loaded model and tokenizer.
    """
    gfm_gallery = {
        'hyenadna':    ('LongSafari/hyenadna-medium-450k-seqlen-hf', 'NTP'),
        'generator':   ('GenerTeam/GENERator-eukaryote-1.2b-base', 'NTP'),
        'nucleotideT': ('InstaDeepAI/nucleotide-transformer-500m-human-ref', 'MLM'),
        'dnabert2':    ('zhihan1996/DNABERT-2-117M', 'MLM'),
    }
    if model_name not in gfm_gallery:
        raise ValueError(f"Model {model_name} not found in gallery.")
    checkpoint = gfm_gallery[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint[0],
        trust_remote_code=True
    )

    if checkpoint[1] == 'NTP':
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint[0],
            trust_remote_code=True
        )
    else: # assuming 'MLM'
        model = AutoModelForMaskedLM.from_pretrained(
            checkpoint[0],
            trust_remote_code=True
        )

    LM_module = AutoModelForCausalLM if checkpoint[1] == 'NTP' else AutoModelForMaskedLM
    model = LM_module.from_pretrained(
        checkpoint[0],
        trust_remote_code=True
    )

    return model.to(device), tokenizer


###############################################################
# ----------------------------------------------------------- #
# VCF FILES HANDLER
# ----------------------------------------------------------- #
###############################################################

def reconstruct_sequence(chrom, start, end, sample, vcf_path, ref_path):
    vcf = pysam.VariantFile(vcf_path)
    ref = pysam.FastaFile(ref_path)
    
    # convert 1-based to 0-based for pysam
    seq = list(ref.fetch(chrom, start-1, end))  

    for rec in vcf.fetch(chrom, start, end):
        if sample not in rec.samples:
            continue
        gt = rec.samples[sample]["GT"]
        if gt is None:
            continue
        # simple consensus: apply ALT if present in any haplotype
        if 1 in gt:
            pos = rec.pos - start  # VCF 1-based
            if 0 <= pos < len(seq):
                if len(rec.alts[0]) == 1: # do not consider SNP alleles with length != 1
                    seq[pos] = rec.alts[0]
    return "".join(seq)


###############################################################
# ----------------------------------------------------------- #
# [GENOMIC SEQUENCE] SEQ FETCHER & VERIFIER
# ----------------------------------------------------------- #
###############################################################

"""
In this section, we define the following two kinds of modules:

[Sequence fetchers]
-------------------
We use the Ensembl website, https://rest.ensembl.org/, to fetch genonic
sequences in mainstream genomic databases including
  - GRCh38 (for HyenaDNA),
  - T2T-CHM13v2.0 (not used in any GFM, fetched for MIA analysis purpose)

[Sequence verifiers]
--------------------
We use the BLAST module in the Biopython package for verifying if a given
sequence is in the GRCh38 dataset, or if there is any similar sequences
(with more than 80% similar subsequences). The Doc of the BLAST module and
biopython can be found at: https://biopython.org/docs/latest/Tutorial/.
"""

def fetch_grch38_region(chrom, start, end) -> str:
    """
    Human reference genome (GRCh38) fetcher using Ensembl REST API.

    [Note] HyenaDNA was trained on this dataset.

    Parameters:
    ----------
    chromosome : str
        Chromosome name (e.g., "1", "X").
    start : int
        1-based start coordinate of region.
    end : int
        1-based end coordinate of region (inclusive).

    Returns
    -------
    sequence : str
        raw nucleotide sequence.
    """

    server = "https://rest.ensembl.org/"

    ext = f"/sequence/region/human/{chrom}:{start}..{end}:1?"

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
    

def verify_grch38_membership(seq: str) -> bool:
    raise NotImplementedError("BLAST service too time-consuming")


def is_clean_dna(seq: str):
    """
    Verify if a sequence only contains A, T, C, G.
    """
    valid_nucleotides = set("ATGC")
    return set(seq.upper()) <= valid_nucleotides


###############################################################
# ----------------------------------------------------------- #
# [SNP] SNP INFO FETCHER & VARIANT SEQUENCE CREATOR
# ----------------------------------------------------------- #
###############################################################

"""
In this section, we define modules to deal with SNPs.
"""

def fetch_snps_in_region(chrom, start, end):
    """
    Fetch SNPs in a given genomic region from Ensembl REST API.
    """

    url = f"https://rest.ensembl.org/overlap/region/human/{chrom}:{start}-{end}?feature=variation"

    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)

    snps = response.json()

    return snps


def create_variant_sequences(chrom, start, end, probs, dataset='grch38', sample="HG00096"):
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
    probs : iterable
        List of fractions of SNPs to replace (e.g., [0.3, 0.4, 0.5])

    Returns
    -------
    variants : dict
        Dictionary mapping {variant_name: variant_sequence}
    """

    if dataset.lower() == 'grch38':
        ref_seq = fetch_grch38_region(chrom, start, end)
    elif dataset.lower() == 'vcf':
        ref_seq = reconstruct_sequence(
            chrom, 
            start, 
            end, 
            sample,
            vcf_path="./data/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz",
            ref_path="./data/Homo_sapiens.GRCh38.dna.chromosome.1.fa"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

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
            continue  # malformed record, though this rarely happens
        res["snp_pos"].append(pos)
    
        alts = alleles[1:]
        for p in probs:
            if random.random() <= p:
                res["var_seq"][f"var_{p}"][pos] = random.choice(alts)

    for k in res["var_seq"].keys():
        res["var_seq"][k] = ''.join(res["var_seq"][k])
    return res