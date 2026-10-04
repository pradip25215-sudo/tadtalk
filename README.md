# TADTalk

A chat assistant that answers cancer genomics questions at the level of 3D chromatin domains (TADs), and says
when a result is no better than chance.

## Why

Most genomics chatbots work gene by gene. Genes sit in topologically associating domains (TADs), and the
TADac work from the Kumar lab (IIIT Delhi) showed that the combined activity of the genes in a TAD tracks drug
response and survival. TADTalk lets anyone ask TAD-level questions in plain English and get computed answers.

## What you can ask

- Which TAD contains ERBB2, and which genes share it?
- Is the ERBB2 TAD a better lapatinib marker than ERBB2 alone in breast cell lines?
- Does the CCND1 TAD predict survival in HNSC?
- Which TADs are most linked to erlotinib response in lung cell lines?
- What is a TAD, and how can a broken TAD border activate an oncogene? Give references.

## How it works

1. **An LLM agent plans** (LangChain + LangGraph). It reads the question and chooses a tool.
2. **Seven Python tools compute.** Every number comes from precomputed tables (DuckDB on Parquet): TAD lookup,
   drug response (CTRPv2), survival (TCGA), gene essentiality (DepMap CRISPR), ranked lists, and PubMed search.
3. **TAD activity** is the GSVA score of the protein-coding genes in a TAD, in each sample (the TADac method).
4. **StatGuard checks every result.** The same statistic is computed for 200 random gene sets and 200 blocks of
   neighbouring genes of the same size. A result passes only if the TAD's own test is significant and the TAD
   beats 95% of both.
5. **General questions** are answered from the model's knowledge, labelled "General knowledge". Every PMID in an
   answer is checked against what the PubMed search really returned.

TADs in the union set are nested. For each gene, TADTalk uses the smallest TAD that still has at least
3 protein-coding genes; larger TADs can be asked about by id.

## The data (not in this repository; see DATA.md)

| What | Number |
| --- | --- |
| TADs used, of 18,304 in the union set (hg19) | 5,913 |
| Protein-coding genes with a TAD | 17,844 |
| Cell lines with drug response (CTRPv2) | 823, 544 drugs |
| Cell lines with CRISPR data (DepMap) | 1,208 |
| TCGA patients (HNSC, BRCA, LUAD) | 2,119 with survival |

## How well it works

| Test | TADTalk | Same LLM, no tools |
| --- | --- | --- |
| 29 data questions (answer key computed by code) | 29 right | 0 right |
| 35 multiple-choice questions on genomics and cancer biology | 35 right | not run |
| Says PASS when the key says FAIL | 0 times | |

Positive control: the breast cell lines with the highest ERBB2-TAD activity are JIMT1, EFM192A, AU565, HCC1419,
HCC2218, HCC1954, ZR7530 and HCC202, all known HER2-amplified lines.

Read these numbers with care: the question sets are small and were written by the author, scoring uses text
rules, and each set was run once. `bench/run_stress.py` adds a harder test with new wording, typos and bad inputs.

## Example results

- ERBB2 TAD and lapatinib in 33 breast lines: rho 0.575 for the TAD, 0.331 for ERBB2 alone, StatGuard PASS.
- CCND1 TAD and survival in 501 HNSC patients: hazard ratio 1.27 per SD (95% CI 1.11 to 1.45), StatGuard PASS.
- CCND1 TAD and paclitaxel in 573 lines: p = 2e-13, yet StatGuard FAIL, because 7% of random gene sets of the
  same size do as well.

These are associations in public data: hypotheses, not clinical claims.

## Run it

    conda create -n tadtalk python=3.11 -y && conda activate tadtalk
    pip install -r requirements.txt
    # put ANTHROPIC_API_KEY=... and MODEL=anthropic:claude-haiku-4-5-20251001 in a file named .env
    # get the data and build the tables: see DATA.md
    streamlit run ui/app.py

Tests: `python -m pytest -q`. Benchmarks: `python -m bench.make_key`, `python -m bench.run --system tadtalk`,
`python -m bench.score`, `python -m bench.run_general`, `python -m bench.run_stress`.

## Limitations

- One union TAD set (hg19) is applied to all cancers; TAD calls depend on the cell type, caller and resolution.
- Cell lines are not patients. Survival links are associations, with no adjustment for stage, age or treatment.
- About 450 TADs have no TCGA score, because the 2015 TCGA table uses some older gene names.
- Some groups are small (33 breast lines with lapatinib data), so estimates are uncertain.
- General-knowledge answers come from the LLM and can be wrong.

## Credits

Built by Pradip Palekar, Computational Genomics Lab (Dr. Vibhor Kumar), IIIT Delhi. TAD activity follows the
TADac preprint: "Deciphering drug response and phenotypic heterogeneity of cancer cells using gene ensembles of
regulatory units defined by chromatin domains" (bioRxiv 2023.01.15.524115).
