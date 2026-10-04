# Data used by TADTalk

No data is stored in this repository. Put the raw files in `data/raw/` under the names below, then run
`bash scripts/run_real.sh prep`, `bash scripts/run_real.sh score` and `bash scripts/run_real.sh finish`.
Check and follow each provider's terms of use, and cite them.

| File in data/raw/ | What it is | Where it comes from |
| --- | --- | --- |
| `union_tads.txt` | Non-redundant union TAD set with the genes of each TAD, hg19 | Kumar lab (TADac). Not public here: ask the lab |
| `gencode.basic.gtf.gz` | Gene positions and types, GRCh37 | GENCODE release 38, GRCh37 mapping (`gencode.v38lift37.basic.annotation.gtf.gz`) |
| `Model.csv`, `CRISPRGeneEffect.csv`, `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | Cell-line information, CRISPR gene effect, expression | DepMap portal, Downloads |
| `ctrpv2_aac_wide.csv` | Drug response (AAC) for CTRPv2 | Made by `scripts/get_drugs.R` with the PharmacoGx R package |
| `tcga_all_tpm.txt.gz`, `TCGA_CancerType_Samples.txt` | TCGA tumour expression (TPM) for 24 cancers, and the cancer of each sample | GEO series GSE62944 |
| `TCGA-CDR.xlsx` | Curated overall survival (TCGA Clinical Data Resource, Liu et al. 2018, Cell) | GDC PanCanAtlas publication page (`TCGA-CDR-SupplementalTableS1.xlsx`) |

Tables built in `data/` (Parquet): `tad_genes`, `gene_tad`, `models`, `crispr`, `ccle_expr`, `drug_aac`,
`survival`, `sets`, `ccle_activity`, `tcga_activity`, `null_cox`.

For a quick test without real data: `python extras/make_sample_data.py data/raw` writes FAKE files in the
plain BED format. Never mix them with real files.
