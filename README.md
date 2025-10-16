# 🔬 DeepVul: Multi-Task Transformer for Gene Essentiality and Drug Response

**DeepVul** is a multi-task transformer-based model designed to jointly predict **gene essentiality** and **drug response** using gene expression data. The model uses a **shared feature extractor** to learn robust biological representations that can be fine-tuned for downstream tasks, such as gene knockout effect prediction or treatment sensitivity profiling.

---

## 📑 Table of Contents

- [🚀 Features](#-features)  
- [📦 Installation](#-installation)  
- [📊 Datasets](#-datasets)  
- [⚙️ Hyperparameters](#️-hyperparameters)  
- [🏃 Running the Model](#-running-the-model)  
- [🧠 Additional Info](#-additional-information)  
- [📄 Citation](#-citation)

---

## 🚀 Features

- Joint prediction of gene essentiality and drug response  
- Shared transformer encoder for multi-task learning  
- Flexible modes: pre-training only, fine-tuning only, or both  
- Compatible with public omics and pharmacogenomic datasets  
- Fully configurable via command-line arguments  

---

## 📦 Installation

Make sure you have [conda](https://docs.conda.io/en/latest/) installed. Then run:

```bash
conda env create --file condaenv.yml
conda activate condaenv
```

---

## 📊 Datasets

To run DeepVul, download the following datasets and place them in the `data/` directory:

| Dataset | Description | Source |
|--------|-------------|--------|
| **Gene Expression** | TPM-log transformed gene expression data | [Download](https://depmap.org/portal/data_page/?tab=allData&releasename=DepMap%20Public%2024Q2&filename=OmicsExpressionProteinCodingGenesTPMLogp1.csv) |
| **Gene Essentiality** | CRISPR-Cas9 knockout effect scores | [Download](https://depmap.org/portal/data_page/?tab=allData&releasename=DepMap%20Public%2024Q2&filename=CRISPRGeneEffect.csv) |
| **Drug Response** | PRISM log-fold change drug response | [Download](https://depmap.org/portal/data_page/?tab=allData&releasename=PRISM%20Repurposing%2019Q4&filename=primary-screen-replicate-collapsed-logfold-change.csv) |
| **Sanger Essentiality** | CERES gene effect data from Sanger | [Download](https://depmap.org/portal/data_page/?tab=allData&releasename=Sanger%20CRISPR%20(Project%20Score%2C%20CERES)&filename=gene_effect.csv) |
| **Somatic Mutation** | Mutation profiles for CCLE lines | [Download](https://depmap.org/portal/data_page/?tab=allData&releasename=Oncomap%20mutations&filename=CCLE_Oncomap3_Assays_2012-04-09.csv) |

---

## ⚙️ Hyperparameters

DeepVul supports flexible training via CLI arguments:

| Parameter | Default | Description |
|----------|---------|-------------|
| `--pretrain_batch_size` | 20 | Batch size during pre-training |
| `--finetuning_batch_size` | 20 | Batch size during fine-tuning |
| `--hidden_state` | 500 | Size of transformer hidden layers |
| `--pre_train_epochs` | 20 | Pre-training epochs |
| `--fine_tune_epochs` | 20 | Fine-tuning epochs |
| `--opt` | Adam | Optimizer type |
| `--lr` | 0.0001 | Learning rate |
| `--dropout` | 0.1 | Dropout rate |
| `--nhead` | 2 | Number of attention heads |
| `--num_layers` | 2 | Transformer encoder layers |
| `--dim_feedforward` | 2048 | Feedforward network size |
| `--fine_tuning_mode` | freeze-shared | Whether to freeze shared layers during fine-tuning |
| `--run_mode` | pre-train / fine-tune / both | Execution mode |

---

## 🏃 Running the Model

### Change directory into the `src` folder:
```bash
cd src
```

### Pre-training
```bash
python run_deepvul.py --run_mode pre-train ...
```

### Fine-tuning
```bash
python run_deepvul.py --run_mode fine-tune ...
```

### Full Pipeline (Pre-train + Fine-tune)
```bash
python run_deepvul.py --run_mode both ...
```

Customize the CLI options as needed based on your experiment setup.

---

## 🧠 Additional Information

- Source code for model architecture, training, and evaluation is located in the `src/` directory.  
- If you encounter issues or have questions, please open a GitHub Issue or contact the maintainers.  
- Model interpretation and evaluation scripts are included in the repo.

---

## 📄 Citation

If you use DeepVul in your work, please cite:

```bibtex
@article {JararwehDeepVul,
	author = {Jararweh, Ala and Bach, My Nguyen and Arredondo, David and Macaulay, Oladimeji and Dicome, Mikaela and Tafoya, Luis and Hu, Yue and Virupakshappa, Kushal and Boland, Genevieve and Flaherty, Keith and Sahu, Avinash},
	title = {DeepVul: A Multi-Task Transformer Model for Joint Prediction of Gene Essentiality and Drug Response},
	elocation-id = {2024.10.17.618944},
	year = {2025},
	doi = {10.1101/2024.10.17.618944},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Despite their potential, current precision oncology approaches benefit only a small fraction of patients due to their limited focus on actionable genomic alterations. To expand its applicability, we propose DeepVul, a multi-task transformer-based model designed to predict gene essentiality and drug response from cancer transcriptome data. DeepVul aligns gene expressions, gene perturbations, and drug perturbations into a latent space, enabling simultaneous and accurate prediction of cancer cell vulnerabilities to numerous genes and drugs. Benchmarking against existing precision oncology approaches revealed that DeepVul not only matches but also complements oncogene-defined precision methods. Through interpretability analyses, DeepVul identifies underlying mechanisms of treatment response and resistance, as demonstrated with BRAF vulnerability prediction. By leveraging whole-genome transcriptome data, DeepVul enhances the clinical actionability of precision oncology, aiding in the identification of optimal treatments across a broader range of cancer patients. DeepVul is publicly available at https://github.com/alaaj27/DeepVul.git.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2025/10/15/2024.10.17.618944},
	eprint = {https://www.biorxiv.org/content/early/2025/10/15/2024.10.17.618944.full.pdf},
	journal = {bioRxiv}
}

```
