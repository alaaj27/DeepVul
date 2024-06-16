
# DeepVul

DeepVul is a model designed to predict gene essentiality and drug response using gene expression data. The model leverages a shared feature extractor to learn representations that can be fine-tuned for specific tasks such as gene essentiality prediction and drug response prediction.

## Installation

To set up the environment, use the provided `condaenv.yml` file with conda. First, ensure you have conda installed, then run the following command:

```bash
conda env create --file condaenv.yml
conda activate condaenv
```
### Datasets

To run the DeepVul model, you will need to download the following datasets and copy them into the `data` directory (with the names shown below):

1. **Gene Expression**: [OmicsExpressionProteinCodingGenesTPMLogp1](https://depmap.org/portal/data_page/?tab=allData&releasename=DepMap%20Public%2024Q2&filename=OmicsExpressionProteinCodingGenesTPMLogp1.csv)
2. **Gene Essentiality**: [CRISPRGeneEffect.csv](https://depmap.org/portal/data_page/?tab=allData&releasename=DepMap%20Public%2024Q2&filename=CRISPRGeneEffect.csv)
3. **Drug Response**: [primary-screen-replicate-collapsed-logfold-change.csv](https://depmap.org/portal/data_page/?tab=allData&releasename=PRISM%20Repurposing%2019Q4&filename=primary-screen-replicate-collapsed-logfold-change.csv)
4. **Sanger Essentiality Data**: [gene_effect.csv](https://depmap.org/portal/data_page/?tab=allData&releasename=Sanger%20CRISPR%20(Project%20Score%2C%20CERES)&filename=gene_effect.csv)
5. **Somatic Mutation Data**: [CCLE_Oncomap3_Assays_2012-04-09.csv](https://depmap.org/portal/data_page/?tab=allData&releasename=Oncomap%20mutations&filename=CCLE_Oncomap3_Assays_2012-04-09.csv)

After downloading these datasets, place them in the `data` directory to ensure the model can access them correctly.
## Hyperparameter Usage and Possible Values

When running the DeepVul model, you can specify various hyperparameters to control its behavior. Below is a list of the hyperparameters along with their possible values:

- `--pretrain_batch_size`: Batch size for pre-training data loading (default: 20)
- `--finetuning_batch_size`: Batch size for fine-tuning data loading (default: 20)
- `--hidden_state`: Hidden state size for the model (default: 1000)
- `--pre_train_epochs`: Number of epochs for pre-training (default: 20)
- `--fine_tune_epochs`: Number of epochs for fine-tuning (default: 20)
- `--opt`: Optimizer type (default: "Adam")
- `--lr`: Learning rate for the optimizer (default: 0.0005)
- `--dropout`: Dropout rate (default: 0.2)
- `--nhead`: Number of heads in the multihead attention models (default: 4)
- `--num_layers`: Number of layers in the model (default: 2)
- `--dim_feedforward`: Dimension of the feedforward network (default: 1024)
- `--fine_tuning_mode`: Mode for fine-tuning (default: "freeze-shared", options: ["freeze-shared", "initial-shared"])
- `--run_mode`: Run mode (options: "pre-train", "fine-tune", "both")

## Running the Model
First, change your current directory to src :

```bash
cd src
```

### Pre-training

To run the pre-training process, use the following command:

```bash
python run_deepvul.py --pretrain_batch_size 20 --hidden_state 1000 --pre_train_epochs 20 --opt "Adam" --lr 0.0005 --dropout 0.2 --nhead 4 --num_layers 2 --dim_feedforward 1024 --run_mode pre-train
```

### Fine-tuning

To run the fine-tuning process, use the following command:

```bash
python run_deepvul.py --finetuning_batch_size 20 --hidden_state 1000 --fine_tune_epochs 20 --opt "Adam" --lr 0.0005 --dropout 0.2 --nhead 4 --num_layers 2 --dim_feedforward 1024 --fine_tuning_mode "freeze-shared" --run_mode fine-tune
```

### Running Both Pre-training and Fine-tuning

To run both pre-training and fine-tuning sequentially, use the following command:

```bash
python run_deepvul.py --pretrain_batch_size 20 --finetuning_batch_size 20 --hidden_state 1000 --pre_train_epochs 20 --fine_tune_epochs 20 --opt "Adam" --lr 0.0005 --dropout 0.2 --nhead 4 --num_layers 2 --dim_feedforward 1024 --fine_tuning_mode "freeze-shared" --run_mode both
```

## Additional Information

For more details on the model and its implementation, please refer to the source code and associated documentation. If you encounter any issues or have questions, feel free to open an issue or contact the maintainers.
