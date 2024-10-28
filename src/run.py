import argparse
from utils import load_drug_data, load_essentiality_data  
from DeepVul import DeepVul


def main(pretrain_batch_size, finetuning_batch_size, hidden_state, pre_train_epochs, fine_tune_epochs, opt, lr, dropout, nhead, num_layers, dim_feedforward, fine_tuning_mode, run_mode):
    if run_mode == "pre-train":
        (
            train_loader, val_loader, test_loader,
            n_input_ess, n_input_exp, ess_gene_list, exp_gene_list
        ) = load_essentiality_data(batch_size=pretrain_batch_size, val_split=False)

        deep_vul = DeepVul(n_input_exp=n_input_exp,
                           hidden_state=hidden_state,
                           n_input_ess=n_input_ess,
                           n_input_drug=1,
                           ess_gene_list= ess_gene_list,
                           exp_gene_list=exp_gene_list,
                           nhead=nhead,
                           num_layers=num_layers,
                           dim_feedforward=dim_feedforward,
                           opt=opt, lr=lr,
                           dropout=dropout)

        deep_vul.pre_train(train_loader, val_loader, test_loader, epochs=pre_train_epochs, l1=False)

    elif run_mode == "fine-tune":
        (
            train_loader, val_loader, test_loader,
            n_input_drug, n_input_exp, ess_gene_list, exp_gene_list
        ) = load_drug_data(batch_size=finetuning_batch_size, val_split=False)

        deep_vul = DeepVul(n_input_exp=n_input_exp,
                           hidden_state=hidden_state,
                           n_input_ess=1,
                           n_input_drug=n_input_drug,
                           ess_gene_list= ess_gene_list,
                           exp_gene_list=exp_gene_list,
                           nhead=nhead,
                           num_layers=num_layers,
                           dim_feedforward=dim_feedforward,
                           opt=opt, lr=lr,
                           dropout=dropout)

        deep_vul.fine_tune(train_loader, val_loader, test_loader, epochs=fine_tune_epochs, mode=fine_tuning_mode)

    elif run_mode == "both":
        (
            train_loader_ess, val_loader_ess, test_loader_ess,
            n_input_ess, n_input_exp
        ) = load_essentiality_data(batch_size=pretrain_batch_size, val_split=False)

        (
            train_loader_drug, val_loader_drug, test_loader_drug,
            n_input_drug, n_input_exp, ess_gene_list, exp_gene_list
        ) = load_drug_data(batch_size=finetuning_batch_size, val_split=False)

        deep_vul = DeepVul(n_input_exp=n_input_exp,
                           hidden_state=hidden_state,
                           n_input_ess=n_input_ess,
                           n_input_drug=n_input_drug,
                           ess_gene_list= ess_gene_list,
                           exp_gene_list=exp_gene_list,
                           nhead=nhead,
                           num_layers=num_layers,
                           dim_feedforward=dim_feedforward,
                           opt=opt, lr=lr,
                           dropout=dropout)

        deep_vul.pre_train(train_loader_ess, val_loader_ess, test_loader_ess, epochs=pre_train_epochs, l1=False)
        deep_vul.fine_tune(train_loader_drug, val_loader_drug, test_loader_drug, epochs=fine_tune_epochs, mode=fine_tuning_mode)

    else:
        raise ValueError("Invalid run_mode. Choose from 'pre-train', 'fine-tune', or 'both'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepVul model")
    parser.add_argument('--pretrain_batch_size', type=int, default=20, help='Batch size for pre-training data loading')
    parser.add_argument('--finetuning_batch_size', type=int, default=20, help='Batch size for fine-tuning data loading')
    parser.add_argument('--hidden_state', type=int, default=1000, help='Hidden state size for the model')
    parser.add_argument('--pre_train_epochs', type=int, default=20, help='Number of epochs for pre-training')
    parser.add_argument('--fine_tune_epochs', type=int, default=20, help='Number of epochs for fine-tuning')
    parser.add_argument('--opt', type=str, default="Adam", help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads in the multihead attention models')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of the feedforward network')
    parser.add_argument('--fine_tuning_mode', type=str, default="freeze-shared", help='Mode for fine-tuning')
    parser.add_argument('--run_mode', type=str, choices=['pre-train', 'fine-tune', 'both'], required=True, help='Run mode: "pre-train", "fine-tune", or "both"')

    args = parser.parse_args()

    main(pretrain_batch_size=args.pretrain_batch_size, finetuning_batch_size=args.finetuning_batch_size,
         hidden_state=args.hidden_state,
         pre_train_epochs=args.pre_train_epochs,
         fine_tune_epochs=args.fine_tune_epochs,
         opt=args.opt,
         lr=args.lr,
         dropout=args.dropout,
         nhead=args.nhead,
         num_layers=args.num_layers,
         dim_feedforward=args.dim_feedforward,
         fine_tuning_mode=args.fine_tuning_mode,
         run_mode=args.run_mode)
