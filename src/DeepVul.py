import numpy as np
import torch
from torch import nn, optim
from model import SharedFeatureExtractor, GeneEssentialityPrediction, DrugResponsePrediction
from utils import get_corrcoef
import time
import os

class DeepVul:
    def __init__(self, n_input_exp, n_input_ess, n_input_drug,
                 ess_gene_list=None, exp_gene_list=None,
                 nhead=2, num_layers=2, dim_feedforward=2048, hidden_state=500,
                 opt="Adam", lr=0.0001,
                 weight_decay=0, device=None, dropout=0.1,
                 gene_essentiality_model_path=None,
                 drug_response_model_path=None,
                 shared_feature_extractor_path=None,
                 out_dir="saved-figures/essentiality/deepVul/",
                 ):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize shared feature extractor
        self.shared_feature_extractor = SharedFeatureExtractor(
            n_features_embedding=hidden_state, 
            n_features=n_input_exp, 
            nhead=nhead, 
            num_layers=num_layers, 
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        # Initialize gene essentiality prediction model
        self.gene_essentiality_model = GeneEssentialityPrediction(
            shared_feature_extractor=self.shared_feature_extractor,
            num_selected_genes=n_input_ess
        ).to(self.device)

        # Initialize drug response prediction model
        self.drug_response_model = DrugResponsePrediction(
            shared_feature_extractor=self.shared_feature_extractor,
            drug_response_dim=n_input_drug
        ).to(self.device)

        if gene_essentiality_model_path or drug_response_model_path or shared_feature_extractor_path:
            self.load_model_weights(gene_essentiality_model_path, drug_response_model_path, shared_feature_extractor_path)

        # Initialize separate optimizers for each model
        self.optimizer_ess = self._get_optimizer(list(self.gene_essentiality_model.parameters()), opt, lr, weight_decay)
        self.optimizer_drug = self._get_optimizer(list(self.drug_response_model.parameters()), opt, lr, weight_decay)

        self.ess_gene_list = ess_gene_list
        self.exp_gene_list = exp_gene_list
        self.out_dir = out_dir

        # Ensure out_dir exists
        os.makedirs(self.out_dir, exist_ok=True)

        # Save the configuration to JSON
        self._save_config({
            "n_input_exp": n_input_exp,
            "hidden_state": hidden_state,
            "n_input_ess": n_input_ess,
            "n_input_drug": n_input_drug,
            "ess_gene_list": ess_gene_list,
            "exp_gene_list": exp_gene_list,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "opt": opt,
            "lr": lr,
            "weight_decay": weight_decay,
            "device": str(self.device),
            "dropout": dropout,
            "gene_essentiality_model_path": gene_essentiality_model_path,
            "drug_response_model_path": drug_response_model_path,
            "out_dir": out_dir,
            "shared_feature_extractor_path": shared_feature_extractor_path
        })


        # Initialize criterion
        self.criterion = nn.MSELoss()

        # Initialize history
        self.history = {
            "Train": {"total_loss": [], "corr": [], "under0": [], "corr10": []},
            "Val": {"total_loss": [], "corr": [], "under0": [], "corr10": []},
            "Test": {"total_loss": [], "corr": [], "under0": [], "corr10": []}
        }


    def _save_config(self, config_dict):
        """Save configuration dictionary to JSON in out_dir as config.json."""
        config_path = os.path.join(self.out_dir, 'config.json')
        with open(config_path, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)
        print(f"Configuration saved to {config_path}")

    def load_model_weights(self, gene_essentiality_model_path=None, drug_response_model_path=None, shared_feature_extractor_path=None):
        """Load pre-trained weights for models and shared feature extractor if paths are provided."""
        if gene_essentiality_model_path:
            self.gene_essentiality_model.load_state_dict(torch.load(gene_essentiality_model_path, map_location=self.device))
            print(f"Loaded gene essentiality model weights from {gene_essentiality_model_path}")

        if drug_response_model_path:
            self.drug_response_model.load_state_dict(torch.load(drug_response_model_path, map_location=self.device))
            print(f"Loaded drug response model weights from {drug_response_model_path}")
        
        if shared_feature_extractor_path:
            self.shared_feature_extractor.load_state_dict(torch.load(shared_feature_extractor_path, map_location=self.device))
            print(f"Loaded shared feature extractor weights from {shared_feature_extractor_path}")

    def save_model_weights(self, gene_essentiality_model_path=None, drug_response_model_path=None, shared_feature_extractor_path=None):
        """Save the current weights of the models and shared feature extractor if paths are provided."""
        
        if gene_essentiality_model_path:
            torch.save(self.gene_essentiality_model.state_dict(), gene_essentiality_model_path)
            print(f"Saved gene essentiality model weights to {gene_essentiality_model_path}")

        if drug_response_model_path:
            torch.save(self.drug_response_model.state_dict(), drug_response_model_path)
            print(f"Saved drug response model weights to {drug_response_model_path}")

        if shared_feature_extractor_path:
            torch.save(self.shared_feature_extractor.state_dict(), shared_feature_extractor_path)
            print(f"Saved shared feature extractor weights to {shared_feature_extractor_path}")

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def l1_regularization(self, model, lambda_l1 = 0.01):
        
        l1_norm = sum(param.abs().sum() for param in model.parameters())
        return lambda_l1 * l1_norm

    def _get_optimizer(self, parameters, opt, lr, weight_decay):
        if opt.lower() == "adam":
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif opt.lower() == "adamw":
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif opt.lower() == "adagrad":
            return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
        elif opt.lower() == "sgd":
            return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {opt}")

    def train(self, loader, mode="pre-train", l1 = False):
        if mode == "pre-train":
            model = self.gene_essentiality_model
            optimizer = self.optimizer_ess
        elif mode == "fine-tune":
            model = self.drug_response_model
            optimizer = self.optimizer_drug
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        model.train()
        train_loss = 0
        all_pred, all_true = [], []

        for batch in loader:
            data_ess, data_exp = batch[0].to(self.device), batch[1].to(self.device)

            optimizer.zero_grad()
            outputs = model(data_exp)  # Pass the input directly to the model

            loss = sum(self.criterion(outputs[:, i], data_ess[:, i]) for i in range(data_ess.shape[1])) / data_ess.shape[1]
            if l1:
                l1_penalty = self.l1_regularization(model)
                loss = loss + l1_penalty
            
            all_pred.append(outputs)
            all_true.append(data_ess)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
          
        train_loss /= len(loader)
        all_pred = torch.cat(all_pred, dim=0)
        all_true = torch.cat(all_true, dim=0)
        
        train_corr, train_corr10 = get_corrcoef(all_pred, all_true)

        self.history["Train"]["total_loss"].append(train_loss)
        self.history["Train"]["corr"].append(train_corr.mean().item())
        self.history["Train"]["corr10"].append(train_corr10.mean().item())
        self.history["Train"]["under0"].append(np.where(train_corr < 0)[0].size)

        return train_loss, all_pred, all_true, train_corr, train_corr10

    def evaluate(self, loader, mode="pre-train", l1 = False):
        if mode == "pre-train":
            model = self.gene_essentiality_model
        elif mode == "fine-tune":
            model = self.drug_response_model
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        model.eval()
        val_loss = 0
        all_pred, all_true = [], []

        with torch.no_grad():
            for batch in loader:
                data_ess, data_exp = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(data_exp)  # Pass the input directly to the model

                loss = sum(self.criterion(outputs[:, i], data_ess[:, i]) for i in range(data_ess.shape[1])) / data_ess.shape[1]
                if l1:
                    l1_penalty = self.l1_regularization(model)
                    loss = loss + l1_penalty
                
                
                all_pred.append(outputs)
                all_true.append(data_ess)

                val_loss += loss.item()

        val_loss /= len(loader)
        all_pred = torch.cat(all_pred, dim=0)
        all_true = torch.cat(all_true, dim=0)
        val_corr, val_corr10 = get_corrcoef(all_pred, all_true)

        self.history["Val"]["total_loss"].append(val_loss)
        self.history["Val"]["corr"].append(val_corr.mean().item())
        self.history["Val"]["corr10"].append(val_corr10.mean().item())
        self.history["Val"]["under0"].append(np.where(val_corr < 0)[0].size)

        return val_loss, all_pred, all_true, val_corr, val_corr10

    def pre_train(self, train_loader, val_loader, test_loader, epochs=15, l1 = False):
        
        
        for epoch in range(epochs):
            start_time = time.time()
            print(f"Epoch {epoch+1}/{epochs}")

            train_loss, y_pred_train , y_true_train, train_corr, train_corr10 = self.train(train_loader, mode="pre-train", l1 = l1)
            if val_loader is not None: val_loss, y_pred_val , y_true_val, val_corr, val_corr10 = self.evaluate(val_loader, mode="pre-train", l1 = l1 )
            test_loss, y_pred_test , y_true_test, test_corr, test_corr10 = self.evaluate(test_loader, mode="pre-train", l1 = l1)

            fun = lambda x: round(float(torch.median(x)), 4)

            under0_train = np.where(train_corr < 0)[0].size
            under0_val = np.where(val_corr < 0)[0].size if val_loader is not None else 0
            under0_test = np.where(test_corr < 0)[0].size

            self.history["Train"]["under0"].append(under0_train)
            self.history["Train"]["corr"].append(fun(train_corr))
            self.history["Train"]["corr10"].append(fun(train_corr10))

            if val_loader is not None:
                self.history["Val"]["under0"].append(under0_val)
                self.history["Val"]["corr"].append(fun(val_corr))
                self.history["Val"]["corr10"].append(fun(val_corr10))

            self.history["Test"]["under0"].append(under0_test)
            self.history["Test"]["corr"].append(fun(test_corr))
            self.history["Test"]["corr10"].append(fun(test_corr10))
            

            

            print(f'\tET: {round(time.time() - start_time, 2)} Seconds')
            print(f'\tTrain Loss: {round(train_loss, 4)}, train_corr: {fun(train_corr)}, train_corr10: {fun(train_corr10)}, #Neg: {under0_train}')
            if val_loader is not None:
                print(f'\tVal Loss: {round(val_loss, 4)}, val_corr: {fun(val_corr)}, val_corr10: {fun(val_corr10)}, #Neg: {under0_val}')
            print(f'\tTest Loss: {round(test_loss, 4)}, test_corr: {fun(test_corr)}, test_corr10: {fun(test_corr10)}, #Neg: {under0_test}\n')

    def fine_tune(self, train_loader, val_loader, test_loader, epochs=15, mode="freeze-shared", l1 = False):

        if mode == "freeze-shared":
            for param in self.shared_feature_extractor.parameters():
                param.requires_grad = False
        elif mode == "tune-shared":
            for param in self.shared_feature_extractor.parameters():
                param.requires_grad = True
        elif mode == "initial-shared":
            self.shared_feature_extractor.apply(self._initialize_weights)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for epoch in range(epochs):
            start_time = time.time()
            print(f"Epoch {epoch+1}/{epochs}")

            train_loss, y_pred_train , y_true_train, train_corr, train_corr10 = self.train(train_loader, mode="fine-tune", l1 = l1)
            if val_loader is not None: val_loss, y_pred_val , y_true_val, val_corr, val_corr10 = self.evaluate(val_loader, mode="fine-tune", l1 = l1) 
            test_loss, y_pred_test , y_true_test, test_corr, test_corr10 = self.evaluate(test_loader, mode="fine-tune", l1 = l1)

            fun = lambda x: round(float(torch.median(x)), 4)

            under0_train = np.where(train_corr < 0)[0].size
            under0_val = np.where(val_corr < 0)[0].size if val_loader is not None else 0
            under0_test = np.where(test_corr < 0)[0].size

            self.history["Train"]["under0"].append(under0_train)
            self.history["Train"]["corr"].append(fun(train_corr))
            self.history["Train"]["corr10"].append(fun(train_corr10))

            if val_loader is not None:
                self.history["Val"]["under0"].append(under0_val)
                self.history["Val"]["corr"].append(fun(val_corr))
                self.history["Val"]["corr10"].append(fun(val_corr10))

            self.history["Test"]["under0"].append(under0_test)
            self.history["Test"]["corr"].append(fun(test_corr))
            self.history["Test"]["corr10"].append(fun(test_corr10))
            

            print(f'\tET: {round(time.time() - start_time, 2)} Seconds')
            print(f'\tTrain Loss: {round(train_loss, 4)}, train_corr: {fun(train_corr)}, train_corr10: {fun(train_corr10)}, #Neg: {under0_train}')
            if val_loader is not None:
                print(f'\tVal Loss: {round(val_loss, 4)}, val_corr: {fun(val_corr)}, val_corr10: {fun(val_corr10)}, #Neg: {under0_val}')
            print(f'\tTest Loss: {round(test_loss, 4)}, test_corr: {fun(test_corr)}, test_corr10: {fun(test_corr10)}, #Neg: {under0_test}\n')

    def fine_tune_on_experimental(self, X_braf, y_braf, epochs=10, lr=0.0001, split_ratio=0.8, out_dir="saved-figures/experimental_data/"):
        """
        Fine-tunes the BRAF head of the essentiality model on experimental BRAF data (binary classification).
        """
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Identify the BRAF head index
        try:
            braf_index = self.ess_gene_list.index("braf")
        except ValueError:
            try:
                braf_index = self.ess_gene_list.index("BRAF")
            except ValueError:
                raise ValueError("BRAF not found in the gene list")

        # Freeze all layers except the BRAF head
        for param in self.shared_feature_extractor.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.gene_essentiality_model.output_layers):
            for param in layer.parameters():
                param.requires_grad = (i == braf_index)  # Unfreeze only the BRAF head

        # Set up optimizer for the BRAF head
        optimizer = self._get_optimizer(
            self.gene_essentiality_model.output_layers[braf_index].parameters(), opt="Adam", lr=lr, weight_decay=0
        )

        # Convert data to tensors and split into train/test sets
        X_braf = torch.tensor(X_braf, dtype=torch.float32).to(self.device)
        y_braf = torch.tensor(y_braf, dtype=torch.float32).to(self.device)
        
        indices = list(range(X_braf.size(0)))
        random.shuffle(indices)
        split = int(len(indices) * split_ratio)
        train_indices, test_indices = indices[:split], indices[split:]

        train_X, train_y = X_braf[train_indices], y_braf[train_indices]
        test_X, test_y = X_braf[test_indices], y_braf[test_indices]

        # Fine-tune over multiple epochs
        best_test_acc = 0
        sigmoid = torch.nn.Sigmoid()

        for epoch in range(epochs):
            train_loss, train_acc, train_f1, train_auc = self._train_braf_head(train_X, train_y, optimizer, braf_index, sigmoid)

            test_loss, test_acc, test_f1, test_auc, preds, outputs, _ = self._evaluate_braf_head(test_X, test_y, braf_index, sigmoid)

            print(f"Experimental Fine-Tuning Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")

            # Save only the BRAF head weights if test accuracy improves
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_auc = test_auc
                best_test_f1 = test_f1
                best_test_preds = preds
                best_test_outputs = outputs
                braf_head_path = os.path.join(out_dir, 'fine_tuned_braf_head.pth')
                torch.save(self.gene_essentiality_model.output_layers[braf_index].state_dict(), braf_head_path)

                braf_extractor_path = os.path.join(out_dir, 'fine_tuned_braf_shared_extractor.pth')
                torch.save(self.gene_essentiality_model.shared_feature_extractor.state_dict(), braf_extractor_path)

                print(f"Saved fine-tuned BRAF head weights to {braf_head_path}")

        return best_test_acc, best_test_auc, best_test_f1, best_test_preds, best_test_outputs

    def few_shot(self, X, y, patient_ids, epochs=15, shots=5, l1=False, lr=0.0001, shared_lr=0.00001,
                fewshot_out_dir="saved-figures/patient_data/",
                experimental_braf_model_path="saved-figures/patient_data/"):
        """
        Few-shot learning on patient data with binary classification, training both the BRAF head and shared extractor.
        """
        if fewshot_out_dir and not os.path.exists(fewshot_out_dir):
            os.makedirs(fewshot_out_dir)

        # Identify the BRAF head index
        try:
            braf_index = self.ess_gene_list.index("braf")
        except ValueError:
            try:
                braf_index = self.ess_gene_list.index("BRAF")
            except ValueError:
                raise ValueError("BRAF not found in the gene list")

        # Load the fine-tuned weights from the experimental BRAF data
        if os.path.exists(experimental_braf_model_path):
            p1 = os.path.join(experimental_braf_model_path, 'fine_tuned_braf_shared_extractor.pth')
            self.gene_essentiality_model.shared_feature_extractor.load_state_dict(torch.load(p1))

            p2 = os.path.join(experimental_braf_model_path, 'fine_tuned_braf_head.pth')
            self.gene_essentiality_model.output_layers[braf_index].load_state_dict(torch.load(p2))
            
            print("Loaded fine-tuned BRAF weights for few-shot training.")
        else:
            raise FileNotFoundError(f"Fine-tuned BRAF weights not found at {experimental_braf_model_path}")

        # Set requires_grad = True for the shared feature extractor and BRAF head
        for param in self.gene_essentiality_model.shared_feature_extractor.parameters():
            param.requires_grad = True
        for param in self.gene_essentiality_model.output_layers[braf_index].parameters():
            param.requires_grad = True

        # Set up two optimizers: one for the BRAF head and one for the shared feature extractor
        braf_optimizer = self._get_optimizer(
            self.gene_essentiality_model.output_layers[braf_index].parameters(), opt="Adam", lr=lr, weight_decay=0
        )
        shared_optimizer = self._get_optimizer(
            self.gene_essentiality_model.shared_feature_extractor.parameters(), opt="Adam", lr=shared_lr, weight_decay=0
        )

        # Convert data to tensors
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Few-shot split
        indices = list(range(X.size(0)))
        random.shuffle(indices)
        train_indices = indices[:shots]
        test_indices = indices[shots:]

        train_X, train_y = X[train_indices], y[train_indices]
        test_X, test_y = X[test_indices], y[test_indices]
        train_patient_ids = [patient_ids[i] for i in train_indices]
        test_patient_ids = [patient_ids[i] for i in test_indices]

        # Few-shot training
        sigmoid = torch.nn.Sigmoid()
        best_y_pred_test, best_y_true_test, best_test_patient_ids = None, None, None
        optimal_f1 = float("-inf")

        for epoch in range(epochs):
            train_loss, train_acc, train_f1, train_auc = self._train_braf_and_shared(
                train_X, train_y, braf_optimizer, shared_optimizer, braf_index, sigmoid
            )
            test_loss, test_acc, test_f1, test_auc, y_pred_test, y_pred_reg_test, y_true_test = self._evaluate_braf_head(
                test_X, test_y, braf_index, sigmoid
            )

            print(f"Few-Shot Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")

            # Save the best test predictions
            if optimal_f1 < test_f1:
                optimal_f1 = test_f1
                best_y_pred_test = y_pred_test
                best_y_pred_reg_test = y_pred_reg_test
                best_y_true_test = y_true_test
                best_test_patient_ids = test_patient_ids
                torch.save(self.gene_essentiality_model.output_layers[braf_index].state_dict(),
                        os.path.join(fewshot_out_dir, f'fine_tuned_braf_head_{shots}_shot.pth'))

        self._save_results(best_y_pred_test, best_y_pred_reg_test, best_y_true_test, best_test_patient_ids, fewshot_out_dir)
        return best_y_pred_test, best_y_pred_reg_test, best_y_true_test, best_test_patient_ids, test_acc, test_f1, test_auc

    def _train_braf_head(self, X, y, optimizer, braf_index, sigmoid):
        """
        Helper function to train only the BRAF head for binary classification.
        """
        self.gene_essentiality_model.train()
        optimizer.zero_grad()
        
        outputs = sigmoid(self.gene_essentiality_model.get_single_head_output(X, braf_index))
        outputs = outputs.view_as(y)

        # print("shapes:", outputs.size(), y.size())

        loss = BCE(outputs, y)
        loss.backward()
        optimizer.step()

        # Compute metrics
        preds = (outputs > 0.5).float()
        accuracy = accuracy_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu())
        auc = roc_auc_score(y.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        return loss.item(), accuracy, f1, auc
    
    def _train_braf_and_shared(self, X, y, braf_optimizer, shared_optimizer, braf_index, sigmoid):
        """
        Helper function to train both the BRAF head and the shared feature extractor for binary classification.
        """
        self.gene_essentiality_model.train()
        # self.shared_feature_extractor.train()

        # Zero the parameter gradients
        braf_optimizer.zero_grad()
        shared_optimizer.zero_grad()

        # Forward pass through shared feature extractor and BRAF head
        # shared_features = self.shared_feature_extractor(X)
        outputs = sigmoid(self.gene_essentiality_model.get_single_head_output(X, braf_index))
        outputs = outputs.view_as(y)

        # Calculate loss and backpropagate for both optimizers
        loss = BCE(outputs, y)
        loss.backward()

        # Update both optimizers
        braf_optimizer.step()
        shared_optimizer.step()

        # Compute metrics
        preds = (outputs > 0.5).float()
        accuracy = accuracy_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu())
        auc = roc_auc_score(y.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        return loss.item(), accuracy, f1, auc

    def _evaluate_braf_head(self, X, y, braf_index, sigmoid):
        """
        Helper function to evaluate the BRAF head for binary classification.
        """
        self.gene_essentiality_model.eval()
        with torch.no_grad():
            outputs = sigmoid(self.gene_essentiality_model.get_single_head_output(X, braf_index))
            loss = BCE(outputs.squeeze(), y)

            preds = (outputs > 0.5).float()
            accuracy = accuracy_score(y.cpu(), preds.cpu())
            f1 = f1_score(y.cpu(), preds.cpu())
            auc = roc_auc_score(y.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        
        return loss.item(), accuracy, f1, auc, preds, outputs, y

    def _save_results(self, y_pred, y_pred_reg, y_true, patient_ids, out_dir):
        # Convert tensors to numpy arrays if they are not already
        binary_preds = y_pred.cpu().numpy().flatten()
        regression_preds = y_pred_reg.cpu().numpy().flatten()
        true_values = y_true.cpu().numpy().flatten()
        patient_ids = np.array(patient_ids).flatten()  # Ensure patient_ids is also 1D

        # Prepare results in a DataFrame
        results_df = pd.DataFrame({
            "PatientID": patient_ids,
            "TrueValue": true_values,
            "RegressionPrediction": regression_preds,
            "BinaryPrediction": binary_preds
        })

        # Define the file path and save as CSV
        output_file = os.path.join(out_dir, "patient_predictions.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
