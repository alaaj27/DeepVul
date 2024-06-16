import numpy as np
import torch
from torch import nn, optim
from model import SharedFeatureExtractor, GeneEssentialityPrediction, DrugResponsePrediction
from utils import get_corrcoef
import time


class DeepVul:
    def __init__(self, n_input_exp, hidden_state,
                 n_input_ess, n_input_drug,
                 nhead, num_layers, dim_feedforward,
                 opt="Adam", lr=0.0001,
                 weight_decay=0, device=None, dropout=0.1):
        
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

        # Initialize separate optimizers for each model
        self.optimizer_ess = self._get_optimizer(list(self.gene_essentiality_model.parameters()), opt, lr, weight_decay)
        self.optimizer_drug = self._get_optimizer(list(self.drug_response_model.parameters()), opt, lr, weight_decay)

        # Initialize criterion
        self.criterion = nn.MSELoss()

        # Initialize history
        self.history = {
            "Train": {"total_loss": [], "corr": [], "under0": [], "corr10": []},
            "Val": {"total_loss": [], "corr": [], "under0": [], "corr10": []},
            "Test": {"total_loss": [], "corr": [], "under0": [], "corr10": []}
        }

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

