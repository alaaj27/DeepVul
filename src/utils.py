import argparse
import numpy as np
import pandas as pd
import random
import warnings
from scipy.stats import spearmanr
import os
import json
import time
import pickle
import datetime
from sklearn.model_selection import train_test_split
import torch
from JointDataset import JointDataset
from torch.utils.data import DataLoader

def load_drug_data(batch_size=None, val_split=False):
    print("Loading drug response data ...")

    drug = pd.read_csv("../data/primary-screen-replicate-collapsed-logfold-change.csv").rename(
        columns={"Unnamed: 0": 'CellLine'}).set_index('CellLine')

    drug.columns = drug.columns.str.replace(r'[^a-zA-Z0-9]', '.', regex=True)

    with open("../data/top500_variable_drug.txt", 'r') as file:
        variable_drug = file.read().splitlines()

    drug = drug[variable_drug].fillna(drug.mean())

    print("Loading gene expression data ...")
    Expression = pd.read_csv("../data/OmicsExpressionProteinCodingGenesTPMLogp1.csv").rename(
        columns={"Unnamed: 0": 'CellLine'}).set_index('CellLine')

    return process_data(drug, Expression, batch_size, val_split)


def load_essentiality_data(batch_size=None, val_split=False):
    print("Loading gene essentiality data ...")
    
    Essentiality = pd.read_csv("../data/CRISPRGeneEffect.csv").rename(
        columns={"ModelID": 'CellLine'}).set_index('CellLine')
       
     
    print("Loading gene expression data ...")
    Expression = pd.read_csv("../data/OmicsExpressionProteinCodingGenesTPMLogp1.csv").rename(
        columns={"Unnamed: 0": 'CellLine'}).set_index('CellLine')
    

    Essentiality.columns = [col.split(" ")[0] for col in Essentiality.columns]
    Expression.columns = [col.split(" ")[0] for col in Expression.columns]

    with open("../data/top1000_variable_genes.txt", 'r') as file:
        var_features = file.read().splitlines()
    Essentiality = Essentiality[var_features].dropna(axis='columns')


    return process_data(Essentiality, Expression, batch_size, val_split)



def process_data(other_modality, Expression, batch_size, val_split):

    Joint_data, ess_gene_list, exp_gene_list = preprocess(other_modality, Expression)

    random_cell_line = random.choice(list(Joint_data.keys()))
    n_input_other = Joint_data[random_cell_line]["data_ess"].size(0)
    n_input_exp = Joint_data[random_cell_line]["data_exp"].size(0)


    if val_split:
        train_cells, val_cells = train_test_split(list(Joint_data.keys()), test_size=0.3, random_state=42)
        val_cells, test_cells = train_test_split(val_cells, test_size=0.5, random_state=42)
        train_data = JointDataset(Joint_data, train_cells)
        val_data = JointDataset(Joint_data, val_cells)
        test_data = JointDataset(Joint_data, test_cells)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        train_cells, test_cells = train_test_split(list(Joint_data.keys()), test_size=0.2, random_state=42)
        train_data = JointDataset(Joint_data, train_cells)
        test_data = JointDataset(Joint_data, test_cells)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = None
        val_data = []

    print(f"Split: {len(train_data)}, Size of Val: {len(val_data)}, Size of Test: {len(test_data)}")
    print(f"Expression: {train_loader.dataset[0][1].shape}")
    print(f"Predict: {train_loader.dataset[0][0].shape}")

    return train_loader, val_loader, test_loader, n_input_other, n_input_exp, ess_gene_list, exp_gene_list


def get_corrcoef (data1 , data2, dim=0, corr="spearman"):
    
    if isinstance(data1, torch.Tensor):

        if data1.is_cuda:
            data1 = data1.cpu().detach().numpy()
            data2 = data2.cpu().detach().numpy()
        else:
            data1 = data1.detach().numpy()
            data2 = data2.detach().numpy()
            
    if corr.lower() == "pearson":
        
        if dim == 0:
            corrcoef =[np.corrcoef(data1[:,i], data2[:,i])[0, 1] for i in range(data1.shape[1])]
            corrcoef = np.array(corrcoef)
            corrcoef = corrcoef[~np.isnan(corrcoef)]
        elif dim == 1:
            corrcoef =[np.corrcoef(data1[i,:], data2[i,:])[0, 1] for i in range(data1.shape[0])]
            corrcoef = np.array(corrcoef)
            corrcoef = corrcoef[~np.isnan(corrcoef)]
        else:
            raise ValueError(f"dim value erro: dim={dim}")
        
    elif corr.lower() == "spearman":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if dim == 0:
                corrcoef =[spearmanr(data1[:,i], data2[:,i])[0] for i in range(data1.shape[1])]
                corrcoef = np.array(corrcoef)
                corrcoef = corrcoef[~np.isnan(corrcoef)]
            elif dim == 1:
                corrcoef =[spearmanr(data1[i,:], data2[i,:])[0] for i in range(data1.shape[0])]
                corrcoef = np.array(corrcoef)
                corrcoef = corrcoef[~np.isnan(corrcoef)]
            else:
                raise ValueError(f"dim value erro: dim={dim}")

    else:
        raise ValueError(f"Correlation Name Error: {corr}")



    corrcoef = torch.tensor(corrcoef)

    highest_indices = torch.topk(corrcoef, int(0.1 * len(corrcoef)), largest=True).indices
    corrcoef10 = corrcoef[highest_indices] 

    return corrcoef, corrcoef10



def preprocess (Essentiality, Expression):

    dataset_processed = dict()
    common_cells = set(Expression.index)  & set(Essentiality.index)
    for cell in common_cells:

        dataset_processed[cell] = {
            "data_ess": torch.tensor(Essentiality.loc[cell]),
            "data_exp": torch.tensor(Expression.loc[cell])
        }
        
    ess_gene_list, exp_gene_list = list(Essentiality.columns), list(Expression.columns)
    return dataset_processed, ess_gene_list, exp_gene_list


def plot_corrcoef(predicted_ess, original_ess, path="", save = True, show=False, title=""):
    
    genes_corrcoef, genes_corrcoef10 = get_corrcoef(predicted_ess, original_ess, dim=0)
    cell_corrcoef, cell_corrcoef10 = get_corrcoef(predicted_ess, original_ess, dim=1)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = [genes_corrcoef, genes_corrcoef10, cell_corrcoef, cell_corrcoef10 ]
        bp = plt.boxplot(data, labels=['Genes', 'Genes 10%', 'Cells', 'Cells 10%'],
                         patch_artist=True, showfliers = True, notch = True, meanline = True)
        
        bp['boxes'][0].set_facecolor("lightblue")
        bp['boxes'][1].set_facecolor("lightblue")
        bp['boxes'][2].set_facecolor("brown")
        bp['boxes'][3].set_facecolor("brown")
        

    plt.ylabel('Correlation')
    plt.title(title, fontsize= 10)
    plt.text(0.65, np.median(genes_corrcoef), round(np.median(genes_corrcoef),2) , fontsize=8, rotation= 90)
    plt.text(1.65, np.median(genes_corrcoef10), round(np.median(genes_corrcoef10),2), fontsize=8, rotation= 90)
    plt.text(2.65, np.median(cell_corrcoef), round(np.median(cell_corrcoef),2), fontsize=8, rotation= 90)
    plt.text(3.65, np.median(cell_corrcoef10), round(np.median(cell_corrcoef10),2), fontsize=8, rotation= 90)
    
    if show:
        plt.show()
    if save:
        plt.savefig(path)
    
    plt.close()
            
def predict(deep_vul, gene_expressions, mode="pre-train"):
    
    
    deep_vul.shared_feature_extractor.eval()
    if mode == "pre-train":
        model = deep_vul.gene_essentiality_model
    elif mode == "fine-tune":
        model = deep_vul.drug_response_model
    else:
        raise ValueError(f"Invalid mode: {mode}")

    model.eval()
    predictions = []

    with torch.no_grad():
        if isinstance(gene_expressions, torch.utils.data.DataLoader):
            for batch in gene_expressions:
                data_exp = batch[1].to(deep_vul.device)
                preds = model(data_exp)
                predictions.append(preds.cpu().numpy())
            predictions = np.vstack(predictions)
            
        elif isinstance(gene_expressions, torch.Tensor):
            gene_expressions = gene_expressions.float().to(deep_vul.device)
            predictions = model(gene_expressions).cpu().numpy()
            
                  
        else:
            gene_expressions = torch.tensor(gene_expressions, dtype=torch.float32).to(deep_vul.device)
            predictions = model(gene_expressions).cpu().numpy()

    return predictions

def plot_scatter_with_correlation(actual, pred, show =False, save = True, path=None, title = ""):

    
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    actual_flat = np.array(actual).flatten()
    pred_flat = np.array(pred).flatten()


    spearman_corr, _ = spearmanr(actual_flat, pred_flat)

    plt.scatter(actual_flat, pred_flat)


    min_val = min(min(actual_flat), min(pred_flat))
    max_val = max(max(actual_flat), max(pred_flat))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', label='x == y line', linestyle='--')

    plt.text(0.98, .1, f'Corr= {spearman_corr:.2f}', 
             horizontalalignment='right', verticalalignment='top', 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title}')

    plt.grid()
    plt.legend()
    
    if show:
        plt.show()
        
    if save:
        plt.savefig(path)
        
    plt.close()
    
def load_sanger(ess_gene_list):
    
    Ess_sanger = pd.read_csv("data/sanger_gene_effect.csv").rename(
    columns={"Unnamed: 0": 'CellLine'}).set_index('CellLine')
    
    Ess_sanger.columns = [col.split(" ")[0] for col in Ess_sanger.columns]
    
    Ess_broad = pd.read_csv("data/CRISPRGeneEffect.csv"
                          ).rename(columns={"ModelID":
                                            'CellLine'}).set_index('CellLine')  

    Ess_broad.columns = [col.split(" ")[0] for col in Ess_broad.columns]
    
    column_index_in = []
    column_names_in = []

    for i, col in enumerate (ess_gene_list):
        
        if col in Ess_sanger.columns:
            column_index_in.append(i)
            column_names_in.append(col)
            
    
    Exp = pd.read_csv("data/OmicsExpressionProteinCodingGenesTPMLogp1.csv").rename(
    columns={"Unnamed: 0": 'CellLine'}).set_index('CellLine') 
    Exp.columns = [col.split(" ")[0] for col in Exp.columns]
    
    common_cells = list(set(Exp.index ) & set(Ess_sanger.index))

    Exp = Exp.loc[common_cells]
    Ess_sanger= Ess_sanger.loc[common_cells]
    Ess_broad = Ess_broad.loc[common_cells]
    
    y_test_sanger = Ess_sanger[column_names_in].fillna(Ess_sanger.mean()).values
    y_test_broad = Ess_broad[column_names_in].fillna(Ess_broad.mean()).values
     
    print(f"Ess_sanger: {y_test_sanger.shape}")
    print(f"Ess_broad: {y_test_broad.shape}")
    
    #Standarize Exp
    X_test = normalize_tensor_rows(normalize_tensor_columns(torch.tensor(Exp.values)))
    
    print(f"Exp: {X_test.shape}")
    
     
    return y_test_sanger, y_test_broad, X_test, column_index_in 
