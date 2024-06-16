from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler


class JointDataset(Dataset):
    def __init__(self, dataset, from_keys, normalize_exp="StandardScaler"):
        
        self.data_ess = []
        self.data_exp = []
        self.cell_line = []

        self.keys = from_keys
        
        for cell in from_keys: #dataset.keys():
            self.data_ess.append(dataset[cell]["data_ess"])
            self.data_exp.append(dataset[cell]["data_exp"])
            self.cell_line.append(cell)
    
        self.data_ess = torch.stack(self.data_ess).float()
        self.data_exp = torch.stack(self.data_exp).float()
        
        if normalize_exp is None :
            print ("No expression normalization ...")
    
        elif normalize_exp == "StandardScaler":

            print ("Normalizing expression- StandardScaler ...")
            self.data_exp = self.normalize_tensor_rows(self.normalize_tensor_columns(self.data_exp))
                
        else:
            raise ValueError (f"normalize_exp error:{normalize_exp}")

        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        return self.data_ess[index], self.data_exp[index], self.cell_line[index]
            

    def normalize_tensor_rows(self, tensor):
        np_array = tensor.numpy().T
        scaler = StandardScaler()
        np_array_scaled = scaler.fit_transform(np_array)
        tensor_scaled = torch.tensor(np_array_scaled.T)

        return tensor_scaled

    def normalize_tensor_columns(self, tensor):

        np_array = tensor.numpy()
        scaler = StandardScaler()
        np_array_scaled = scaler.fit_transform(np_array)
        tensor_scaled = torch.tensor(np_array_scaled)
        return tensor_scaled