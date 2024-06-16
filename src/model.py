import torch
from torch import nn

class SharedFeatureExtractor(nn.Module):
    def __init__(
        self, 
        n_features_embedding: int,  # The size of the embedding for each feature after initial linear transformation.
        n_features: int,  # The number of input features (19193 for gene expression data).
        nhead: int,  # The number of attention heads in the transformer encoder.
        num_layers: int,  # The number of transformer encoder layers.
        dim_feedforward: int,  # The dimension of the feedforward network in the transformer.
        activation: str = 'relu',  # The activation function to use in the transformer encoder.
        dropout:float =0.1
    ):
        super(SharedFeatureExtractor, self).__init__()
        self.n_features = n_features
        self.n_features_embedding = n_features_embedding
        self.nhead = nhead
        self.num_encoder_layers = num_layers
        self.dim_feedforward = dim_feedforward

        # Initial linear transformation to reduce feature dimensions
        self.linear_1 = nn.Linear(self.n_features, self.n_features_embedding)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.n_features_embedding, 
            dim_feedforward=self.dim_feedforward, 
            nhead=self.nhead, 
            activation=activation,
            dropout=dropout
        )
        
        # Transformer encoder
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_encoder_layers)
        
        # Layer normalization
        self.layer_norm_out = nn.LayerNorm(self.n_features_embedding)        
   
    def forward(self, x):  
        # Apply initial linear transformation
        out = self.linear_1(x)
        
        out = self.dropout(out)
        
        if out.dim() == 2:  # Add a dummy sequence dimension if necessary
            out = out.unsqueeze(1)
            
        # Permute dimensions for transformer encoder
        out = out.permute(1, 0, 2)
        
        # Pass through transformer encoder
        out = self.encoder(out)
        
        # Permute back
        out = out.permute(1, 0, 2) 
        
        # Mean pooling
        out = torch.mean(out, dim=1)
        
        # Layer normalization
        out = self.layer_norm_out(out)        
        
        return out
    

class GeneEssentialityPrediction(nn.Module):
    def __init__(self, shared_feature_extractor, num_selected_genes: int):
        super(GeneEssentialityPrediction, self).__init__()
        
        
        self.shared_feature_extractor = shared_feature_extractor
        self.num_selected_genes = num_selected_genes
        
        # Output layers for each gene's essentiality prediction
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.shared_feature_extractor.n_features_embedding, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(self.num_selected_genes)
        ])
        
    def forward(self, x):  
        shared_features = self.shared_feature_extractor(x)
        
        # Generate outputs for each gene's essentiality
        outputs = [output_layer(shared_features) for output_layer in self.output_layers]
        outputs = torch.cat(outputs, dim=1)
        
        return outputs


class DrugResponsePrediction(nn.Module):
    def __init__(self, shared_feature_extractor, drug_response_dim: int):
        super(DrugResponsePrediction, self).__init__()
        self.shared_feature_extractor = shared_feature_extractor
        
        # Output layer for drug response prediction
        self.drug_response_layer = nn.Sequential(
            nn.Linear(self.shared_feature_extractor.n_features_embedding, 256),
            nn.ReLU(),
            nn.Linear(256, drug_response_dim)
        )
        
    def forward(self, x):  
        shared_features = self.shared_feature_extractor(x)
        
        # Generate output for drug response
        drug_response_output = self.drug_response_layer(shared_features)
        
        return drug_response_output

    