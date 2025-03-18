import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import plotly.express as px
import plotly.graph_objects as goc
import pandas as pd
import pickle
import gc
import copy
from tqdm import tqdm

class GenomicSignalVAE(nn.Module):
    def __init__(self, n_features=None, window_size=None, latent_dim=None):
        super(GenomicSignalVAE, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.input_dim = n_features * window_size
        
        self.enc1 = nn.Linear(self.input_dim, 12288)
        self.enc_norm1 = nn.BatchNorm1d(12288)
        self.enc2 = nn.Linear(12288, 6144)
        self.enc_norm2 = nn.BatchNorm1d(6144)
        self.enc3 = nn.Linear(6144, 3072)
        self.enc_norm3 = nn.BatchNorm1d(3072)
        self.enc4 = nn.Linear(3072, 1536)
        self.enc_norm4 = nn.BatchNorm1d(1536)
        
        self.mish = nn.Mish()
        
        self.fc_mu = nn.Linear(1536, latent_dim)
        self.fc_logvar = nn.Linear(1536, latent_dim)
        
        self.dec1 = nn.Linear(latent_dim, 1536)
        self.dec_norm1 = nn.BatchNorm1d(1536)
        self.dec2 = nn.Linear(1536, 3072)
        self.dec_norm2 = nn.BatchNorm1d(3072)
        self.dec3 = nn.Linear(3072, 6144)
        self.dec_norm3 = nn.BatchNorm1d(6144)
        self.dec4 = nn.Linear(6144, 12288)
        self.dec_norm4= nn.BatchNorm1d(12288)
        
        self.output_layer = nn.Linear(12288, self.input_dim)
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        e1 = self.mish(self.enc_norm1(self.enc1(x)))
        e2 = self.mish(self.enc_norm2(self.enc2(e1)))
        e3 = self.mish(self.enc_norm3(self.enc3(e2)))
        e4 = self.mish(self.enc_norm4(self.enc4(e3)))
        self.skip_connections = [e1, e2, e3, e4]
        mu = self.fc_mu(e4)
        logvar = self.fc_logvar(e4)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        d1 = self.mish(self.dec_norm1(self.dec1(z)))
        d1 = d1 + self.skip_connections[3]
        d2 = self.mish(self.dec_norm2(self.dec2(d1)))
        d2 = d2 + self.skip_connections[2]
        d3 = self.mish(self.dec_norm3(self.dec3(d2)))
        d3 = d3 + self.skip_connections[1]
        d4 = self.mish(self.dec_norm4(self.dec4(d3)))
        d4 = d4 + self.skip_connections[0]
        output = self.output_layer(d4)
        output = output.view(output.size(0), self.n_features, self.window_size)
        
        return output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
    total_loss = recon_loss + beta * kl_div
    return total_loss

def train_vae(data, epochs=10, batch_size=1024, latent_dim=32, lr=1e-3, beta=1.0, device='cpu', load_from_file=False):
    """
    Train the VAE model on genomic signal data
    """
    if not load_from_file:
        x_train, x_val = train_test_split(data, test_size=0.1, random_state=42)
        
        train_tensor = torch.FloatTensor(x_train)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        val_tensor = torch.FloatTensor(x_val)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=True)
        
        model = GenomicSignalVAE(n_features=data.shape[1], window_size=data.shape[2], latent_dim=latent_dim)
        model = nn.DataParallel(model)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loss_list = []
        val_loss_list = []
        
        epoch_pbar = tqdm(range(epochs), desc="Training VAE")
        for epoch in epoch_pbar:

            batch_train_loss = 0
            for batch_idx, (x,) in enumerate(train_loader):
                x = x.to(device)
                model.train()
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = model(x)
                loss = vae_loss(recon_batch, x, mu, logvar, beta)
                batch_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            batch_val_loss = 0
            for batch_idx, (x,) in enumerate(val_loader):
                x = x.to(device)
                model.eval()
                with torch.no_grad():
                    recon, mu, logvar = model(x)
                    val_loss = vae_loss(recon, x, mu, logvar, beta)
                    batch_val_loss += val_loss.item()

            epoch_pbar.set_postfix({
                "train_loss": f"{batch_train_loss / len(train_loader):.4f}",
                "val_loss": f"{batch_val_loss / len(val_loader):.4f}"
            })

            train_loss_list.append(batch_train_loss / len(train_loader))
            val_loss_list.append(batch_val_loss / len(val_loader))
        
        unwrapped_model = model.module
        torch.save(unwrapped_model.state_dict(), 'vae_model.pth')

        plt.figure(figsize=(10, 4))
        plt.plot(range(1, epochs+1), train_loss_list, label='Training Loss')
        plt.plot(range(1, epochs+1), val_loss_list, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss (MSE)')
        plt.legend(loc='upper right')
        plt.savefig('vae_loss.png', dpi=300)
        plt.close()
    else:
        model = GenomicSignalVAE(n_features=data.shape[1], window_size=data.shape[2], latent_dim=latent_dim)
        model.load_state_dict(torch.load('vae_model.pth'))
        model = nn.DataParallel(model)
        model = model.to(device)
    
    full_tensor = torch.FloatTensor(data)
    full_loader = DataLoader(TensorDataset(full_tensor), batch_size=batch_size)

    batch_latent_representation = []
    for batch_idx, (x,) in enumerate(full_loader):
        x = x.to(device)
        model.eval()
        with torch.no_grad():
            mu, _ = model.module.encode(x)
            batch_latent_representation.append(mu.cpu().numpy())

    latent_representation = np.concatenate(batch_latent_representation, axis=0)
    
    return latent_representation

def visualize_latent_space_plotly(latent_representation, ccre_metadata, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', n_epochs=None, unique=False, init='spectral'):
    """
    Create interactive Plotly visualization of latent space with metadata
    """
    n_samples = int(latent_representation.shape[0] * 0.2)
    sampled_indices = np.random.choice(latent_representation.shape[0], size=n_samples, replace=False)
    sampled_metadata = ccre_metadata.iloc[sampled_indices].reset_index(drop=True)
    sampled_latent_representation = copy.deepcopy(latent_representation[sampled_indices])
    print(f"Sampled latent representation shape: {sampled_latent_representation.shape}")

    del latent_representation
    gc.collect()

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, verbose=True, n_epochs=n_epochs, unique=unique, init=init)
    embedding = reducer.fit_transform(sampled_latent_representation)
    
    viz_df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
    })
    
    if not len(viz_df) == len(sampled_metadata):
        raise ValueError("Length of viz_df must match length of ccre_df")
    
    viz_df['rDHS'] = sampled_metadata['rDHS'].values
    viz_df['cCRE_type'] = sampled_metadata['cCRE_type'].values
    viz_df['location'] = sampled_metadata['chrom'].astype(str) + ':' + \
                        sampled_metadata['start'].astype(str) + '-' + \
                        sampled_metadata['end'].astype(str)
    
    viz_df['hover_text'] = viz_df.apply(
        lambda row: f"<b>{row['rDHS']}</b><br>" +
                   f"Type: {row['cCRE_type']}<br>" +
                   f"Location: {row['location']}",
        axis=1
    )
    
    fig = px.scatter(
        viz_df, 
        x='x', 
        y='y',
        color='cCRE_type',
        hover_data=['rDHS', 'cCRE_type', 'location'],
        custom_data=['hover_text'],
        title=f'Latent Space Visualization using UMAP',
        labels={'x': 'Component 1', 'y': 'Component 2'},
        color_discrete_map={
            'PLS': '#FF0000',
            'pELS': '#FFA700',
            'dELS': '#FFCD00',
            'CA-H3K4me3': '#FFAAAA',
            'CA-CTCF': '#00B0F0',
            'CA-only': '#06DA93',
            'CA-TF': '#BE28E5',
            'TF-only': '#D876EC'
        },
        opacity=0.5
    )
    
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
        marker=dict(size=4)
    )

    fig.update_layout(
        legend_title_text='cCRE Type',
        height=800,
        width=1000,
        template='plotly_white'
    )
    
    return fig

if __name__ == "__main__":
    # print("Loading data...")
    # data = np.load("features.npy")
    # data = data.reshape(-1, 244, 160)

    # print("Training VAE...")
    # latent_representation = train_vae(data, epochs=10, batch_size=16384, latent_dim=512, lr=1e-3, beta=2.0, device='cuda', load_from_file=False)

    # print("Saving latent representation...")
    # np.save("latent_representation.npy", latent_representation)
    # print(latent_representation.shape)

    print("Loading ccre metadata...")
    ccre_metadata = pd.read_csv("ccre_df.tsv", sep="\t", header=0)

    print("Loading latent representation...")
    latent_representation = np.reshape(np.load("latent_representation.npy"), (169124, -1))

    print("Visualizing latent space...")
    fig = visualize_latent_space_plotly(latent_representation, ccre_metadata, n_neighbors=30, min_dist=1e-1, n_components=2, metric='correlation', unique=True)

    print("Saving visualization...")
    fig.write_html("ccre_vae_results.html")