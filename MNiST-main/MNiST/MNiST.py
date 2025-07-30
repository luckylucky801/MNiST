import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess,preprocess1, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import EncoderExplainWrapper,Encoder, Encoder_sparse, Encoder_map, Encoder_sc
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import shap
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft
import scipy.sparse as ss
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.sparse import identity
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import identity, csr_matrix

def plot_signals_overall(original_signals, filtered_signals, title="Overall Signal Comparison"):


    avg_original_signal = np.mean(original_signals, axis=1)
    avg_filtered_signal = np.mean(filtered_signals, axis=1)


    plt.figure(figsize=(9, 6))
    plt.plot(avg_original_signal, label="Before Filtering", alpha=0.6, color="#0057b8", linewidth=2)  
    plt.plot(avg_filtered_signal, label="After Filtering", alpha=0.8, color="#e76f51", linewidth=2)  

    plt.title(title, fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Samples", fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel("Amplitude", fontsize=14, fontweight='bold', labelpad=10)

    plt.legend(fontsize=12, loc="upper right", frameon=True, shadow=True)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.2) 
        spine.set_color("black")  

    plt.show()

def filter_frequency_signal(freq_signal, method='lowpass', cutoff=0.2, smooth=True, sigma=2, wavelet='db1', level=None):

    adj_matrix = _fast_build_adjacency(freq_signal, cutoff)
    
    T_k = _fast_chebyshev_polynomials(adj_matrix, method, cutoff)
    
    if method == 'lowpass':
        filtered_signal = _fast_lowpass_filter(freq_signal, T_k, cutoff)
    elif method == 'wavelet':
        filtered_signal = _fast_wavelet_filter(freq_signal, T_k, level)
    else:
        raise ValueError("Unsupported method. Only 'lowpass' and 'wavelet' are supported.")

    if smooth:
        if freq_signal.ndim > 1:
            for i in range(filtered_signal.shape[1]):
                filtered_signal[:, i] = gaussian_filter1d(filtered_signal[:, i], sigma=sigma)
        else:
            filtered_signal = gaussian_filter1d(filtered_signal, sigma=sigma)
    
    return filtered_signal


def obtain_chebyshev_features(adata, chebyshev_mtx_list, num_features):

    if isinstance(adata, torch.Tensor):
        X = adata.cpu().detach().numpy()
    else:
        X = adata if not ss.issparse(adata) else adata.A
    X = normalize(X, norm='l2', axis=0)
    n_samples, n_input_features = X.shape
    total_features = n_input_features * (len(chebyshev_mtx_list) + 1)
    if total_features > num_features * 4:
        all_features = np.zeros((n_samples, num_features * 2))
        feature_idx = 0
    else:
        all_features = np.zeros((n_samples, total_features))
        feature_idx = 0
    if feature_idx + n_input_features <= all_features.shape[1]:
        all_features[:, feature_idx:feature_idx + n_input_features] = X
        feature_idx += n_input_features
    for i, T in enumerate(chebyshev_mtx_list):
        if feature_idx >= all_features.shape[1]:
            break
        linear_features = T @ X

        if i < 3:  
            if i % 2 == 0:
                enhanced_features = np.tanh(linear_features * 0.5) 
            else:
                enhanced_features = linear_features * (1.0 + 0.1 * np.sign(linear_features))
        else:
            enhanced_features = linear_features

        weight = np.exp(-i * 0.15)  
        weighted_features = weight * enhanced_features

        remaining_slots = all_features.shape[1] - feature_idx
        features_to_add = min(weighted_features.shape[1], remaining_slots)
        
        if features_to_add > 0:
            all_features[:, feature_idx:feature_idx + features_to_add] = weighted_features[:, :features_to_add]
            feature_idx += features_to_add

    valid_features = all_features[:, :feature_idx]

    if valid_features.shape[1] > num_features:

        feature_vars = np.var(valid_features, axis=0)
        top_indices = np.argpartition(feature_vars, -num_features)[-num_features:]
        final_features = valid_features[:, top_indices]
    else:
        final_features = valid_features

    final_features = normalize(final_features, norm='l2', axis=0)
    
    return final_features


def chebyshev_polynomials(adj_mtx, order=2):

    if not isinstance(adj_mtx, csr_matrix):
        adj_mtx = csr_matrix(adj_mtx)

    n = adj_mtx.shape[0]
    I = identity(n, format='csr')
    

    deg_mtx = np.array(adj_mtx.sum(axis=1)).flatten()
    deg_mtx[deg_mtx == 0] = 1 
    deg_inv_sqrt = np.diagflat(np.power(deg_mtx, -0.5))

    lap_mtx = I - deg_inv_sqrt @ adj_mtx @ deg_inv_sqrt
    lap_mtx = lap_mtx.toarray() if ss.issparse(lap_mtx) else np.asarray(lap_mtx)
    
    matrix_norm = np.linalg.norm(lap_mtx, 'fro')
    if matrix_norm > 1e-10:
        scaling_factor = min(2.0 / matrix_norm, 1.0)
        lap_mtx_scaled = scaling_factor * lap_mtx
    else:
        lap_mtx_scaled = lap_mtx

    effective_order = min(order, max(2, int(np.log2(n) // 2) + 1))

    T_k = [np.eye(n)]  # T_0 = I
    
    if effective_order > 0:
        T_k.append(lap_mtx_scaled)  # T_1 = L_scaled

        for k in range(2, effective_order + 1):
            T_next = 2.0 * lap_mtx_scaled @ T_k[-1] - T_k[-2]
            T_k.append(T_next)
    
    return T_k


def _fast_build_adjacency(signal, cutoff):

    if signal.ndim > 1:
        signal_repr = signal[:, :min(5, signal.shape[1])]
        signal_1d = np.mean(signal_repr, axis=1)
    else:
        signal_1d = signal
    
    n = len(signal_1d)

    row_indices = []
    col_indices = []
    data = []

    window_size = min(8, int(1/cutoff * 3))
    
    for i in range(n):
        for j in range(i+1, min(i + window_size, n)):
            sim = np.exp(-abs(signal_1d[i] - signal_1d[j]) / cutoff)
            if sim > 0.1:  
                row_indices.extend([i, j])
                col_indices.extend([j, i])
                data.extend([sim, sim])
    if len(data) > 0:
        adj_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        return adj_sparse.toarray()
    else:
        adj = np.eye(n)
        for i in range(n-1):
            adj[i, i+1] = adj[i+1, i] = 0.5
        return adj


def _fast_chebyshev_polynomials(adj_matrix, method, cutoff):

    if method == 'lowpass':
        order = max(2, min(4, int(8 * cutoff))) 
    else:  
        order = 3 
    
    return chebyshev_polynomials(adj_matrix, order)


def _fast_lowpass_filter(signal, T_k, cutoff):

    if signal.ndim > 1:
        filtered = np.zeros_like(signal)
        for i, T in enumerate(T_k):
            weight = 1.0 if i / len(T_k) <= cutoff else np.exp(-(i / len(T_k) - cutoff) / cutoff)
            filtered += weight * (T @ signal)
        return filtered / len(T_k)
    else:
        filtered = np.zeros_like(signal)
        for i, T in enumerate(T_k):
            weight = 1.0 if i / len(T_k) <= cutoff else np.exp(-(i / len(T_k) - cutoff) / cutoff)
            filtered += weight * (T @ signal)
        return filtered / len(T_k)


def _fast_wavelet_filter(signal, T_k, level):

    if level is None:
        level = min(3, len(T_k) - 1)  
    
    if signal.ndim > 1:

        coeffs = []
        current_signal = signal.copy()
        
        for i in range(min(level, len(T_k)-1)):
            low_freq = T_k[0] @ current_signal
            high_freq = T_k[i+1] @ current_signal
            coeffs.append(high_freq)
            current_signal = low_freq
        
        coeffs.append(current_signal)

        threshold = np.std(signal) * 0.1 
        for i in range(len(coeffs)-1):
            coeffs[i] = np.where(np.abs(coeffs[i]) > threshold, coeffs[i], 0)

        reconstructed = coeffs[-1]
        for i in range(len(coeffs)-2, -1, -1):
            reconstructed = reconstructed + coeffs[i]
        
        return reconstructed
    else:

        coeffs = []
        current_signal = signal.copy()
        
        for i in range(min(level, len(T_k)-1)):
            low_freq = T_k[0] @ current_signal
            high_freq = T_k[i+1] @ current_signal
            coeffs.append(high_freq)
            current_signal = low_freq
        
        coeffs.append(current_signal)
        
        threshold = np.std(signal) * 0.1
        for i in range(len(coeffs)-1):
            coeffs[i] = np.where(np.abs(coeffs[i]) > threshold, coeffs[i], 0)
        
        reconstructed = coeffs[-1]
        for i in range(len(coeffs)-2, -1, -1):
            reconstructed = reconstructed + coeffs[i]
        
        return reconstructed


class MNiST():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        n_top_genes=3000,
        deconvolution = False,
        datatype = '10X',
        # weight_decay=0.0001,  
        l1_lambda=0.0001,   
        use_frequency_features=True,  
        chebyshev_order=3, 
        num_frequency_features=50  
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning. 
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning. 
            The default is 1.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional    
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data. 
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        if isinstance(random_seed, str) and random_seed.strip().lower() == "random":
            self.random_seed = random.randint(1, 1000)  
        elif isinstance(random_seed, str) and random_seed.isdigit():
            self.random_seed = int(random_seed) 
        else:
            self.random_seed = random_seed 

        print(f"Using random seed: {self.random_seed} (type: {type(self.random_seed)})") 
        self.use_frequency_features = use_frequency_features
        self.chebyshev_order = chebyshev_order
        self.num_frequency_features = num_frequency_features
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        # self.random_seed = random_seed
        self.alpha = alpha
        self.n_top_genes = n_top_genes
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata,n_top_genes=self.n_top_genes)
        
        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:    
              construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs
                 
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)

     
        if use_frequency_features:
            self.integrate_frequency_features()

    def integrate_frequency_features(self):

        print("Integrating frequency features with Chebyshev polynomials...")

      
        adj_mtx = self.adata.obsm['adj']
        chebyshev_mtx_list = chebyshev_polynomials(adj_mtx, order=self.chebyshev_order)

       
        chebyshev_features = obtain_chebyshev_features(
            adata=self.features, 
            chebyshev_mtx_list=chebyshev_mtx_list, 
           
            num_features=50
        )

       
        filtered_chebyshev_features = np.array([
            filter_frequency_signal(freq_signal, method='lowpass', smooth=True, sigma=2) 
            for freq_signal in chebyshev_features.T
        ]).T

       
        chebyshev_features_tensor = torch.tensor(filtered_chebyshev_features, dtype=torch.float32).to(self.device)
        combined_features = torch.cat((self.features, chebyshev_features_tensor), dim=1)

    
        print("Performing PCA to reduce dimensions...")
        combined_features_np = combined_features.cpu().detach().numpy()
        n_components = self.features.shape[1]  
        print(f"Reducing to {n_components} dimensions...")
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(combined_features_np)

        self.features = torch.tensor(reduced_features, dtype=torch.float32).to(self.device)
        print(f"Features after PCA reduction: {self.features.shape}")


    def train(self):

        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input,
                                        self.dim_output,
                                        self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input,
                                self.dim_output,
                                self.graph_neigh).to(self.device)

        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        print("Begin to train ST data...")
        self.model.train()

    
        loss_total_hist, loss_feat_hist, loss_sl1_hist, loss_sl2_hist = [], [], [], []

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            self.features_a = permutation(self.features)  
            self.hiden_feat, self.emb, ret, ret_a = self.model(
                self.features, self.features_a, self.adj)

            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)

            loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_total_hist.append(loss.item())
            loss_feat_hist.append(self.loss_feat.item())
            loss_sl1_hist.append(self.loss_sl_1.item())
            loss_sl2_hist.append(self.loss_sl_2.item())

        print("Optimization finished for ST data!")

   
        # plt.figure(figsize=(6, 4))
        # plt.plot(loss_total_hist, label="total")
        # plt.plot(loss_feat_hist, label="feat")
        # plt.plot(loss_sl1_hist, label="sl1")
        # plt.plot(loss_sl2_hist, label="sl2")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("Training Loss Curve")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()  # 只展示，不保存

   
        with torch.no_grad():
            self.model.eval()
            if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                return self.emb_rec
            else:
                if self.datatype in ['Stereo', 'Slide']:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                    self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
                else:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()

                self.adata.obsm['emb'] = self.emb_rec
                return self.adata

    def train_sc(self):
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        
        print('Begin to train scRNA data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            
            emb = self.model_sc(self.feat_sc)
            loss = F.mse_loss(emb, self.feat_sc)
            
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
            
        print("Optimization finished for cell representation learning!")
        
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
         
            return emb_sc
        
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        
        # Normalize features for consistence between ST and scRNA-seq
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
          
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()

            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
             
            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE 

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
            
        print("Mapping matrix learning finished!")
        
 
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell
            
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell

            return self.adata, self.adata_sc         

    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M        
