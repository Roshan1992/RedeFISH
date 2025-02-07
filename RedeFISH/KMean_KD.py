# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:52:24 2023

@author: 32792
"""
import torch
from scipy.spatial import KDTree



class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=True):

        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")])
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
    

    def fit(self, x):
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,))
        init_points = x[init_row]
        self.centers = init_points
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        #self.representative_sample()
        

    def nearest_center(self, x):
        
        tree = KDTree(self.centers)
        self.labels = torch.tensor(tree.query(x, k=1)[1])
        
        #dists = torch.cdist(x, self.centers)
        #self.labels = torch.argmin(dists, dim=1)
    
        #if self.started:
        #    self.variation = torch.sum(self.dists - dists)
        #self.dists = dists
        #self.started = True
        
        

    def update_center(self, x):
        
        centers = torch.zeros(self.centers.shape)
               
        center_index = torch.stack([self.labels, torch.zeros(len(self.labels))], dim=0)
        x_sparse = torch.sparse_coo_tensor(center_index, x[:,0], (self.n_clusters, 1))
        y_sparse = torch.sparse_coo_tensor(center_index, x[:,1], (self.n_clusters, 1))
        count = torch.sparse_coo_tensor(center_index, torch.ones(len(self.labels)), (self.n_clusters, 1))
        count = count.to_dense().reshape(-1)
        
        centers[:,0] = x_sparse.to_dense().reshape(-1) / (count + 1e-6)
        centers[:,1] = y_sparse.to_dense().reshape(-1) / (count + 1e-6)
        
        self.centers = centers
        

         





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    