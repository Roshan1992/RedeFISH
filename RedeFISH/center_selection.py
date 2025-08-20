# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:57:01 2023

@author: 32792
"""

import torch
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import random
from sklearn.cluster import DBSCAN, MeanShift,  AgglomerativeClustering
from sklearn import neighbors
import scanpy as sc
from scipy.stats import pearsonr


from .KMean_KD import KMEANS



def generate_grid(coordinate, scale):
    
    max_x = np.max(coordinate[:,0])
    min_x = np.min(coordinate[:,0])
    max_y = np.max(coordinate[:,1])
    min_y = np.min(coordinate[:,1])
    delta_x = max_x - min_x
    delta_y = max_y - min_y

    count_x = max(int(delta_x / scale),1)
    count_y = max(int(delta_y / scale),1)
    shift_x = min(delta_x, scale) / 2
    shift_y = min(delta_y, scale) / 2
    all_x = []
    all_y = []
    for m in np.linspace(min_x + shift_x, max_x - shift_x, count_x):
        for n in np.linspace(min_y + shift_y, max_y - shift_y, count_y):
            all_x.append(m)
            all_y.append(n)
            
    grid_coordinate = np.asarray([all_x, all_y]).T
            
    return(grid_coordinate)



class center_selection():
    
    def setup_seed(self, seed):
        
        """
        Setup random seed
        """
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def __init__(self, 
                 scale, 
                 st_data,
                 sc_data,
                 number_gene_for_cluster = 3,
                 number_cluster_for_genegroup = 10, 
                 number_min_genes = 10,
                 ):
        
        
        print("Initializing ...")
        self.setup_seed(0)
        self.scale = scale
        self.number_gene_for_cluster = number_gene_for_cluster
        self.number_cluster_for_genegroup = number_cluster_for_genegroup
        
        
        gene1 = np.unique(st_data['gene'])
        gene2 = sc_data.var_names
        
        inter_gene = np.intersect1d(gene1, gene2)
        
        sc_data = sc_data[:,inter_gene] 
        sc.pp.filter_cells(sc_data, min_genes=number_min_genes)
        sc.pp.normalize_per_cell(sc_data, len(sc_data.var_names))
        sc_data.obs['annotation'] = [str(x) for x in sc_data.obs['annotation']]
        sc.tl.rank_genes_groups(sc_data,groupby='annotation',use_raw=False)
        
        self.sc_data = sc_data
        self.st_data = st_data
        
        
        
    def predict_cell_location(self):
        
        
        print("Choosing Gene ...")
        self.all_gene, self.mean_expression = self.gene_selection()
        
        print("Grouping Gene ...")
        result, all_cutoff, mRNA_count, gene_group = self.gene_grouping()

        
        print("Begin cell location prediction ...")
        coordinate = np.asarray(self.st_data[['x', 'y']])
        grid_coordinate = generate_grid(coordinate, 2 * self.scale)
        
        self.all_center = []
        for i in range(len(gene_group)):
            data = self.st_data[self.st_data['gene'].isin(gene_group[i])]
            coordinate = np.asarray([data['x'], data['y']]).T
            center_this = self.process_center_prediction(i, coordinate, grid_coordinate, all_cutoff, gene_group[i])
            self.all_center.extend(center_this)
            
        
        self.post_process()
        
        return(self.cluster_centers)
            
           
    def post_process(self):
        
        self.all_center = np.asarray(self.all_center)
        self.all_center = self.all_center[np.sum(self.all_center, axis=1) > 1e-6, :]
            
        df = pd.DataFrame()
        df['x'] = np.asarray(self.all_center[:,0])   
        df['y'] = np.asarray(self.all_center[:,1]) 
        #df['tag'] = all_gene
        
        ms = MeanShift(bandwidth=self.scale/5)
        ms.fit(self.all_center)
        self.cluster_centers = ms.cluster_centers_
        
            
    
    def process_center_prediction(self, i, coordinate, grid_coordinate, all_cutoff, gene):
        
        
        tree = neighbors.KDTree(coordinate)
        number = tree.query_radius(grid_coordinate, np.sqrt(np.square(2 * self.scale) / np.pi), count_only=True)  
    
        cutoff = np.mean(number[number > -np.sort(-number)[int(len(grid_coordinate) * 0.1)] * 0.01])
        
        dbscan = DBSCAN(eps=2 * self.scale, min_samples = max(2,int(cutoff)), algorithm='kd_tree')
       
    
        labels = dbscan.fit_predict(coordinate)
        
        max_label = np.max(labels)
        
        relabel = np.zeros(coordinate.shape[0]) - 2
        relabel[labels == -1] = -1
        
        
        #relabel = labels
        all_center = []
        
        for j in range(max_label + 1):
            
            index = labels == j
            
            coordinate_this = coordinate[index, :]
            sub_grid_coordinate = generate_grid(coordinate_this, 2 * self.scale)
    
            tree = neighbors.KDTree(coordinate_this)   
            number = tree.query_radius(sub_grid_coordinate, np.sqrt(np.square(2 * self.scale) / np.pi), count_only=True)     
            
            #k_number = int(len(number[number > cutoff]) / 2 / np.sqrt(2) * (np.log(np.max(all_cutoff) / all_cutoff[i]) + 1))
            k_number = int(len(number[number > cutoff]) / 4 * (np.log(np.max(all_cutoff) / all_cutoff[i]) + 1))
            if k_number > len(sub_grid_coordinate):
                k_number = len(sub_grid_coordinate)
            if k_number == 0:
                k_number = 1 
            
            
            KM = KMEANS(max_iter=10, n_clusters=k_number, verbose=False)
            KM.fit(torch.from_numpy(coordinate_this).float())
            KM_label = KM.labels
            relabel[index] = np.max(relabel) + 1 + KM_label.cpu().numpy()
            all_center.extend(list(KM.centers.cpu().numpy()))
            

            
        print("Finish cluster " + str(i) + ":", int(np.max(relabel)), len(coordinate), cutoff)     
        print("\tRelated gene:", "|".join(gene))
        
        '''
        color = []
        for each in relabel:
            if each == -1:
                color.append('black')
            elif each % 10 == 0:
                color.append('red')
            elif each % 10 == 1:
                color.append('orange')
            elif each % 10 == 2:
                color.append('yellow')
            elif each % 10 == 3:
                color.append('green')
            elif each % 10 == 4:
                color.append('cyan')
            elif each % 10 == 5:
                color.append('blue')
            elif each % 10 == 6:
                color.append('purple')
            elif each % 10 == 7:
                color.append('pink')
            elif each % 10 == 8:
                color.append('brown')
            elif each % 10 == 9:
                color.append('grey')
        '''
        #import matplotlib.pylab as plt
        #plt.figure(figsize=(18,20))
        #plt.scatter(coordinate[:,0], coordinate[:,1], c=color, s=0.1)
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.savefig(str(i) + ".jpg", dpi=300, bbox_inches='tight')
       
        
        return(all_center)


     
        
        
        
        
    def gene_selection(self):
        
        markers = pd.DataFrame(self.sc_data.uns['rank_genes_groups']['names'])

        df = self.sc_data.to_df()
        df['anno'] = self.sc_data.obs['annotation']
        df2 = df.groupby('anno').mean()

       
        percentage = np.asarray(df2) / np.sum(np.asarray(df2) + 1e-6, axis=0)
        percentage = pd.DataFrame(percentage)
        percentage.columns = df2.columns
        percentage.index = df2.index

        score = np.log(df2 + 1) * percentage
        
        all_gene = []
        for i in range(len(markers.columns)):
            tag = markers.columns[i]
            gene = list(df2.columns[np.argsort(-np.asarray(score.T[tag]))[0:self.number_gene_for_cluster]])
            #gene = list(markers[markers.columns[i]][0:3])
            all_gene.extend(gene)

        all_gene = list(set(all_gene))
        all_gene.sort()

        
        return(all_gene, df2)
    
    
    def gene_grouping(self):
        
        coordinate = np.asarray(self.st_data[['x', 'y']])
        grid_coordinate = generate_grid(coordinate, 2 * self.scale)
        count = np.zeros([len(self.all_gene), len(grid_coordinate)])
        
        for i in range(len(self.all_gene)):
            temp = self.st_data.loc[self.st_data['gene'] == self.all_gene[i]]
            temp = np.asarray(temp[['x', 'y']])
            tree = neighbors.KDTree(temp)
            count[i,:] = tree.query_radius(grid_coordinate, np.sqrt(4 * self.scale * self.scale / np.pi), count_only=True)
        
        
        count = count / np.max(count, axis=1)[:,None]
        mean_count = np.mean(count, axis=1)
        std_count = np.std(count, axis=1)
        
        count = (count - mean_count[:,None]) / (std_count[:,None] + 1e-8)
        count += np.random.randn(count.shape[0], count.shape[1]) * 1e-6
        
        similarity = np.zeros([len(self.all_gene), len(self.all_gene)])      
        for i in range(len(self.all_gene)):
            for j in range(i,len(self.all_gene)):
                A = count[i,:]
                B = count[j,:]
                similarity[i,j], _ = pearsonr(A, B)
                similarity[j,i] = similarity[i,j]
        
        model=AgglomerativeClustering(n_clusters = self.number_cluster_for_genegroup, distance_threshold = None, compute_distances=True)
        
        result = model.fit_predict(similarity)
        result = result + 1
        
        
        df = self.sc_data.to_df()
        df['anno'] = self.sc_data.obs['annotation']
        df2 = df.groupby('anno').mean()
        
        all_cutoff = []
        mRNA_count = []
        gene_group = []
        for i in range(np.max(result)):
            gene = list(np.asarray(self.all_gene)[result == (i+1)])
            temp = np.asarray(df2[gene])
            all_cutoff.append(np.sum(np.max(temp, axis=0)))
            mRNA_count.append(np.sum(self.st_data['gene'].isin(gene)))
            gene_group.append(gene)
            
        index = np.argsort(-np.asarray(mRNA_count))    
        all_cutoff = np.asarray(all_cutoff)[index]    
        mRNA_count = np.asarray(mRNA_count)[index]
        gene_group = [gene_group[x] for x in index]
        
        return(result, all_cutoff, mRNA_count, gene_group)














    


    
    
    
    



