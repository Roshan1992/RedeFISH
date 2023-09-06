# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------
#
# File: RedeFISH.py
#
# System:         Linux, Windows
# Component Name: RedeFISH
# Version:        20230906
# Language: python3
# Latest Revision Time: 2023/09/06
#
# License: To-be-decided
# Licensed Material - Property of CPNL.
#
# (c) Copyright CPNL. 2023
#
# Address:
# 28#, ZGC Science and Technology Park, Changping District, Beijing, China
#
# Author: Zhong Yunshan
# E-Mail: 327922729@qq.com
#
# Description: Main function of RedeFISH
#
# Change History:
# Date         Author            Description
# 2023/09/06   Zhong Yunshan     Release v1.0.0
# ------------------------------------------------------------------------------------


import os
import time
import pandas as pd
import numpy as np
from scipy import spatial
import scanpy as sc
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import lr_scheduler
import torch.nn.functional as F
import random
import warnings
import alphashape
import torch.multiprocessing
import shapely
from shapely.geometry import Polygon


warnings.filterwarnings("ignore")


def calculate_cell_feature_each(each, max_value, mean_x, mean_y, cell_name):
    
    """
    Calculate multiple cell features for each aligned cell.

    Parameters
    ----------
    each : Shapely Polygon, required
        Shapely Polygon of aligned cells.
        
    max_value : float, required
        Maximun of transcripts coordinate

    mean_x : float, required
        mean x axis of transcripts for each aligned cell

    mean_y : float, required
        mean y axis of transcripts for each aligned cell
        
    cell_name : str, required
        cell_name

    Returns
    -------
    result : dict()
        Dictionary of cell features for each aligned cell.
    """
    
    cell_feature_this = dict()
    center_this = np.asarray(each.centroid)
    
    boundary = np.asarray(each.boundary)
    boundary = boundary * max_value
    boundary[:,0] = boundary[:,0] + mean_x
    boundary[:,1] = boundary[:,1] + mean_y
    center_x = center_this[0] * max_value + mean_x
    center_y = center_this[1] * max_value + mean_y
    
    area = each.area * max_value * max_value 
    girth = np.sqrt(np.square(boundary[1:,0] - boundary[0:-1,0]) + np.square(boundary[1:,1] - boundary[0:-1,1]))
    girth = np.sum(girth) 
    
    roundness1 = 4 * np.pi * area / np.power(girth,2)
    # Output mRNA class
    x = boundary[0:-1,0]
    y = boundary[0:-1,1]
    distance = np.mean(np.sqrt(np.square(x - center_x) + np.square(y - center_y)))
    sigma = np.sum(np.square(np.sqrt(np.square(x - center_x) + np.square(y - center_y)) - distance)) / len(x)
    roundness2 = 1 - np.sqrt(sigma) / distance
    
    boundary = np.round(boundary, 4)
    #boundary = np.round(boundary, 4).tolist()
    
    cell_feature_this['center_x'] = center_x
    cell_feature_this['center_y'] = center_y
    cell_feature_this['boundary'] = boundary
    cell_feature_this['girth'] = girth
    cell_feature_this['area'] = area
    cell_feature_this['roundness'] = roundness1
    #cell_feature_this['roundness2'] = roundness2
    cell_feature_this['index'] = cell_name
    
    return(cell_feature_this)



def calculate_cell_feature(sub_data, cell_id, cell_count_cutoff, alphashape_value):
    
    """
    Apply alphashape for post-processing of aligned cells. Then calculated multiple
    features for aligned cells.

    Parameters
    ----------
    sub_data : Pandas DataFrame, required
        Aligned cells related transcript coordinates.
        
    cell_id : str, required
        cell_name

    cell_count_cutoff : int, required
        cutoff of transcripts number of aligned cells

    alphashape_value : float, required
        alpha value for alphashape program
        

    Returns
    -------
    result : dict()
        Dictionary of cell features for aligned cells.
    """
    
    if len(sub_data) < cell_count_cutoff:
        return([],[],False)
    
    mean_x = np.mean(sub_data['x'])
    mean_y = np.mean(sub_data['y'])
    x = np.asarray(sub_data['x'])
    y = np.asarray(sub_data['y'])
    
    x = x - mean_x
    y = y - mean_y
    
    
    if np.sum(np.abs(x)) < 1e-5 or np.sum(np.abs(y)) < 1e-5:
        x = x + np.random.randn(len(x)) * 1e-10
        y = y + np.random.randn(len(y)) * 1e-10
    
    
    max_x = np.max(np.abs(x))
    max_y = np.max(np.abs(y))
    max_value = max(max_x, max_y)
    
    x_norm = x / max_value
    y_norm = y / max_value
    
    points  = np.asarray([x_norm,y_norm], dtype=np.float32).T

    alpha_shape = alphashape.alphashape(points, alphashape_value)
    
    cell_feature = []
    
    index = 1
    if alpha_shape.type == 'MultiPolygon':
        
        for each in alpha_shape:
            cell_name = str(cell_id) + "_" + str(index)
            cell_feature_this = calculate_cell_feature_each(each, max_value, mean_x, mean_y, cell_name)
            cell_feature.append(cell_feature_this)
            index += 1

    elif alpha_shape.type == 'Polygon':
        cell_name = str(cell_id) + "_" + str(index)
        cell_feature_this = calculate_cell_feature_each(alpha_shape, max_value, mean_x, mean_y, cell_name)
        cell_feature.append(cell_feature_this)
    else:
        return([],[],False)
            
    polygon_list = []
    for i in range(len(cell_feature)):
        polygon_list.append(Polygon(cell_feature[i]['boundary']))
    
        
    result_class = []
    for i in range(len(x)):
        is_class1 = False
        for j in range(len(cell_feature)):
            is_class2 = False
            boundary_this = cell_feature[j]['boundary']
            for k in range(len(boundary_this)):
                x_new = 0.999999 * (x[i] + mean_x) + 0.000001 * boundary_this[k,0]
                y_new = 0.999999 * (y[i] + mean_y) + 0.000001 * boundary_this[k,1]
                bool_this = polygon_list[j].contains(shapely.geometry.Point(x_new, y_new))
                if bool_this:
                    result_class.append(str(cell_id) + '_' + str(j+1))
                    is_class1 = True
                    is_class2 = True
                    break
            if is_class2:
                break
        if not is_class1:
            result_class.append('0')
            
    result_dict = dict()
    for each in result_class:
        if not each in result_dict:
            result_dict[each] = 0
        result_dict[each] += 1
        
    for i in range(len(cell_feature)):
        if cell_feature[i]['index'] in result_dict:
            cell_feature[i]['mRNA_count'] = result_dict[cell_feature[i]['index']]
        else:
            cell_feature[i]['mRNA_count'] = 0
        if cell_feature[i]['mRNA_count'] >= cell_count_cutoff:
            cell_feature[i]['is_true_cell'] = True
        else:
            cell_feature[i]['is_true_cell'] = False
    
    sub_data['result_class'] = result_class
    
    return(cell_feature, sub_data, True)

    



class center_selection(nn.Module):
    
    
    def setup_seed(self, seed):
        
        """
        Setup random seed
        """
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        
    def calculate_neibor(self):
        
        
        """
        Calculate nearest neibor for transcripts
        """
        
        
        self.index = np.random.choice(a=self.mRNA_number, size=int(self.mRNA_number / self.subsample), replace=False)
        self.xy_index_use = self.xy_index[self.index,:]
        tree = spatial.KDTree(data=self.xy_index_use)
        self.nearest_neibor = tree.query(self.xy_index_use ,k=self.k_number+1,workers=self.worker_number)
        self.sub_mRNA_number = len(self.xy_index_use)
        self.sample_neighbor = torch.from_numpy(self.nearest_neibor[1])
        self.nearest_neibor = self.nearest_neibor[0]
        self.sample_neighbor = self.sample_neighbor[:,1:]
        self.nearest_neibor = self.nearest_neibor[:,1:]
        
        
    def calculate_distance_score(self):
        
        """
        Calculate distance score for transcripts
        """
        '''
        cutoff = int(self.k_number / 5)
        distance1 = np.mean(self.nearest_neibor[:,0:cutoff], axis=1)
        distance2 = np.mean(self.nearest_neibor[:,0:2*cutoff], axis=1)
        distance3 = np.mean(self.nearest_neibor[:,0:3*cutoff], axis=1)
        distance4 = np.mean(self.nearest_neibor[:,0:4*cutoff], axis=1)
        distance5 = np.mean(self.nearest_neibor, axis=1)
        
        distance_score1 = np.mean(distance1) / (distance1 + 1e-4)
        distance_score2 = np.mean(distance2) / (distance2 + 1e-4)
        distance_score3 = np.mean(distance3) / (distance3 + 1e-4)
        distance_score4 = np.mean(distance4) / (distance4 + 1e-4)
        distance_score5 = np.mean(distance5) / (distance5 + 1e-4)
        
        distance_score = (distance_score1 + distance_score2 + distance_score3 + distance_score4 + distance_score5) / 5
        distance_score = distance_score / np.mean(distance_score)
        self.distance_score = torch.from_numpy(distance_score).type(torch.float32)
        
        
        '''
        distance = torch.from_numpy(np.mean(self.nearest_neibor, axis=1)).type(torch.float32)
        cutoff = torch.sort(distance, descending=True).values[[int(len(distance) * self.distance_percentage_cutoff)]]
        #distance_score = torch.round(self.distance2 / (cutoff / 1000)) / 1000
        distance_score = distance / cutoff
        distance_score[distance_score >= 1] = 1
        distance_score = 1 - distance_score
        self.distance_score = distance_score.type(torch.float32).to(self.device)
        
        
        distance = torch.from_numpy(np.mean(self.nearest_neibor, axis=1)).type(torch.float32)
        if not self.scale:
            self.scale = torch.sort(distance, descending=True).values[[int(len(distance) * self.distance_percentage_cutoff)]]
        self.sample_neighbor_weight = self.nearest_neibor / self.scale
        self.sample_neighbor_weight[self.sample_neighbor_weight >= 1] = 1
        self.sample_neighbor_weight = 1 - self.sample_neighbor_weight
        
        self.weight_ratio = torch.sum(self.sample_neighbor_weight) / torch.sum(self.sample_neighbor_weight > -1)

        
        
    
    def __init__(self, 
                 xy_index, 
                 prior_count=100, 
                 subsample=1, 
                 worker_number=1, 
                 quantile_cutoff = 0.999,
                 scale=None,
                 device='cpu'):
        
        """
        Initialing candidate cell center selection model.
    
        Parameters
        ----------
        xy_index : Numpy Array, required
            The coordinate of transcripts.
            
        prior_count : int, default=100
            Prior estimate of number of transcripts in each cells. No need to be precise
    
        subsample : int, default=1
            Perform candidate center selection on Randomly select number_of_transcripts/subsample  
            transcripts. This can be used to This can avoid excessive use of CPU or GPU memory.
    
        worker_number : int, default=1
            Number of CPU cores.
            
        quantile_cutoff : float, default=0.999
            Cutoff of the quantile function for distance distribution. Used for calculating 
            distance score
            
        scale: float, default=None
            Prior radius of cells.
            
        device : str, default='cpu'
            Set device for running center selection.

        """
        
        
        super(center_selection, self).__init__()
        
        print("\nEstimating Candidate Cell Locations ...\n")
        
        
        #xy_index = coordinate_normalization(xy_index)
        
        self.setup_seed(0)
        self.device = device
        self.mRNA_number = len(xy_index)
        self.xy_index = xy_index
        self.distance_percentage_cutoff = 1 - quantile_cutoff
        self.scale = scale

        self.distance_percentage_cutoff = min(max(self.distance_percentage_cutoff,0),1)
        
        self.worker_number = worker_number
        
        if int(prior_count / subsample) < 50:
            self.k_number = 50
            self.subsample = int(prior_count / 50) + 1
        else:
            self.k_number = int(prior_count / subsample) 
            self.subsample = subsample
        
        
        print("  Initialing data ...")
        self.calculate_neibor()
        self.calculate_distance_score()
        
        self.mRNA_prob = nn.Parameter(torch.cat((torch.ones(self.sub_mRNA_number, 1) * (0), torch.ones(self.sub_mRNA_number, 1) * (-3)), dim=1))                 
        
        
        lr_this = max(self.sub_mRNA_number / 10000000 / 100, 0.01)
        #print("Learning Rate:", lr_this)
        self.optimizer = torch.optim.Adam([{'params': self.parameters(), 'initial_lr': lr_this}], lr=lr_this)     
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, 0.996, 1)




    def train(self, 
              epochs=1000):
        
        """
        Training the candidate cell center selection model.
        
        Parameters
        ----------
        epochs : int, default=1000
            Number of epochs for the training.
            
        Returns
        -------
        result : ndarray
            A vector refer to index of selected cells.
        """
        
        loss_all = 0
        number = 0
        batch_size = int(self.mRNA_number / self.subsample / 100) + 1 
        self.cell_choose = torch.zeros(self.sub_mRNA_number).type(torch.long) 
        
        for i in range(epochs):
           
            shuffle = torch.randperm(self.sub_mRNA_number)
            self.coverage = torch.zeros(self.sub_mRNA_number).to(self.device)
            for j in range(0, self.sub_mRNA_number, batch_size):
     
                upper = min(self.sub_mRNA_number, j + batch_size)
                lower = j
                
                shuffle_this = shuffle[lower:upper]
                distance_score_this = self.distance_score[shuffle_this]
                sample_neighbor_this = self.sample_neighbor[shuffle_this,:]
                sample_neighbor_weight_this = self.sample_neighbor_weight[shuffle_this,:]
                
                reward = self.forward(shuffle_this, distance_score_this, sample_neighbor_this, sample_neighbor_weight_this, True)
                
                
                
                log_prob = (self.dist.log_prob(self.action))
                #print(reward)
                #print(log_prob.T)
                loss = - (reward * log_prob.T).mean()
                loss = loss.requires_grad_()
                
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                loss_all += loss.cpu().detach().numpy()
                number += 1
                
                self.cell_choose[shuffle_this] = self.action.cpu()
                
            self.scheduler.step() 
            if i % 20 == 0:
                print("  Epoch:", i, "Loss:", loss_all / number)
                loss_all = 0
                number = 0
                #print("Coverage:",len(self.coverage.nonzero())/self.sub_mRNA_number)
                #print("Number of cell center:", torch.sum(self.cell_choose).numpy())
                torch.cuda.empty_cache()
                
                
                
        self.cell_choose = torch.zeros(self.sub_mRNA_number).type(torch.long) 
        shuffle = torch.tensor(range(self.sub_mRNA_number))
        for j in range(0, self.sub_mRNA_number, batch_size):
        
            upper = min(self.sub_mRNA_number, j + batch_size)
            lower = j
            
            shuffle_this = shuffle[lower:upper]
            distance_score_this = self.distance_score[shuffle_this]
            sample_neighbor_this = self.sample_neighbor[shuffle_this,:]
            sample_neighbor_weight_this = self.sample_neighbor[shuffle_this,:]
            
            reward = self.forward(shuffle_this, distance_score_this, sample_neighbor_this, sample_neighbor_weight_this, False)
            
            log_prob = (self.dist.log_prob(self.action))
         
            self.cell_choose[shuffle_this] = self.action.cpu()
            
            
        print("")
        print("  Number of Candidate Cell Locations:", torch.sum(self.cell_choose).numpy())
        #print("Coverage:",len(self.coverage.nonzero()) / self.sub_mRNA_number)
        
        final_choose_cell = np.sort(self.cell_choose.nonzero().reshape(-1).cpu().numpy())
            
        return(self.xy_index[self.index[final_choose_cell],:])
    



    def forward(self, shuffle_this, distance_score_this, sample_neighbor_this, sample_neighbor_weight_this, is_train):
        
       
        shuffle_this = shuffle_this.to(self.device)
        distance_score_this = distance_score_this.to(self.device)
        sample_neighbor_this = sample_neighbor_this.to(self.device)
        sample_neighbor_weight_this = sample_neighbor_weight_this.to(self.device)
        
        self.cell_feature = self.mRNA_prob[shuffle_this,:]
       
        self.dist = Categorical(logits=self.cell_feature)
        
        if is_train:
            self.action = self.dist.sample()
        else:
            self.action = torch.max(self.cell_feature, dim=1).indices

        
        self.coverage_this = torch.index_select(self.coverage, 0, sample_neighbor_this.reshape(-1)).reshape(-1,self.k_number)
        self.coverage_this_sum = torch.sum(self.coverage_this, dim=1)
        #self.coverage_score = self.k_number / (1 / self.weight_ratio * self.coverage_this_sum + self.k_number)
        self.coverage_score = self.k_number / (2 * self.coverage_this_sum + self.k_number)
        #self.coverage_score = self.k_number * self.weight_ratio / (self.coverage_this_sum + self.k_number)
        self.coverage_score[self.action == 0] = 1 - self.coverage_score[self.action == 0]
        #distance_score_this[self.action == 1] = 1 / distance_score_this[self.action == 1]
        
        
        use_ful = sample_neighbor_this.reshape(-1)[self.action.reshape(-1,1).repeat(1, self.k_number).reshape(-1) == 1]
        use_ful_weight = sample_neighbor_weight_this.reshape(-1)[self.action.reshape(-1,1).repeat(1, self.k_number).reshape(-1) == 1]
        temp = torch.bincount(use_ful, weights=use_ful_weight)
        #temp = torch.bincount(use_ful)
        mRNA_index = temp.nonzero().reshape(-1)
        mRNA_value = temp[temp > 0]
        self.coverage[mRNA_index] += mRNA_value
        self.coverage[shuffle_this[self.action == 1]] += 1
    
        self.reward = self.coverage_score * distance_score_this
        
        
        return(self.reward)
    
    
    




class alignment(nn.Module):
    
    
    def setup_seed(self, seed):
        
        """
        Setup random seed
        """
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    
    def coordinate_normalization(self, coordinate, candidate_cell_center):
        
        """
        Co-normalize coordinate of transcripts and candidate cell center
        """
        
        # Transcripts
        max_x = np.max(coordinate[:,0])
        max_y = np.max(coordinate[:,1])
        min_x = np.min(coordinate[:,0])
        min_y = np.min(coordinate[:,1])
        delta_x = max_x - min_x
        delta_y = max_y - min_y

        up_x = delta_x / (delta_x + delta_y) * 20000
        up_y = delta_y / (delta_x + delta_y) * 20000

        x = (coordinate[:,0] - min_x) / (max_x - min_x) * up_x
        y = (coordinate[:,1] - min_y) / (max_y - min_y) * up_y
        center_x = (candidate_cell_center[:,0] - min_x) / (max_x - min_x) * up_x
        center_y = (candidate_cell_center[:,1] - min_y) / (max_y - min_y) * up_y
        
        coordinate_normal = np.asarray([x, y]).T
        coordinate_cell_center = np.asarray([center_x, center_y]).T
        
        
        xy_index = torch.from_numpy(coordinate_normal).type(torch.float32)
        xy_mean = torch.mean(xy_index, dim=0)
        xy_std = torch.std(xy_index, dim=0)
        self.xy_index = (xy_index - xy_mean) / xy_std

        return(coordinate_normal, coordinate_cell_center)
    
    
    
    
    def build_graph(self, coordinate_normal, coordinate_cell_center):
        
        """
        Build KNN-Graph for downstream cell alignment.
        """
        
        tree = spatial.KDTree(data=coordinate_normal)

        nearest_neibor = tree.query(coordinate_normal,k=self.k_number_for_distance+1,workers=self.worker_number)
        distance = torch.tensor(nearest_neibor[0][:,1:] / np.mean(nearest_neibor[0][:,1:])).reshape(-1)
       
        
        distance2 = torch.mean(distance.reshape((-1,self.k_number_for_distance)), dim=1)
        
        mean_value = torch.mean(distance2)
        std_value = torch.std(distance2)
        N = torch.distributions.Normal(mean_value, std_value)
        log_P1 = N.log_prob(distance2)
        
        self.distance2 = distance2  
        cutoff = torch.sort(self.distance2, descending=True).values[[int(len(self.distance2) * self.distance_percentage_cutoff)]]
        #distance_score = torch.round(self.distance2 / (cutoff / 1000)) / 1000
        distance_score = self.distance2 / cutoff
        distance_score[distance_score >= 1] = 1
        #distance_score = torch.log(self.distance2 + 1)
        #distance_score = distance_score / torch.max(distance_score)
        distance_score = 1 - distance_score
        #distance_score  = torch.exp(log_P1)
        #distance_score = 1 - torch.tanh(self.distance2)
        #distance_score[distance2 < 1] = 2 * torch.max(distance_score) - distance_score[distance2 < 1]
        #distance_score = (distance_score - torch.min(distance_score)) / (torch.max(distance_score) - torch.min(distance_score))
        #distance_score[distance2 < 1] = 1
        #distance_score = (distance_score - torch.mean(distance_score)) / torch.std(distance_score)
        self.distance_score = distance_score.type(torch.float32).to(self.device)
        
        
        tree = spatial.KDTree(data=coordinate_cell_center)
        nearest_neibor = tree.query(coordinate_normal,k=self.k_number,workers=self.worker_number)
        self.candidate = torch.from_numpy(nearest_neibor[1])
        
        
    def prepare_sc_data(self, sc_data):
        
        """
        Manage single cell reference. Filter low quality single cells.
        """
    
        gene = list(set(self.gene))
        gene.sort()
        sc_gene = sc_data.var_names
        self.inter_gene = np.intersect1d(sc_gene, gene)
        self.sc_data_all_gene = sc_data
        sc_data = sc_data[:,self.inter_gene]
        sc.pp.filter_cells(sc_data, min_genes=self.number_min_genes)
        self.sc_data_all_gene = self.sc_data_all_gene[sc_data.obs_names, :]
        
        
        '''
        anno = sc_data.obs['annotation']
        anno_class = list(set(sc_data.obs['annotation']))
        single_cell_id = []
        for each in anno_class:
            temp = anno[anno == each]
            print(len(temp))
            if len(temp) <= 1000:
                single_cell_id.extend(temp.index)
            else:
                single_cell_id.extend(temp.index[torch.randperm(len(temp)).numpy()[0:1000]])
        self.sc_data = sc_data[single_cell_id,:]
        self.sc_expression = torch.from_numpy(np.asarray(sc_data.to_df()))
        #self.sc_data_all_gene = self.sc_data_all_gene[sc_data.obs_names, :]
        ''' 
        
        self.sc_data = sc_data
        self.sc_expression = torch.from_numpy(np.asarray(sc_data.to_df()))
        
        
        
        self.annotation = self.sc_data.obs['annotation']
        self.annotation_ori = sc_data.obs['annotation']
        print("\nCell type distribution of sc/snRNA:")
        print(self.annotation.value_counts())
        
        print("Intersection Gene:", len(self.inter_gene))
        
        
        
        gene2index = dict()
        gene2index_ori = dict()
        gene2index['UNK'] = 0
        index1 = 1
        index2 = 0
        for each in self.inter_gene:
            gene2index[each] = int(index1)
            index1 += 1
        for each in gene:
            gene2index_ori[each] = int(index2)
            index2 += 1
           
        self.gene_index = torch.from_numpy(np.asarray(pd.Series(self.gene).map(gene2index).fillna(0))).type(torch.long)
        self.gene_index_ori = torch.from_numpy(np.asarray(pd.Series(self.gene).map(gene2index_ori).fillna(0))).type(torch.long)
        self.expression_dim = torch.max(self.gene_index) + 1
        
        
        
        self.sc_data_dict = dict()
        for i in range(len(self.annotation)):
            if not self.annotation[i] in self.sc_data_dict:
                self.sc_data_dict[self.annotation[i]] = []
            self.sc_data_dict[self.annotation[i]].append(self.sc_expression[i,:])

        for each in self.sc_data_dict:
            self.sc_data_dict[each] = torch.stack(self.sc_data_dict[each])
            
        self.number_cell_type = len(self.sc_data_dict)
        
        
    
    def __init__(self, 
                 coordinate, 
                 gene,
                 candidate_cell_center, 
                 sc_data,
                 prior_count = 100,
                 number_min_genes = 10,
                 number_cell_state = 10,
                 cell_count_cutoff = None,
                 k_number = 8, 
                 k_number_for_distance = 100,
                 quantile_cutoff = 0.999,
                 alphashape_value = 5,
                 top_N_single_cell = 20,
                 worker_number = 1,
                 output_dir = "./",
                 device='cpu'):
        
        """
        Initailing RedeFISH alignment model.

        Parameters
        ----------
        coordinate : Numpy Array, required
            The coordinate of transcripts. dimention: [Transcrips number, 2]
            
        gene : list, required
            Gene name of transcrips

        candidate_cell_center : Numpy Array, required
            Coordinates of candidate cell centers. This can be obtained by center_selection
            program.

        sc_data : Scanpy AnnData, required
            Scanpy AnnData of single cell reference. Must include annotation information.
            Thus: sc_data.obs.annotation
            
        prior_count : int, default=100
            Prior estimate of number of transcripts in each cells. No need to be precise
            
        number_min_genes : int, default=10
            Minimun number of genes for single cells.
            sc.pp.filter_cells(sc_data, min_genes=number_min_genes)
        
        number_cell_state : int, default=10
            Number of cell state for each cell type reference.
            
        cell_count_cutoff : int, AutoSelect
            Minimun number of transcripts in each aligned cells.
            default to be prior_count/5.
        
        k_number : int, default=100
            k_number for KNN-graph.
            
        k_number_for_distance: int, default=8
            k_number for calculating distance score
            
        quantile_cutoff : float, default=0.999
            Cutoff of the quantile function for distance distribution. Used for calculating 
            distance score
            
        alphashape_value : int, default=5
            Alpha value for running alphashape program.
            
        top_N_single_cell: int, default=20
            Number of single cell used for cell type inference and expression prediction.
            
        worker_number : int, default=1
            Number of CPU cores.

        output_dir : str, default="./"
            output directory 
            
        device : str, default='cpu'
            Set device for running alignment.
        """
        
        super(alignment, self).__init__()

        print("\n\nRunning RedeFISH ...")
        
        print("\nInitialing data ...")
        
        self.setup_seed(0)
        self.device = device
        self.output_dir = output_dir
        
        self.k_number = int(k_number)
        self.k_number_for_distance = int(k_number_for_distance)
        self.worker_number = int(worker_number)
        self.number_min_genes = int(number_min_genes)
        
        
        self.E_mu_1 = prior_count
        self.E_sigma_1 = 100
        self.E_mu_2 = 0
        self.E_sigma_2 = 100
        self.prior_count = prior_count
        self.cell_number = len(candidate_cell_center)
        self.mRNA_number = len(coordinate)
        self.number_cell_state = number_cell_state
        self.distance_percentage_cutoff = 1 - quantile_cutoff
        self.alphashape_value = alphashape_value
        self.top_N_single_cell = top_N_single_cell

        self.distance_percentage_cutoff = min(max(self.distance_percentage_cutoff,0),1)
        
        if pd.isna(cell_count_cutoff):
            self.cell_count_cutoff = int(prior_count / 5)
        else:
            self.cell_count_cutoff = int(cell_count_cutoff)
        
        self.is_train = True
        self.dropout = nn.Dropout(p=0.5)
        
        
        self.gene = gene
        
        
        self.coordinate = coordinate
        coordinate_normal, coordinate_cell_center = self.coordinate_normalization(coordinate, candidate_cell_center)
        self.build_graph(coordinate_normal, coordinate_cell_center)
        self.prepare_sc_data(sc_data)
        
        
        
        self.zero_pad = torch.zeros((self.cell_number,1)).to(device)
        self.is_noise = nn.Parameter(torch.cat((1-self.distance_score.reshape(-1,1), self.distance_score.reshape(-1,1)),dim=1))
        self.cell_feature = nn.Parameter(0.1 * torch.randn(self.cell_number, 30))
        #self.choose_cell = torch.zeros(self.mRNA_number).type(torch.long).to(device)

    
        self.mRNA2cell = torch.nn.Linear(32, 1)
       
        self.sc_data_list = []
        self.weight_list = nn.ParameterList()
        for each in self.sc_data_dict:
            self.sc_data_list.append(self.sc_data_dict[each].to(device))
            self.weight_list.append(nn.Parameter(0.1 * torch.rand((self.number_cell_state, self.sc_data_dict[each].shape[0])).to(device)))
            #self.weight_list.append(nn.Parameter(1 + 0.5 * torch.rand((self.number_cell_state, self.sc_data_dict[each].shape[0])).to(device)))
        self.weight = nn.Parameter(0.1 * torch.rand((self.cell_number), number_cell_state * len(self.sc_data_list)))
            
           
        self.weight.requires_grad=False
        for i in range(len(self.weight_list)):
            self.weight_list[i].requires_grad=False
            
            
        self.some_parameters = []
        self.some_parameters.append(self.is_noise)
        self.other_parameters = [p for p in self.parameters() if ((p not in set(self.some_parameters)))]

        self.optimizer1 = torch.optim.Adam([{'params': self.some_parameters, 'lr': 0.01, 'initial_lr':0.01},
                                            {'params': self.other_parameters, 'lr': 0.001, 'initial_lr':0.001}], 
                                           lr=0.001)
        self.optimizer2 = torch.optim.Adam([{'params': self.some_parameters, 'lr': 0.01, 'initial_lr':0.01},
                                            {'params': self.other_parameters, 'lr': 0.001, 'initial_lr':0.001}], 
                                           lr=0.001)

        self.scheduler1 = lr_scheduler.ExponentialLR(self.optimizer1, 0.996, 1)
        self.scheduler2 = lr_scheduler.ExponentialLR(self.optimizer2, 0.996, 1)
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    
        
    def train(self,
              epochs = 500,
              batch_size = None):
        
        
        """
        Training RedeFISH alignment model and output alignment results.

        Parameters
        ----------
        epochs : int, default=500
            Number of epochs for the training.
            
        batch_size : int, AutoSelect
            Batch size for the training.
        """
        
        print("")
        if pd.isna(batch_size):
            batch_size = int(min(self.mRNA_number / 50, 2000000)) + 1
            print("Auto select batch size:", batch_size)
        if batch_size > 2000000:
            batch_size = 2000000
            print("Large batch size, set to:", batch_size)
        batch_size = int(batch_size)
        
        
        print("\n  Begin training ...")
        mRNA_index = torch.stack([self.candidate[:,0].reshape(-1), self.gene_index.reshape(-1)], dim=0)
        mRNA_sparse = torch.sparse_coo_tensor(mRNA_index, torch.ones(self.mRNA_number), (self.cell_number, self.expression_dim))
        self.cell_expression = mRNA_sparse.to_dense().to(self.device)
        self.cell_forward(-1)
        
        
        
        
        for i in range(epochs):
            sampling_loss = 0
            count = 0
            del_mRNA = 0
            self.cell_expression = torch.zeros((self.cell_number, self.expression_dim)).to(self.device)
            shuffle = torch.randperm(self.mRNA_number)
            for j in range(0, self.mRNA_number, batch_size):
           
                upper = min(self.mRNA_number, j + batch_size)
                lower = j
                
                shuffle_this = shuffle[lower:upper]
                
                ratio = upper / self.mRNA_number
                
                xy_index_this = self.xy_index[shuffle_this,:]
                mRNA_feature = xy_index_this
                candidate_this = self.candidate[shuffle_this,:]
                gene_this = self.gene_index[shuffle_this]
                
               
                #batch_loss, mRNA_label_this, del_mRNA_this = self.batch_mRNA(ratio, lower, upper, True)
                batch_loss, mRNA_label_this, del_mRNA_this = self.batch_mRNA(ratio, lower, upper, mRNA_feature, candidate_this, gene_this, shuffle_this, True)
                
                del_mRNA += torch.sum(del_mRNA_this).detach()
                sampling_loss += batch_loss.detach()
                count += 1
                

                self.optimizer1.zero_grad()
                batch_loss.backward(retain_graph=True)
                self.optimizer1.step()
                
  
            deconvolution_loss = self.cell_forward(i) 
            
            self.scheduler1.step() 
            self.scheduler2.step()

            sampling_loss = sampling_loss / count
            
            print("  Epoch:", str(i+1) + "/" + str(epochs),
                  "\tRegion Loss:", sampling_loss.cpu().numpy(),
                  "\tAlignment Loss:", deconvolution_loss.cpu().numpy())
            
            number_cell = len(self.prob[self.true_cell])
            #print("del mRNA:",del_mRNA.cpu().detach().numpy(), "number_cell:", number_cell)
            #print(self.E_mu_1.cpu().detach().numpy(), 
            #      self.E_sigma_1.cpu().detach().numpy(),
            #      self.E_mu_2.cpu().detach().numpy(),
            #      self.E_sigma_2.cpu().detach().numpy())
            #print("---------------------------------------------------------------------")        
        
        
        print("\nPredicting ...")
        mRNA_label = torch.zeros((self.mRNA_number, self.k_number))
        del_mRNA = torch.zeros((self.mRNA_number)).type(torch.bool)
        self.cell_expression = torch.zeros((self.cell_number, self.expression_dim)).to(self.device)
        shuffle = torch.tensor(range(self.mRNA_number))
        for j in range(0, self.mRNA_number, batch_size):
        
            upper = min(self.mRNA_number, j + batch_size)
            lower = j
            
            shuffle_this = shuffle[lower:upper]
            
            ratio = upper / self.mRNA_number
            xy_index_this = self.xy_index[shuffle_this,:]
            candidate_this = self.candidate[shuffle_this,:]
            gene_this = self.gene_index[shuffle_this]
            mRNA_feature = xy_index_this.to(self.device)
            
           
            batch_loss, mRNA_label_this, del_mRNA_this = self.batch_mRNA(ratio, lower, upper, mRNA_feature, candidate_this, gene_this, shuffle_this, False)
            mRNA_label[shuffle_this,:] = mRNA_label_this.detach().cpu()
            del_mRNA[shuffle_this] = del_mRNA_this.cpu()
            
        deconvolution_loss = self.cell_forward(0) 
        
        
        mRNA_sort = torch.sort(mRNA_label.cpu(), dim=1, descending=True)
        candidate_sort = torch.gather(self.candidate, 1, mRNA_sort.indices)
        mRNA_class = candidate_sort[:,0]
        mRNA_class[del_mRNA == 1] = self.cell_number
        self.mRNA_class = mRNA_class.type(torch.long).detach()
        
        number_cell = len(self.prob[self.true_cell])
        #print("number_cell:", number_cell)
        

        self.post_process()
       
        
        
    def batch_mRNA(self, ratio, lower, upper, mRNA_feature, candidate, gene_this, shuffle_this, is_train):
    #def batch_mRNA(self, ratio, lower, upper, is_train):
        
        
        self.lower = lower
        self.upper = upper
        self.ratio = ratio
        self.shuffle_this = shuffle_this.to(self.device)
        self.candidate_this = candidate.to(self.device)
        self.gene_this = gene_this.to(self.device)
        self.mRNA_feature = mRNA_feature.to(self.device)
        self.is_train = is_train
            
        
        
        if self.is_train:
            self.mRNA_label = torch.cat((self.mRNA_feature.repeat((1,self.k_number)).reshape(-1,2), self.dropout(self.cell_feature[self.candidate_this.reshape(-1)])), dim=1)
        else:
            self.mRNA_label = torch.cat((self.mRNA_feature.repeat((1,self.k_number)).reshape(-1,2), self.cell_feature[self.candidate_this.reshape(-1)]), dim=1)
        self.mRNA_label = self.mRNA2cell(self.mRNA_label)
        self.mRNA_label = self.mRNA_label.reshape(-1,self.k_number)
        
        
        
        if is_train:
            
            #self.mRNA_label = torch.softmax(self.mRNA_label, dim=-1)
            
            #self.mRNA_label = self.dropout(self.mRNA_label)
            
            self.mRNA_label = torch.softmax(self.mRNA_label, dim=-1)
        
            self.dist = Categorical(probs=self.mRNA_label)
            self.action = self.dist.sample()
            self.choose_cell_this = self.candidate_this[range(len(self.action)), self.action]
            
            self.noise_dist = Categorical(logits=self.is_noise[self.shuffle_this])
            self.noise_action = self.noise_dist.sample()
            
        else:
            
            self.mRNA_label = torch.softmax(self.mRNA_label, dim=-1)
            
            self.dist = Categorical(probs=self.mRNA_label)
            self.action = torch.max(self.mRNA_label, dim=1).indices
            self.choose_cell_this = self.candidate_this[range(len(self.action)), self.action]
            
            self.noise_dist = Categorical(logits=self.is_noise[self.shuffle_this])
            self.noise_action = torch.max(self.is_noise[self.shuffle_this], dim=1).indices
        
        #self.choose_cell[self.shuffle_this] = self.choose_cell_this
       
        
        self.mRNA_index = torch.stack([self.choose_cell_this.reshape(-1), self.gene_this.reshape(-1)], dim=0)
        self.mRNA_sparse = torch.sparse_coo_tensor(self.mRNA_index, self.noise_action, (self.cell_number, self.expression_dim))
        self.cell_expression_this = self.mRNA_sparse.to_dense()
        self.cell_expression += self.cell_expression_this
        
        
        
        
        self.del_mRNA = self.noise_action == 0   
        self.cell_expression_similarity_normal1 = self.delta_expression[self.choose_cell_this, self.gene_this] * (1 - self.cell_expression_similarity[self.choose_cell_this])
        self.cell_expression_similarity_normal2 = (1 - self.delta_expression[self.choose_cell_this, self.gene_this]) * (1 - self.cell_expression_similarity[self.choose_cell_this])
        #self.cell_expression_similarity_normal1 = self.delta_expression[self.choose_cell_this, self.gene_this]
        #self.cell_expression_similarity_normal2 = (1 - self.delta_expression[self.choose_cell_this, self.gene_this])
        #self.cell_expression_similarity_normal1 = (1 - self.cell_expression_similarity[self.choose_cell_this])
        #self.cell_expression_similarity_normal2 = (1 - self.cell_expression_similarity[self.choose_cell_this])
        
       
        
        #self.batch_reward1 = (self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal1 * (self.prob[self.choose_cell_this].reshape(-1))
        #self.batch_reward2 = (1 - self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal2
       
        self.batch_reward1 = (self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal1 * (self.prob[self.choose_cell_this].reshape(-1))
        self.batch_reward2 = (1 - self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal2 * (self.prob[self.choose_cell_this].reshape(-1))
        #self.batch_reward1 = (self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal1
        #self.batch_reward2 = (1 - self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal2
        
        self.batch_reward = self.batch_reward1.clone().type(torch.float32)
        self.batch_reward[self.del_mRNA] = self.batch_reward2[self.del_mRNA].type(torch.float32)
        
        
        noise_log_prob = self.noise_dist.log_prob(self.noise_action)
        log_prob = self.dist.log_prob(self.action)
        
        
        
        batch_loss = - (self.batch_reward * (log_prob.T + noise_log_prob.T)).mean()
      
  
        
        return(batch_loss, self.mRNA_label, self.del_mRNA)
        
        
        
    def cell_forward(self, epoch):
        
       
        if epoch == -1:
            self.cell_classification()
            self.analysis_expression()
            return(1)
        
        
        for i in range(len(self.weight_list)):
            self.weight_list[i].requires_grad=True
        self.weight.requires_grad=True

       

        self.cell_classification()
          
        deconvolution_loss = 0
        decon_size = 30
        for i in range(decon_size):
            self.analysis_expression()
            #cell_loss = (torch.mean((1 - self.cell_expression_similarity[self.true_cell]))).requires_grad_()
            cell_loss = (torch.mean((1 - self.cell_expression_similarity))).requires_grad_()
            #deconvolution_loss += cell_loss.detach()
            deconvolution_loss += torch.mean((1 - self.cell_expression_similarity[self.true_cell])).detach()
            
            self.optimizer2.zero_grad()
            cell_loss.backward(retain_graph=True)
            self.optimizer2.step()
            
        
        for j in range(len(self.weight_list)):
            self.weight_list[j].requires_grad=False
        self.weight.requires_grad=False
                
        return(deconvolution_loss / decon_size)
            
          
            
          
    def analysis_expression(self):
        
        
        self.cell_expression = self.cell_expression / (torch.sum(self.cell_expression[:,1:],dim=1) + 1e-6)[:,None] * (self.expression_dim-1)
        
        


        self.expression_state = []
        for i in range(len(self.weight_list)):
            self.weight_use = F.sigmoid(10 * self.weight_list[i])
            #self.weight_use = torch.relu(10 * self.weight_list[i])
            if self.is_train:
                self.weight_use = self.dropout(self.weight_use)
            self.weight_use = self.weight_use / (torch.sum(self.weight_use, dim=1)[:,None] + 1e-10)
            self.expression_state.append(torch.matmul(self.weight_use, self.sc_data_list[i]))
            
        self.expression_state = torch.cat(self.expression_state)
        self.weight_normal = torch.sigmoid(10 * self.weight)
        #self.weight_normal = torch.relu(10 * self.weight)
        #self.weight_normal = F.elu(self.weight)
        if self.is_train:
            self.weight_normal = self.dropout(self.weight_normal)
        self.weight_normal = self.weight_normal / (torch.sum(self.weight_normal, dim=1)[:,None] + 1e-10)
        
        
        self.predict_cell_expression = torch.matmul(self.weight_normal, self.expression_state)
        self.predict_cell_expression = self.predict_cell_expression / (torch.sum(self.predict_cell_expression, dim=1) + 1e-6)[:,None] *  (self.expression_dim-1)
        self.cell_expression_similarity = torch.cosine_similarity(self.cell_expression[:,1:], self.predict_cell_expression, dim=1)
        
        
        self.delta_expression =  + self.predict_cell_expression - self.cell_expression[:,1:]
        #self.delta_expression = (self.delta_expression - torch.mean(self.delta_expression,dim=1)[:,None]) / torch.std(self.delta_expression,dim=1)[:,None]
        self.delta_expression = torch.cat((self.zero_pad, self.delta_expression), dim=1)
        self.delta_expression = (self.delta_expression - torch.min(self.delta_expression,dim=1).values[:,None]) / (torch.max(self.delta_expression,dim=1).values[:,None] - torch.min(self.delta_expression,dim=1).values[:,None])
        
          
        
        
        
            
    def cell_classification(self):
        
        self.cell_count = torch.sum(self.cell_expression, dim=1)
        
        if self.E_sigma_2 >= self.E_sigma_1:
            self.E_sigma_2 = self.E_sigma_1
        
        N1 = torch.distributions.Normal(self.prior_count, self.E_sigma_1 + self.prior_count / 100)
        N2 = torch.distributions.Normal(0, self.E_sigma_2 + self.prior_count / 100)
        
        self.log_P1 = N1.log_prob(self.cell_count.reshape(-1,1))
        self.log_P2 = N2.log_prob(self.cell_count.reshape(-1,1))
        
    
        self.prob = (1 / (1 + torch.exp(self.log_P2 - self.log_P1))).reshape(-1,1)
        self.cell_noise = torch.cat((self.prob, 1-self.prob), dim=1)
        
        self.true_cell = (self.prob >= 0.5).reshape(-1)
    
        self.E_mu_1 = torch.sum(self.cell_count * self.cell_noise[:,0]) / (torch.sum(self.cell_noise[:,0]) + 1e-2)
        self.E_sigma_1 = torch.sqrt(torch.sum(torch.square(self.cell_count - self.E_mu_1) * self.cell_noise[:,0]) / (torch.sum(self.cell_noise[:,0]) + 1e-2))
        self.E_mu_2 = torch.sum(self.cell_count * self.cell_noise[:,1]) / (torch.sum(self.cell_noise[:,1]) + 1e-2)
        self.E_sigma_2 = torch.sqrt(torch.sum(torch.square(self.cell_count - self.E_mu_2) * self.cell_noise[:,1]) / (torch.sum(self.cell_noise[:,1]) + 1e-2))
        
        
        '''
        N2 = torch.distributions.Normal(0, self.prior_count / 5)
        self.prob = N2.cdf(self.cell_count).reshape(-1,1)
        self.cell_noise = torch.cat((self.prob, 1-self.prob), dim=1)
        self.true_cell = (self.prob >= 0.5).reshape(-1)
        
        self.E_mu_1 = torch.sum(self.cell_count * self.cell_noise[:,0]) / (torch.sum(self.cell_noise[:,0]) + 1e-2)
        self.E_sigma_1 = torch.sqrt(torch.sum(torch.square(self.cell_count - self.E_mu_1) * self.cell_noise[:,0]) / (torch.sum(self.cell_noise[:,0]) + 1e-2))
        self.E_mu_2 = torch.sum(self.cell_count * self.cell_noise[:,1]) / (torch.sum(self.cell_noise[:,1]) + 1e-2)
        self.E_sigma_2 = torch.sqrt(torch.sum(torch.square(self.cell_count - self.E_mu_2) * self.cell_noise[:,1]) / (torch.sum(self.cell_noise[:,1]) + 1e-2))
        '''

    
    def post_process(self):
        
        print("\nPost processing ...")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        mRNA_class1 = list(self.mRNA_class.detach().numpy())
        use_cell = list(set(mRNA_class1))
        use_cell.sort()
        self.df_mRNA_class = pd.DataFrame()
        self.df_mRNA_class['x'] = self.coordinate[:,0]
        self.df_mRNA_class['y'] = self.coordinate[:,1]
        self.df_mRNA_class['mRNA_class'] = mRNA_class1
        
        print("Managing class ...")
        x = list(self.coordinate[:,0])
        y = list(self.coordinate[:,1])
        
        
        noise_class = np.max(mRNA_class1)
        
        sub_data = dict()
        for i in range(len(mRNA_class1)):
            if mRNA_class1[i] == noise_class:
                continue
            if not mRNA_class1[i] in sub_data:
                sub_data[mRNA_class1[i]] = dict()
                sub_data[mRNA_class1[i]]['x'] = []
                sub_data[mRNA_class1[i]]['y'] = []
                sub_data[mRNA_class1[i]]['mRNA_class'] = []
                sub_data[mRNA_class1[i]]['index'] = []
            sub_data[mRNA_class1[i]]['x'].append(x[i])
            sub_data[mRNA_class1[i]]['y'].append(y[i])
            sub_data[mRNA_class1[i]]['mRNA_class'].append(mRNA_class1[i])
            sub_data[mRNA_class1[i]]['index'].append(str(i))
            
        for each in sub_data:
            sub_data[each] = pd.DataFrame(sub_data[each], index=sub_data[each]['index'])
        
        result = []
        print("Generate boundary ...")
        pool = torch.multiprocessing.Pool(processes=self.worker_number)
        for each in sub_data:
            result.append(pool.apply_async(calculate_cell_feature, args=(sub_data[each], each, self.cell_count_cutoff, self.alphashape_value, )))
        pool.close()
        
        

        cell_feature_list = []
        sub_data_list = []
        for each in result:
            temp = each.get()
            if temp[2]:
                cell_feature_list.extend(temp[0])
                sub_data_list.append(temp[1])
                
        
        
        use_cell = dict()
        index = 1
        for each in cell_feature_list:
            if each['is_true_cell']:
                use_cell[each['index']] = index
                index += 1
                
        print("Number of cells:", len(use_cell))
            
        
        
        sub_data_all = pd.concat(sub_data_list)  
        
        index2result_class = dict()
        for a,b in zip(sub_data_all['index'], sub_data_all['result_class']):
            index2result_class[int(a)] = b
        mRNA_class2 = []
        for i in range(len(mRNA_class1)):
            if i in index2result_class:
                temp = index2result_class[i]
                if temp in use_cell:
                    mRNA_class2.append(use_cell[temp])
                else:
                    mRNA_class2.append(0)
            else:
                mRNA_class2.append(0)
        
        
        
        mRNA_class = torch.tensor(np.asarray(mRNA_class2).reshape(-1))
        mRNA_index = torch.stack([mRNA_class, self.gene_index_ori.reshape(-1)])
        mRNA_sparse = torch.sparse_coo_tensor(mRNA_index, torch.ones(self.mRNA_number), (torch.max(mRNA_class)+1, torch.max(self.gene_index_ori)+1))
        cell_expression = mRNA_sparse.to_dense().type(torch.int)
        
        cell_expression = cell_expression[1:,:]
        gene = list(set(self.gene))
        gene.sort()
        self.df_cell_expression = pd.DataFrame(cell_expression.type(torch.long).numpy(), columns=gene)

        
        
        
        
        cell2feature = dict()
        for each in cell_feature_list:
            if each['index'] in use_cell:
                cell2feature[use_cell[each['index']]] = each

        
        
        seg_expression = torch.from_numpy(np.asarray(self.df_cell_expression[self.inter_gene])).to(self.device)
        self.sc_expression = self.sc_expression.to(self.device)
        sc_expression_all = torch.from_numpy(np.asarray(self.sc_data_all_gene.to_df()))
        sc_expression_all = sc_expression_all / (torch.sum(sc_expression_all, dim=1) + 1e-10)[:,None] * sc_expression_all.shape[1]
        seg_expression_predict = torch.zeros((seg_expression.shape[0], sc_expression_all.shape[1]))
        
        annotation_class = list(set(self.annotation_ori))
        cell_type_prob = dict()
        for each in annotation_class:
            cell_type_prob[each] = 0
            
            
 
        TF = torch.zeros((seg_expression.shape[0], 2 * self.top_N_single_cell))
        houxuan = torch.zeros((seg_expression.shape[0], 2 * self.top_N_single_cell)).type(torch.long)
        pinshu = torch.zeros(self.sc_expression.shape[0])
            
        
        for i in range(len(seg_expression)):
            cell_type_prob_this = cell_type_prob.copy()
            seg_expression_this = seg_expression[i,:]
            seg_expression_this = seg_expression_this.reshape(-1,len(self.inter_gene)).repeat((self.sc_expression.shape[0], 1))
            cos_sim = torch.cosine_similarity(seg_expression_this, self.sc_expression)
            cos_sim_sort = torch.sort(cos_sim, descending=True)
            
            topN_index = cos_sim_sort.indices[0:self.top_N_single_cell]
            topN_sim = cos_sim_sort.values[0:self.top_N_single_cell]
            #top_sim = cos_sim_sort.values[0].cpu().numpy()

            for j in range(len(topN_index)):
                cell_type_prob_this[self.annotation_ori[int(topN_index[j].cpu())]] += topN_sim[j].cpu().numpy()
            #cell2feature[i+1]['top_similarity'] = top_sim
            max_cell_type = each
            max_prob = 0
            for each in cell_type_prob_this:
                if cell_type_prob_this[each] > max_prob:
                    max_prob = cell_type_prob_this[each]
                    max_cell_type = each
            if max_prob >= 0.5:
                cell2feature[i+1]['cell_type'] = max_cell_type
            else:
                cell2feature[i+1]['cell_type'] = "UNK"
                
            TF[i, :] = cos_sim_sort.values[0:2 * self.top_N_single_cell].cpu()
            houxuan[i, :] = cos_sim_sort.indices[0:2 * self.top_N_single_cell].cpu()
            pinshu[cos_sim_sort.indices[0:2 * self.top_N_single_cell].cpu()] += 1
                
                
                
        for i in range(len(seg_expression)):
            
            pinshu_this = pinshu[houxuan[i, :]]
            IDF = torch.log(seg_expression.shape[0] / pinshu_this)
            score = TF[i, :] * IDF
            score_sort = torch.sort(score, descending=True)
            choose_cell = houxuan[i, score_sort.indices[0:self.top_N_single_cell]]
            weight = score_sort.values[0:self.top_N_single_cell]
            weight = weight / torch.sum(weight) 
            value = sc_expression_all[choose_cell,:]
            seg_expression_predict[i,:] = torch.sum(value * weight[:,None], dim=0)
            
            '''
            value = sc_expression_all[cos_sim_sort.indices[0:self.top_N_single_cell].cpu(),:]
            weight = cos_sim_sort.values[0:self.top_N_single_cell].cpu()
            weight = weight / torch.sum(weight)
            seg_expression_predict[i,:] = torch.sum(value * weight[:,None], dim=0)
            '''
           
        #self.df_seg_expression_predict = torch.round(seg_expression_predict).type(torch.int).numpy()
        

        self.df_cell2feature = pd.DataFrame(cell2feature).T
        self.df_cell_expression.index = self.df_cell2feature.index
        
        
        
        
         
        
        add_gene = []
        for each in self.sc_data_all_gene.var_names:
            if each in self.df_cell_expression.columns:
                add_gene.append(False)
            else:
                add_gene.append(True)
         
        '''
        self.add_gene = add_gene
        self.gene = gene
        self.cell_expression = cell_expression
        '''

        seg_expression_predict = seg_expression_predict[:,add_gene]
        self.df_seg_expression_predict = seg_expression_predict / (torch.sum(seg_expression_predict, dim=1) + 1e-10)[:,None] * seg_expression_predict.shape[1]
        self.df_seg_expression_predict = torch.concat((cell_expression, self.df_seg_expression_predict), dim=1)
        self.df_seg_expression_predict = self.df_seg_expression_predict / (torch.sum(self.df_seg_expression_predict, dim=1) + 1e-10)[:,None] * self.df_seg_expression_predict.shape[1]
        self.df_seg_expression_predict = pd.DataFrame(self.df_seg_expression_predict.numpy(), columns=list(self.df_cell_expression.columns) + list(self.sc_data_all_gene.var_names[add_gene]))
        self.df_seg_expression_predict.index = self.df_cell2feature.index
        
        
        self.output_result(mRNA_class2)
        
        
        
    def output_result(self, mRNA_class2):
        
        print("\nOutput result ...")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Creat output dir " + self.output_dir)
        
        
        boundary = list(self.df_cell2feature['boundary'])
        new_id = list(self.df_cell2feature.index)
        import h5py
        w = h5py.File(os.path.join(self.output_dir, "cell_boundary.h5"),'w')
        for i in range(len(boundary)):
            w.create_dataset(str(new_id[i]),data=boundary[i])
        w.close()

        del self.df_cell2feature['boundary']
        del self.df_cell2feature['index']
        del self.df_cell2feature['is_true_cell']
        self.df_cell2feature.to_csv(os.path.join(self.output_dir, "cell_feature.csv"), sep='\t')
        
        self.df_cell_expression = sc.AnnData(self.df_cell_expression)
        sc.write(os.path.join(self.output_dir, "cell_expression.h5ad"), self.df_cell_expression)

        self.df_mRNA_class['mRNA_class'] = mRNA_class2
        self.df_mRNA_class.to_csv(os.path.join(self.output_dir,"transcripts_classification.csv"), index=False, sep='\t')
        
        self.df_seg_expression_predict = sc.AnnData(self.df_seg_expression_predict)
        sc.write(os.path.join(self.output_dir, "cell_expression.predict.h5ad"), self.df_seg_expression_predict)
        
        
     















