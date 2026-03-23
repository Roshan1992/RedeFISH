# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------
#
# File: RedeFISH.py
#
# System:         Linux, Windows
# Component Name: RedeFISH
# Version:        20260317
# Language: python3
# Latest Revision Time: 2026/03/17
#
# License: To-be-decided
# Licensed Material - Property of CPNL.
#
# (c) Copyright CPNL. 2026
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
# 2026/03/17   Zhong Yunshan     Release v1.2.0
# ------------------------------------------------------------------------------------


import os
import time
import pandas as pd
import numpy as np
from scipy import spatial, interpolate
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
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint
from shapely.ops import unary_union, voronoi_diagram, nearest_points

#from KMean_KD import KMEANS
#import .center_selection 

warnings.filterwarnings("ignore")





def get_boundary(sub_data, scale, smooth_boundary, alphashape_value):

    
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
    max_value = scale
    
    x_norm = x / max_value
    y_norm = y / max_value
    
    points  = np.asarray([x_norm,y_norm], dtype=np.float32).T

    alpha_shape = alphashape.alphashape(points, alphashape_value)

    if alpha_shape.geom_type != 'MultiPolygon' and alpha_shape.geom_type != 'Polygon':
        return(np.asarray([]))
     

    if alpha_shape.geom_type == 'MultiPolygon':
        alpha_shape_use = get_largest_polygon(alpha_shape)
    elif alpha_shape.geom_type == 'Polygon':
        alpha_shape_use = alpha_shape
    else:
        alpha_shape_use = alpha_shape.geoms[0]

    x_center, y_center = np.asarray(alpha_shape_use.centroid.xy)
        
    boundary_this = calculate_cell_feature_each(alpha_shape_use, max_value, smooth_boundary, mean_x, mean_y, x_center, y_center)
    boundary_this = boundary_this[~np.isnan(boundary_this).any(axis=1)]

    return(boundary_this)


    
def get_boundary_with_nuclei(sub_data, center, nuclei, scale, smooth_boundary, alphashape_value):
    
    if len(sub_data) <= 5:
        return(nuclei)
    
    max_value = scale
    mean_x = np.mean(sub_data[:, 0])
    mean_y = np.mean(sub_data[:, 1])
    x = np.asarray(sub_data[:, 0])
    y = np.asarray(sub_data[:, 1])
    x = x - mean_x
    y = y - mean_y
    
    if np.sum(np.abs(x)) < 1e-5 or np.sum(np.abs(y)) < 1e-5:
        x = x + np.random.randn(len(x)) * 1e-10
        y = y + np.random.randn(len(y)) * 1e-10
    
    x_norm = x / max_value
    y_norm = y / max_value
    points  = np.asarray([x_norm,y_norm], dtype=np.float32).T
    
    x = np.asarray(nuclei[:, 0])
    y = np.asarray(nuclei[:, 1])
    x = x - mean_x
    y = y - mean_y
    
    if np.sum(np.abs(x)) < 1e-5 or np.sum(np.abs(y)) < 1e-5:
        x = x + np.random.randn(len(x)) * 1e-10
        y = y + np.random.randn(len(y)) * 1e-10
    
    x_norm = x / max_value
    y_norm = y / max_value
    nuclei_boundary  = Polygon(np.asarray([x_norm,y_norm], dtype=np.float32).T)
    if not nuclei_boundary.is_valid:
        nuclei_boundary = nuclei_boundary.buffer(0)

    alpha_shape = alphashape.alphashape(points, alphashape_value)

    if alpha_shape.geom_type != 'MultiPolygon' and alpha_shape.geom_type != 'Polygon':
        return(nuclei)

    if alpha_shape.geom_type == 'MultiPolygon':
        max_area = 0
        alpha_shape_use = alpha_shape.geoms[0]
        for each in alpha_shape.geoms:
            area_this = each.intersection(nuclei_boundary).area
            if area_this > max_area:
                max_area = area_this
                alpha_shape_use = each
    elif alpha_shape.geom_type == 'Polygon':
        alpha_shape_use = alpha_shape
    else:
        alpha_shape_use = nuclei_boundary
        
    if not alpha_shape_use.intersection(nuclei_boundary).area == 0:
        alpha_shape_use = alpha_shape_use.union(nuclei_boundary)
    else:
        alpha_shape_use = nuclei_boundary
        
    if not alpha_shape_use.geom_type == 'Polygon':
        alpha_shape_use = alpha_shape_use.geoms[0]

    x_center = (center['x'] - mean_x) / max_value
    y_center = (center['y'] - mean_y) / max_value

    boundary_this = calculate_cell_feature_each(alpha_shape_use,  max_value, smooth_boundary, mean_x, mean_y, x_center, y_center)
    boundary_this = boundary_this[~np.isnan(boundary_this).any(axis=1)]

    return(boundary_this)


def get_largest_polygon(multi_poly):
    if not isinstance(multi_poly, MultiPolygon):
        return multi_poly
    return max(multi_poly.geoms, key=lambda p: p.area)


def smooth_polygon(polygon, n_points=50, s=0):
    x, y = polygon.exterior.xy
    x, y = np.array(x), np.array(y)
    tck, u = interpolate.splprep([x, y], s=s, per=True)
    #tck, u = interpolate.splprep([x, y], per=True)
    t_new = np.linspace(0, 1, n_points)
    x_smooth, y_smooth = interpolate.splev(t_new, tck)
    return Polygon(np.column_stack([x_smooth, y_smooth]))



def calculate_cell_feature_each(each, max_value, smooth_boundary, mean_x, mean_y, x_center, y_center):

    smoothed = smooth_polygon(each, n_points=50, s=smooth_boundary)
    
  
    x_new, y_new = smoothed.exterior.xy
    x_new, y_new = np.array(x_new), np.array(y_new)
    value = np.abs(x_new - x_center) + np.abs(y_new - y_center)
    x_new = x_new + 0.025 * (x_new - x_center) / value * 6.5
    y_new = y_new + 0.025 * (y_new - y_center) / value * 6.5
    x_new = x_new * max_value + mean_x
    y_new = y_new * max_value + mean_y
    
    boundary = np.asarray([x_new, y_new]).T

    return(boundary)



    
    

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
    
    
    def coordinate_normalization(self, coordinate, candidate_cell_center, candidate_boundary=None):
        
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
        xy_mean = torch.mean(xy_index)
        xy_std = torch.std(xy_index)
        self.xy_index = (xy_index - xy_mean) / xy_std
        
        coordinate_normal = (torch.from_numpy(coordinate_normal).type(torch.float32) - xy_mean) / xy_std
        coordinate_cell_center = (torch.from_numpy(coordinate_cell_center).type(torch.float32) - xy_mean) / xy_std

        if candidate_boundary is not None:
            center_x = (candidate_boundary[:,0] - min_x) / (max_x - min_x) * up_x
            center_y = (candidate_boundary[:,1] - min_y) / (max_y - min_y) * up_y
            coordinate_boundary = np.asarray([center_x, center_y]).T
            coordinate_boundary = (torch.from_numpy(coordinate_boundary).type(torch.float32) - xy_mean) / xy_std
            return(coordinate_normal, coordinate_cell_center, coordinate_boundary)
        else:
            return(coordinate_normal, coordinate_cell_center)
    
    
    
    
    def build_graph(self, coordinate_normal, coordinate_cell_center):
        
        """
        Build KNN-Graph for cell segmentation.
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
        distance_score = self.distance2 / cutoff
        distance_score[distance_score >= 1] = 1
        distance_score = 1 - distance_score
        self.distance_score = distance_score.type(torch.float32).to(self.device)
        
        tree = spatial.KDTree(data=coordinate_cell_center)
        nearest_neibor = tree.query(coordinate_normal,k=self.k_number,workers=self.worker_number)
        self.candidate = torch.from_numpy(nearest_neibor[1])


    def build_graph_with_prior(self, coordinate_normal, coordinate_cell_center, coordinate_boundary, class_boundary):
        
        """
        Build KNN-Graph with prior nuclei boundary for cell segmentation.
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
        distance_score = self.distance2 / cutoff
        distance_score[distance_score >= 1] = 1
        distance_score = 1 - distance_score
        self.distance_score = distance_score.type(torch.float32).to(self.device)



        coordinate_normal = coordinate_normal.numpy()
        coordinate_cell_center = coordinate_cell_center.numpy()
        coordinate_boundary = coordinate_boundary.numpy()
        
        tree = spatial.KDTree(data=coordinate_cell_center)
        nearest_neibor = tree.query(coordinate_normal, k=self.k_number, workers=self.worker_number)
        self.candidate = np.asarray(nearest_neibor[1])


        nuclei2boundary = dict()
        for i in range(len(class_boundary)):
            if not class_boundary[i] in nuclei2boundary:
                nuclei2boundary[class_boundary[i]] = []
            nuclei2boundary[class_boundary[i]].append(coordinate_boundary[i,])
        
        for each in nuclei2boundary:
            nuclei2boundary[each] = Polygon(np.asarray(nuclei2boundary[each]))


        polygons = np.array([nuclei2boundary[idx] for idx in self.candidate.ravel()])
        coordinate_normal2 = shapely.points(np.repeat(coordinate_normal, repeats=self.candidate.shape[1], axis=0))
        distances = shapely.distance(polygons, coordinate_normal2)
        self.all_delta = torch.from_numpy(distances.reshape(self.candidate.shape).astype(np.float32))
        self.candidate = torch.from_numpy(self.candidate)


        
        
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
        
        
        
        self.sc_data = sc_data
        self.sc_expression = torch.from_numpy(np.asarray(sc_data.to_df())).to(self.device)
        
        
        
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
        
        
        self.sc_expression = self.sc_expression / torch.sum(self.sc_expression, dim=1)[:,None] * self.sc_expression.shape[1]
        
        
        self.celltype = list(set(self.annotation))
        self.celltype.sort()
        
        self.number_cell_type = len(self.celltype)
        

        '''
        self.sc_data_dict = dict()
        for i in range(len(self.annotation)):
            if not self.annotation[i] in self.sc_data_dict:
                self.sc_data_dict[self.annotation[i]] = []
            self.sc_data_dict[self.annotation[i]].append(self.sc_expression[i,:])

        for each in self.sc_data_dict:
            self.sc_data_dict[each] = torch.stack(self.sc_data_dict[each])
            self.sc_data_dict[each] = self.sc_data_dict[each] / torch.sum(self.sc_data_dict[each], dim=1)[:,None] * self.sc_data_dict[each].shape[1]           
        '''
        
        
        
        
        
    
    def __init__(self, 
                 coordinate, 
                 gene,
                 data_prior, 
                 sc_data,
                 scale = 6.5,
                 cell_count_cutoff = 10,
                 number_min_genes = 10,
                 number_cell_state = 10,
                 k_number = 8, 
                 k_number_for_distance = 100,
                 quantile_cutoff = 0.999,
                 alphashape_value = 5,
                 top_N_single_cell = 20,
                 worker_number = 1,
                 smooth_boundary = 0.2,
                 boundary_prior = False,
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

        data_prior : pandas DataFrame, required
            Must contain x, y and cell_id columns.
            The data prior can be either a cell location prior (each cell_id corresponds to one x and y) 
            or a nuclear boundary prior (each cell_id corresponds to multiple x and y).

        sc_data : Scanpy AnnData, required
            Scanpy AnnData of single cell reference. Must include annotation information.
            Thus: sc_data.obs.annotation
            
        cell_count_cutoff : int, default=10
            Minimun number of transcripts in each aligned cells.
            default to be 10.
            
        number_min_genes : int, default=10
            Minimun number of genes for single cells.
            sc.pp.filter_cells(sc_data, min_genes=number_min_genes)
        
        number_cell_state : int, default=10
            Number of cell state for each cell type reference.
        
        k_number : int, default=8
            k_number for KNN-graph.
            
        k_number_for_distance: int, default=100
            k_number for calculating distance score
            
        quantile_cutoff : float, default=0.999
            Cutoff of the quantile function for distance distribution. Used for calculating 
            distance score
            
        alphashape_value : int, default=5
            Alpha value for running alphashape program.
            
        top_N_single_cell: int, default=20
            Number of single cell used for cell type inference and expression prediction.

        smooth_boundary: float, default=0
            The degree of boundary smoothing; the smaller the value, the less smooth the boundary, with a minimum of 0.

        boundary_prior: boolean, default=False
            False represents using cell location as the prior, True represents using nuclear boundary.
            
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
        self.boundary_prior = boundary_prior
        self.output_dir = output_dir
        
        self.k_number = int(k_number)
        self.k_number_for_distance = int(k_number_for_distance)
        self.worker_number = int(worker_number)
        self.number_min_genes = int(number_min_genes)
        
        
        self.scale = scale
        self.smooth_boundary = smooth_boundary
        self.cell_number = len(np.unique(data_prior['cell_id']))
        self.mRNA_number = len(coordinate)
        self.number_cell_state = number_cell_state
        self.distance_percentage_cutoff = 1 - quantile_cutoff
        self.alphashape_value = alphashape_value
        self.top_N_single_cell = top_N_single_cell

        self.distance_percentage_cutoff = min(max(self.distance_percentage_cutoff,0),1)
        
        self.cell_count_cutoff = int(cell_count_cutoff)
        
        self.is_train = True
        self.dropout = nn.Dropout(p=0.2)
        
        
        self.gene = gene
        self.coordinate = coordinate

        
        
        if self.boundary_prior:
            boundary = np.float32(data_prior[['x','y']])
            cell_id = data_prior['cell_id']
            x = data_prior['x']
            y = data_prior['y']
            nuclei2boundary = dict()
            cell2index = dict()
            index = 0
            for i in range(len(cell_id)):
                if not cell_id[i] in cell2index:
                    cell2index[cell_id[i]] = index
                    index += 1
            for i in range(len(cell_id)):
                if not cell2index[cell_id[i]] in nuclei2boundary:
                    nuclei2boundary[cell2index[cell_id[i]]] = []
                nuclei2boundary[cell2index[cell_id[i]]].append([x[i],y[i]])
            candidate_cell_center = []
            for each in nuclei2boundary:
                nuclei2boundary[each] = np.asarray(nuclei2boundary[each])
                candidate_cell_center.append(np.asarray(Polygon(nuclei2boundary[each]).centroid.xy).T)
            candidate_cell_center = np.concatenate(candidate_cell_center, axis=0)
            
            class_boundary = []
            for i in range(len(cell_id)):
                class_boundary.append(cell2index[cell_id[i]])
            
            self.coordinate_normal, self.coordinate_cell_center, self.coordinate_boundary = self.coordinate_normalization(coordinate, candidate_cell_center, boundary)
            self.build_graph_with_prior(self.coordinate_normal, self.coordinate_cell_center, self.coordinate_boundary, class_boundary)
            self.nuclei2boundary = nuclei2boundary


        else:
            candidate_cell_center = np.float32(data_prior[['x','y']])
            self.coordinate_normal, self.coordinate_cell_center = self.coordinate_normalization(coordinate, candidate_cell_center)
            self.build_graph(self.coordinate_normal, self.coordinate_cell_center)
            self.coordinate_cell_center = torch.tensor(np.float32(self.coordinate_cell_center)).to(self.device)
        
        self.prepare_sc_data(sc_data)
        
        
        
        self.zero_pad = torch.zeros((self.cell_number,1)).to(device)
        self.is_noise = nn.Parameter(torch.cat((1-self.distance_score.reshape(-1,1), self.distance_score.reshape(-1,1)),dim=1))
        self.cell_feature = nn.Parameter(0.1 * torch.randn(self.cell_number, 20))
        self.gene_feature = nn.Parameter(0.1 * torch.randn(self.expression_dim, 20))
        #self.cell_feature = torch.zeros([self.cell_number, 1]).to(self.device)
        #self.choose_cell = torch.zeros(self.mRNA_number).type(torch.long).to(device)

        self.location2feature1 = torch.nn.Linear(2, 10)
        self.location2feature2 = torch.nn.Linear(10, 20)
        self.mRNA2cell1 = torch.nn.Linear(20, 10)
        self.mRNA2cell2 = torch.nn.Linear(10, 5)
        self.mRNA2cell3 = torch.nn.Linear(5, 1)
        
        #self.layer_norm = torch.nn.LayerNorm([20])
              
        
        
        self.sc_drop = nn.Parameter((torch.zeros(self.sc_expression.shape)+1).to(self.device))
        self.weight_list = nn.Parameter((0.1 * torch.rand(self.number_cell_state, self.sc_expression.shape[0])).to(self.device))
        self.weight = nn.Parameter(0.1 * torch.rand((self.cell_number), self.number_cell_state * len(self.celltype)))
         
        self.CT_index = []
        for i in range(self.number_cell_type):
            temp = torch.tensor(self.annotation == self.celltype[i]).to(self.device)
            self.CT_index.append(temp)
           
           
           
        self.weight.requires_grad=False
        self.weight_list.requires_grad=False
        self.sc_drop.requires_grad=False
       
            
            
        self.some_parameters = []
        self.some_parameters.append(self.is_noise)
        self.other_parameters = [p for p in self.parameters() if ((p not in set(self.some_parameters)))]

        self.optimizer1 = torch.optim.Adam([{'params': self.some_parameters, 'lr': 0.01, 'initial_lr':0.01},
                                            {'params': self.other_parameters, 'lr': 0.002, 'initial_lr':0.002}], 
                                           lr=0.002)
        self.optimizer2 = torch.optim.Adam([{'params': self.some_parameters, 'lr': 0.01, 'initial_lr':0.01},
                                            {'params': self.other_parameters, 'lr': 0.002, 'initial_lr':0.002}], 
                                           lr=0.002)

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
        self.cell_x = torch.zeros((self.cell_number)).to(self.device)
        self.cell_y = torch.zeros((self.cell_number)).to(self.device)
        mRNA_index = torch.stack([self.candidate[:,0].reshape(-1), self.gene_index.reshape(-1)], dim=0)
        mRNA_sparse = torch.sparse_coo_tensor(mRNA_index, torch.ones(self.mRNA_number), (self.cell_number, self.expression_dim))
        self.cell_expression = mRNA_sparse.to_dense().to(self.device)
        self.cell_forward(-1)
        
        
        
        
        for i in range(epochs):
            sampling_loss = 0
            count = 0
            del_mRNA = 0
            self.cell_expression = torch.zeros((self.cell_number, self.expression_dim)).to(self.device)
            self.cell_x = torch.zeros((self.cell_number)).to(self.device)
            self.cell_y = torch.zeros((self.cell_number)).to(self.device)
            shuffle = torch.randperm(self.mRNA_number)
            for j in range(0, self.mRNA_number, batch_size):
           
                upper = min(self.mRNA_number, j + batch_size)
                lower = j
                
                shuffle_this = shuffle[lower:upper]
                
                ratio = upper / self.mRNA_number
                
                
                candidate_this = self.candidate[shuffle_this,:]
                gene_this = self.gene_index[shuffle_this]
                if self.boundary_prior:
                    mRNA_feature = self.all_delta[shuffle_this,:]
                else:
                    mRNA_feature = self.xy_index[shuffle_this,:]
                
                batch_loss, mRNA_label_this, del_mRNA_this = self.batch_mRNA(ratio, lower, upper, mRNA_feature, candidate_this, gene_this, shuffle_this, True)
                
                del_mRNA = del_mRNA + torch.sum(del_mRNA_this).detach()
                sampling_loss = sampling_loss + batch_loss.detach()
                count += 1
                

                self.optimizer1.zero_grad()
                batch_loss.backward(retain_graph=True)
                self.optimizer1.step()
                
  
            deconvolution_loss = self.cell_forward(i) 
            
            self.scheduler1.step() 
            self.scheduler2.step()

            sampling_loss = sampling_loss / count
            
            print("  Epoch:", str(i+1) + "/" + str(epochs),
                  "\tSegmentation Loss:", sampling_loss.cpu().numpy(),
                  "\tDeconvolution Loss:", deconvolution_loss.cpu().numpy())
                       

            #print("del mRNA:",del_mRNA.cpu().detach().numpy(), "number_cell:", len(self.cell_count[self.cell_count > 0]))
            #print(self.cell_count.cpu().detach().numpy())
            #print(self.weight_entropy)
            #print(torch.mean(self.batch_reward))
            #print("---------------------------------------------------------------------")        
        
        
        print("\nPredicting ...")
        mRNA_label = torch.zeros((self.mRNA_number, self.k_number))
        del_mRNA = torch.zeros((self.mRNA_number)).type(torch.bool)
        self.cell_expression = torch.zeros((self.cell_number, self.expression_dim)).to(self.device)
        self.cell_x = torch.zeros((self.cell_number)).to(self.device)
        self.cell_y = torch.zeros((self.cell_number)).to(self.device)
        shuffle = torch.tensor(range(self.mRNA_number))
        for j in range(0, self.mRNA_number, batch_size):
        
            upper = min(self.mRNA_number, j + batch_size)
            lower = j
            
            shuffle_this = shuffle[lower:upper]
            
            ratio = upper / self.mRNA_number
            candidate_this = self.candidate[shuffle_this,:]
            gene_this = self.gene_index[shuffle_this]
            if self.boundary_prior:
                mRNA_feature = self.all_delta[shuffle_this,:]
            else:
                mRNA_feature = self.xy_index[shuffle_this,:]
           
            batch_loss, mRNA_label_this, del_mRNA_this = self.batch_mRNA(ratio, lower, upper, mRNA_feature, candidate_this, gene_this, shuffle_this, False)
            mRNA_label[shuffle_this,:] = mRNA_label_this.detach().cpu()
            del_mRNA[shuffle_this] = del_mRNA_this.cpu()
            
        deconvolution_loss = self.cell_forward(-1) 
        
        
        mRNA_sort = torch.sort(mRNA_label.cpu(), dim=1, descending=True)
        candidate_sort = torch.gather(self.candidate, 1, mRNA_sort.indices)
        mRNA_class = candidate_sort[:,0]
        mRNA_class[del_mRNA == 1] = self.cell_number
        self.mRNA_class = mRNA_class.type(torch.long).detach()


        self.post_process()
       
    
    def layer_norm(self, vector, eps = 1e-5):
        
        mean_value = torch.mean(vector, dim=-1)
        std_value = torch.std(vector, dim=-1)
        vector = (vector - mean_value[:,None]) / (std_value[:,None] + eps)
        
        return(vector)
        
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
            
              
        
        if self.boundary_prior:

            self.mRNA_label = torch.relu(self.location2feature1(torch.concat([self.mRNA_feature.reshape(-1,1), self.mRNA_feature.reshape(-1,1)], dim=1)))
            #self.mRNA_label = torch.relu(self.location2feature1((self.mRNA_feature.repeat((1,self.k_number)).reshape(-1,2) - self.coordinate_cell_center[self.candidate_this.reshape(-1)])))
            self.mRNA_label = self.layer_norm(self.location2feature2(self.mRNA_label))
        else:
            self.mean_cell = torch.concat([self.mean_cell_x.reshape(-1,1), self.mean_cell_y.reshape(-1,1)], dim=1)  
            self.mRNA_label = torch.relu(self.location2feature1((self.mRNA_feature.repeat((1,self.k_number)).reshape(-1,2) - self.mean_cell[self.candidate_this.reshape(-1)])))
            self.mRNA_label = self.layer_norm(self.location2feature2(self.mRNA_label))




        if self.is_train:
            
            self.mRNA_label = self.dropout(self.mRNA_label + \
                self.layer_norm(self.cell_feature[self.candidate_this.reshape(-1)]) + \
                self.layer_norm(self.gene_feature[self.gene_this.repeat((1,self.k_number)).reshape(-1)]))
            
        else:
            
            self.mRNA_label = self.mRNA_label + \
                self.layer_norm(self.cell_feature[self.candidate_this.reshape(-1)]) + \
                self.layer_norm(self.gene_feature[self.gene_this.repeat((1,self.k_number)).reshape(-1)])
            
                
        self.mRNA_label = torch.relu(self.mRNA2cell1(self.mRNA_label))
        self.mRNA_label = torch.relu(self.mRNA2cell2(self.mRNA_label))
        self.mRNA_label = self.mRNA2cell3(self.mRNA_label)
        self.mRNA_label = self.mRNA_label.reshape(-1,self.k_number)
        
        
        if is_train:
            
            #self.mRNA_label = self.dropout(self.mRNA_label)
            #self.mRNA_label = torch.softmax(self.mRNA_label, dim=-1)
        
            #self.dist = Categorical(probs=self.mRNA_label)
            self.dist = Categorical(logits=self.mRNA_label)
            self.action = self.dist.sample()
            self.choose_cell_this = self.candidate_this[range(len(self.action)), self.action]
            
            self.noise_dist = Categorical(logits=self.is_noise[self.shuffle_this])
            self.noise_action = self.noise_dist.sample()
            
        else:
            
            #self.mRNA_label = torch.softmax(self.mRNA_label, dim=-1)
            
            #self.dist = Categorical(probs=self.mRNA_label)
            self.dist = Categorical(logits=self.mRNA_label)
            self.action = torch.max(self.mRNA_label, dim=1).indices
            self.choose_cell_this = self.candidate_this[range(len(self.action)), self.action]
            
            self.noise_dist = Categorical(logits=self.is_noise[self.shuffle_this])
            self.noise_action = torch.max(self.is_noise[self.shuffle_this], dim=1).indices
        
  
        
        self.mRNA_index = torch.stack([self.choose_cell_this.reshape(-1), self.gene_this.reshape(-1)], dim=0)
        self.mRNA_sparse = torch.sparse_coo_tensor(self.mRNA_index, self.noise_action, (self.cell_number, self.expression_dim))
        self.cell_expression_this = self.mRNA_sparse.to_dense()
        self.cell_expression = self.cell_expression + self.cell_expression_this
        
        self.del_mRNA = self.noise_action == 0  
        self.real_mRNA = self.noise_action == 1
        

        self.cell_expression_similarity_normal1 = (self.delta_expression[self.choose_cell_this, self.gene_this].detach())
        self.cell_expression_similarity_normal2 = (1 - self.delta_expression[self.choose_cell_this, self.gene_this].detach())

        

        if self.boundary_prior:
            #self.delta_x = self.mRNA_feature[:,0] - self.coordinate_cell_center[self.choose_cell_this][:,0]
            #self.delta_y = self.mRNA_feature[:,1] - self.coordinate_cell_center[self.choose_cell_this][:,1]
            #self.delta_x = self.all_delta[self.shuffle_this, self.choose_cell_this].to(self.device)
            #self.delta_y = self.all_delta[self.shuffle_this, self.choose_cell_this].to(self.device)
            #self.delta_score = self.mRNA_feature[:,self.choose_cell_this].to(self.device)
            #self.delta_score = torch.gather(self.mRNA_feature, 1, self.action) 
            self.delta_score = torch.gather(self.mRNA_feature, 1, self.action.unsqueeze(1)).reshape(-1)
        else:
            self.center_index = torch.stack([self.choose_cell_this.reshape(-1), torch.zeros(len(self.choose_cell_this)).to(self.device)], dim=0)
            self.x_sparse = torch.sparse_coo_tensor(self.center_index, self.noise_action * self.mRNA_feature[:,0], (self.cell_number, 1))
            self.y_sparse = torch.sparse_coo_tensor(self.center_index, self.noise_action * self.mRNA_feature[:,1], (self.cell_number, 1))
            self.cell_x = self.cell_x + self.x_sparse.to_dense().reshape(-1)
            self.cell_y = self.cell_y + self.y_sparse.to_dense().reshape(-1)
            self.delta_x = self.mRNA_feature[:,0] - self.mean_cell[self.choose_cell_this][:,0]
            self.delta_y = self.mRNA_feature[:,1] - self.mean_cell[self.choose_cell_this][:,1]
            self.delta_score = torch.sqrt((torch.square(self.delta_x) + torch.square(self.delta_y)) / 2)
        cutoff = torch.sort(self.delta_score, descending=True).values[[int(len(self.delta_score) * 0.1)]]
        #distance_score = torch.round(self.distance2 / (cutoff / 1000)) / 1000
        self.delta_score = self.delta_score / cutoff
        self.delta_score[self.delta_score >= 1] = 1
        self.delta_score = 1 - self.delta_score       
        
        
        #self.batch_reward1 = (self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal1 * self.within_distance[range(len(self.action)), self.action]
        #self.batch_reward2 = (1 - self.distance_score[self.shuffle_this]) * self.cell_expression_similarity_normal2 * self.within_distance[range(len(self.action)), self.action]
        self.batch_reward1 = (self.distance_score[self.shuffle_this]) * (self.cell_expression_similarity_normal1 * self.delta_score)
        self.batch_reward2 = (1 - self.distance_score[self.shuffle_this]) * (self.cell_expression_similarity_normal2 * self.delta_score)
        #self.batch_reward1 = (self.distance_score[self.shuffle_this]) * (self.delta_score)
        #self.batch_reward2 = (1 - self.distance_score[self.shuffle_this]) * (self.delta_score)




        self.batch_reward = self.batch_reward1.clone().type(torch.float32)
        self.batch_reward[self.del_mRNA] = self.batch_reward2[self.del_mRNA].type(torch.float32)
        
        
        noise_log_prob = self.noise_dist.log_prob(self.noise_action)
        log_prob = self.dist.log_prob(self.action)
        
        
        
        batch_loss = - (self.batch_reward * (log_prob.T + noise_log_prob.T)).mean()
      
  
        
        return(batch_loss, self.mRNA_label, self.del_mRNA)
        
        
        
    def cell_forward(self, epoch):
        
       
        if epoch == -1:
            if not self.boundary_prior:
                self.cell_count = torch.sum(self.cell_expression, dim=1)
                self.mean_cell_x = self.cell_x / (self.cell_count + 1e-6)
                self.mean_cell_y = self.cell_y / (self.cell_count + 1e-6)
            self.analysis_expression()
            return(1)
        
        
        self.weight_list.requires_grad=True
        self.weight.requires_grad=True
        self.sc_drop.requires_grad=True

        if not self.boundary_prior:
            self.cell_count = torch.sum(self.cell_expression, dim=1)
            self.mean_cell_x = self.cell_x / (self.cell_count + 1e-6)
            self.mean_cell_y = self.cell_y / (self.cell_count + 1e-6)

          
        deconvolution_loss = 0
        decon_size = 30

        #haha = torch.sum(self.cell_expression[:,1:], dim=1)
        #print(haha, len(haha[haha>0]))
        #print(self.mRNA_feature)
         
        for i in range(decon_size):
            self.analysis_expression()
            #cell_loss = (torch.mean((1 - self.cell_expression_similarity[self.true_cell]))).requires_grad_()
            #deconvolution_loss += cell_loss.detach()
            
            deconvolution_loss = deconvolution_loss + self.cell_loss.detach()
            
            self.optimizer2.zero_grad()
            self.cell_loss.backward(retain_graph=True)
            self.optimizer2.step()
            
            
        self.weight_list.requires_grad=False
        self.weight.requires_grad=False
        self.sc_drop.requires_grad=False
                
        return(deconvolution_loss / decon_size)
            
          
            
          
    def analysis_expression(self):
        
        
        self.sc_trans = self.sc_expression * (5 * F.softplus(self.sc_drop))
        self.sc_trans = self.sc_trans / (torch.sum(self.sc_trans, dim=1)[:,None] + 1e-6) * (self.expression_dim - 1)
        
        self.weight_use = F.softplus(5 * self.weight_list)
        
        self.expression_state = []
        for i in range(self.number_cell_type):
            self.weight_use_this = self.weight_use[:, self.CT_index[i]]
            if self.is_train:
                self.weight_use_this = self.dropout(self.weight_use_this)
            self.weight_use_this = self.weight_use_this / (torch.sum(self.weight_use_this, dim=1)[:,None] + 1e-7)
            self.weight_use[:, self.CT_index[i]] = self.weight_use_this
            self.expression_state.append(torch.matmul(self.weight_use_this, self.sc_expression[self.CT_index[i]]))
            
            
        self.expression_state = torch.cat(self.expression_state)
        self.weight_normal = F.softplus(5 * self.weight)
        #self.weight_normal = torch.relu(10 * self.weight)
        #self.weight_normal = F.elu(self.weight)
        if self.is_train:
            self.weight_normal = self.dropout(self.weight_normal)
        self.weight_normal = self.weight_normal / (torch.sum(self.weight_normal, dim=1)[:,None] + 1e-7)
        
        
        self.predict_cell_expression = torch.matmul(self.weight_normal, self.expression_state)
        self.predict_cell_expression = self.predict_cell_expression / (torch.sum(self.predict_cell_expression, dim=1) + 1e-6)[:,None] * (self.expression_dim-1)
        self.cell_expression_similarity = torch.cosine_similarity(self.cell_expression[:,1:], self.predict_cell_expression, dim=1)
       

        self.cell_expression = self.cell_expression + 1e-6
        self.cell_expression = (self.cell_expression) / (torch.sum(self.cell_expression[:,1:],dim=1))[:,None] * (self.expression_dim-1)
        
        '''
        self.delta_expression = self.predict_cell_expression - self.cell_expression[:,1:]
        #self.delta_expression = (self.delta_expression - torch.mean(self.delta_expression,dim=1)[:,None]) / torch.std(self.delta_expression,dim=1)[:,None]
        self.delta_expression = torch.cat((self.zero_pad, self.delta_expression), dim=1)
        self.delta_expression = (self.delta_expression - torch.min(self.delta_expression,dim=1).values[:,None]) / (torch.max(self.delta_expression,dim=1).values[:,None] - torch.min(self.delta_expression,dim=1).values[:,None])
        '''
        
        self.delta_expression = torch.log(self.predict_cell_expression+1) - torch.log(self.cell_expression[:,1:]+1)
        self.delta_expression = torch.sigmoid(self.delta_expression)
        self.delta_expression = torch.cat((self.zero_pad + 0.5, self.delta_expression), dim=1)
        #self.delta_expression = (self.delta_expression - torch.min(self.delta_expression,dim=1).values[:,None]) / (torch.max(self.delta_expression,dim=1).values[:,None] - torch.min(self.delta_expression,dim=1).values[:,None])
        
        
        self.weight_entropy =  - torch.sum(self.weight_normal * torch.log(self.weight_normal + 1e-7), dim=1) / np.log(self.expression_state.shape[0])
        
        self.trans_similarity = torch.cosine_similarity(self.sc_trans , self.sc_expression, dim=1)
        self.loss_trans_sim = 1 * (torch.matmul(torch.sum(self.weight_use, dim=0), 1 - self.trans_similarity) / torch.sum(self.weight_use))
        
        self.cell_loss = (torch.mean((1 - self.cell_expression_similarity) + 1 * self.weight_entropy)).requires_grad_()



    def calculate_boundary(self, mRNA_class1):

        noise_class = np.max(mRNA_class1)

        if self.boundary_prior:

            nuclei_center = []
            for each in self.nuclei2boundary:
                self.nuclei2boundary[each] = np.asarray(self.nuclei2boundary[each])
                nuclei_center.append(np.asarray(Polygon(self.nuclei2boundary[each]).centroid.xy).T)
            nuclei_center = np.concatenate(nuclei_center, axis=0)

            
            cell2point = dict()
            cell2position = dict()
            for i in range(len(nuclei_center)):
                cell2position[i] = {}
                cell2position[i]['x'] = float(nuclei_center[i,0])
                cell2position[i]['y'] = float(nuclei_center[i,1])
                cell2point[i] = []
                

            use_index = [i for i in range(len(mRNA_class1)) if not mRNA_class1[i] == noise_class]
            x = [cell2position[int(index)]['x'] for index in mRNA_class1 if not index == noise_class]
            y = [cell2position[int(index)]['y'] for index in mRNA_class1 if not index == noise_class]
            x2 = self.coordinate[use_index,0]
            y2 = self.coordinate[use_index,1]
            distance = np.sqrt((np.square(x2 - x) + np.square(y2 - y))/2)
            mRNA_class2 = mRNA_class1[use_index]
            
            for i in range(len(distance)):
                if distance[i] <= self.scale * 2.56:
                    cell2point[mRNA_class2[i]].append([x2[i], y2[i]])
            for each in cell2point:
                cell2point[each] = np.asarray(cell2point[each])


            result = []
            
            pool = torch.multiprocessing.Pool(processes=self.worker_number)
            for each in cell2point:
                sub_data = cell2point[each]
                center = cell2position[each]
                nuclei = self.nuclei2boundary[each]
                result.append(pool.apply_async(get_boundary_with_nuclei, args=(sub_data, center, nuclei, self.scale, self.smooth_boundary, self.alphashape_value,)))
            pool.close()  
            
            cell2boundary = dict()
            index = 0
            for each in result:
                boundary_this = each.get()
                if len(boundary_this) > 0:
                    cell2boundary[index] = boundary_this
                    index += 1



            nuclei_center_MultiPoint = MultiPoint(nuclei_center.tolist())
            voronoi = voronoi_diagram(nuclei_center_MultiPoint)
            voronoi_center = []
            for i in range(len(voronoi.geoms)):
                voronoi_center.append(np.asarray(voronoi.geoms[i].centroid.xy).T)
            voronoi_center = np.concatenate(voronoi_center)
            tree = spatial.KDTree(nuclei_center)
            
            voronoi2boundary = dict()
            for i in range(len(voronoi.geoms)):
                nearest_center = tree.query(voronoi_center[i,])
                if Point(nuclei_center[nearest_center[1],]).within(voronoi.geoms[i]):
                    voronoi2boundary[nearest_center[1]] = np.asarray(voronoi.geoms[i].exterior.xy).T
                else:
                    nearest_center = tree.query(voronoi_center[i,], k = len(voronoi_center))
                    for j in range(len(nearest_center[0])):
                        if Point(nuclei_center[nearest_center[1][j],]).within(voronoi.geoms[i]):
                            voronoi2boundary[nearest_center[1]][j] = np.asarray(voronoi.geoms[i].exterior.xy).T
                            break
                    
                
            tree = spatial.KDTree(nuclei_center)
            nearest_nuclei = tree.query(nuclei_center, k=10, workers=self.worker_number)
            
            for i in range(len(cell2boundary)):
                candidate = nearest_nuclei[1][i,1:]
                A = Polygon(cell2boundary[i]).buffer(0)
                for each in candidate:
                    B = Polygon(self.nuclei2boundary[each]).buffer(0)
                    if A.intersects(B):
                        inter_region = A.intersection(B)
                        A = A.difference(B)
                        if A.geom_type == 'MultiPolygon':
                            A = get_largest_polygon(A)
                        cell2boundary[i] = np.asarray(A.exterior.xy).T



                   
            for each in cell2boundary:
                A = Polygon(cell2boundary[each]).buffer(0)
                B = Polygon(self.nuclei2boundary[each]).buffer(0)
                C = A.union(B).buffer(0)
                if C.geom_type == 'MultiPolygon':
                    C = get_largest_polygon(C)
                cell2boundary[each] = np.asarray(C.exterior.xy).T




            
            for i in range(len(cell2boundary)):
                candidate = nearest_nuclei[1][i,1:]
                A = Polygon(cell2boundary[i]).buffer(0)
                for j in candidate:
                    B = Polygon(cell2boundary[j]).buffer(0)    
                    if A.intersects(B):
                        inter_region = A.intersection(B).buffer(0)
                        A2 = Polygon(voronoi2boundary[i]).buffer(0)
                        B2 = Polygon(voronoi2boundary[j]).buffer(0)
                        try:
                            A = A.difference(inter_region).buffer(0)
                        except:
                            A = A
                        try:
                            B = B.difference(inter_region).buffer(0)
                        except:
                            B = B
                        inter_region_A = A2.intersection(inter_region)
                        inter_region_B = B2.intersection(inter_region)
                        A = A.union(inter_region_A).buffer(0)
                        B = B.union(inter_region_B).buffer(0)
                        if A.geom_type == 'MultiPolygon':
                            A = get_largest_polygon(A)
                        if B.geom_type == 'MultiPolygon':
                            B = get_largest_polygon(B)
                        cell2boundary[i] = np.asarray(A.exterior.xy).T
                        cell2boundary[j] = np.asarray(B.exterior.xy).T

        else:

            x = list(self.coordinate[:,0])
            y = list(self.coordinate[:,1])
            
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
            pool = torch.multiprocessing.Pool(processes=self.worker_number)
            for each in sub_data:
                result.append(pool.apply_async(get_boundary, args=(sub_data[each], self.scale, self.smooth_boundary, self.alphashape_value, )))
            pool.close()


            cell2boundary = dict()
            index = 0
            for each in result:
                boundary_this = each.get()
                if len(boundary_this) > 0:
                    cell2boundary[index] = boundary_this
                    index += 1

        return(cell2boundary)

        


    def post_process(self):
        
        print("\nPost processing ...")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        mRNA_class1 = self.mRNA_class.detach().numpy()
        noise_class = np.max(mRNA_class1)
        use_cell = list(set(mRNA_class1))
        use_cell.sort()
        self.df_mRNA_class = pd.DataFrame()
        self.df_mRNA_class['x'] = self.coordinate[:,0]
        self.df_mRNA_class['y'] = self.coordinate[:,1]
        self.df_mRNA_class['mRNA_class'] = mRNA_class1

        #self.df_mRNA_class.to_csv("initial_assignment.csv", index=False, sep='\t')
        

        print("Generate boundary ...")
        self.cell2boundary = self.calculate_boundary(mRNA_class1)
        
        print("Transcripts assignment ...")

        cell_center = []
        for each in self.cell2boundary:
            cell_center.append(np.mean(self.cell2boundary[each], axis=0))
        cell_center = np.asarray(cell_center)
        
        tree = spatial.KDTree(cell_center)
        nearest_neibor = tree.query(self.coordinate, k=self.k_number, workers=self.worker_number)
        
        for each in self.cell2boundary:
            self.cell2boundary[each] = Polygon(self.cell2boundary[each])


        
        polygons = np.array([self.cell2boundary[idx] for idx in nearest_neibor[1].ravel()])
        coordinate = shapely.points(np.repeat(self.coordinate, repeats=self.k_number, axis=0))
        distances = shapely.distance(polygons, coordinate).reshape([len(self.coordinate), self.k_number])

        '''
        seg_expression = np.zeros([len(self.cell2boundary), int(torch.max(self.gene_index_ori)) + 1])
        transcripts_assignment = np.zeros(distances.shape[0]) + noise_class
        for i in range(distances.shape[0]):
            if mRNA_class1[i] == noise_class:
                continue
            gene_this = int(self.gene_index_ori[i])
            for j in range(distances.shape[1]):
                if distances[i, j] <= 1e-8:
                    seg_expression[mRNA_class1[i], gene_this] += 1
                    transcripts_assignment[i] = mRNA_class1[i]
                    continue
        '''
        
        seg_expression = np.zeros([len(self.cell2boundary), int(torch.max(self.gene_index_ori)) + 1])
        transcripts_assignment = np.zeros(distances.shape[0]) + len(self.cell2boundary)
        for i in range(distances.shape[0]):
            if mRNA_class1[i] == noise_class:
                continue
            gene_this = int(self.gene_index_ori[i])
            for j in range(distances.shape[1]):
                if distances[i, j] <= 1e-8:
                    seg_expression[nearest_neibor[1][i,j], gene_this] += 1
                    transcripts_assignment[i] = nearest_neibor[1][i,j]
                    continue
        
        
        gene = list(set(self.gene))
        gene.sort()
        
        seg_expression = pd.DataFrame(np.int32(seg_expression))
        seg_expression.columns = gene
        cell_expression = sc.AnnData(seg_expression)
        
        seg_expression = torch.from_numpy(np.asarray(seg_expression[self.inter_gene])).to(self.device)
        seg_expression = seg_expression / (torch.sum(seg_expression, dim=1) + 1e-10)[:,None] * seg_expression.shape[1]
        self.sc_expression = self.sc_expression.to(self.device)
        self.sc_expression = self.sc_expression / (torch.sum(self.sc_expression, dim=1) + 1e-10)[:,None] * self.sc_expression.shape[1]
        sc_expression_all = torch.from_numpy(np.asarray(self.sc_data_all_gene.to_df()))
        sc_expression_all = sc_expression_all / (torch.sum(sc_expression_all, dim=1) + 1e-10)[:,None] * sc_expression_all.shape[1]
        seg_expression_predict = torch.zeros((seg_expression.shape[0], sc_expression_all.shape[1]))
        
        annotation_class = list(set(self.annotation_ori))
        cell_type_prob = dict()
        for each in annotation_class:
            cell_type_prob[each] = 0

    
        print("Label transfering ...")
        TF = torch.zeros((seg_expression.shape[0], 2 * self.top_N_single_cell))
        houxuan = torch.zeros((seg_expression.shape[0], 2 * self.top_N_single_cell)).type(torch.long)
        pinshu = torch.zeros(self.sc_expression.shape[0])
        CT = []
        
        for i in range(len(seg_expression)):
            cell_type_prob_this = cell_type_prob.copy()
            seg_expression_this = seg_expression[i,:]
            seg_expression_this = seg_expression_this.reshape(-1,len(self.inter_gene)).repeat((self.sc_expression.shape[0], 1))
            cos_sim = torch.cosine_similarity(torch.log(seg_expression_this+1), torch.log(self.sc_expression+1))
            #cos_sim = torch.cosine_similarity(seg_expression_this, self.sc_expression)
            cos_sim_sort = torch.sort(cos_sim, descending=True)
            
            topN_index = cos_sim_sort.indices[0:self.top_N_single_cell]
            topN_sim = cos_sim_sort.values[0:self.top_N_single_cell]
            #top_sim = cos_sim_sort.values[0].cpu().numpy()
        
            for j in range(len(topN_index)):
                cell_type_prob_this[self.annotation_ori[int(topN_index[j].cpu())]] += topN_sim[j].cpu().numpy()
         
            max_cell_type = each
            max_prob = 0
            for each in cell_type_prob_this:
                if cell_type_prob_this[each] > max_prob:
                    max_prob = cell_type_prob_this[each]
                    max_cell_type = each
            if max_prob >= 0.5:
                CT.append(max_cell_type)
            else:
                CT.append("UNK")
                
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


        print("Generating output ...")
        cell_feature = []
        index = 1
        mRNA_count = np.sum(cell_expression.to_df(), axis=1)
        
        cell_old2new = dict()
        use_cell = []
        for i in range(len(self.cell2boundary)):
            if mRNA_count[i] >= self.cell_count_cutoff or self.boundary_prior: 
                center_x, center_y = np.asarray(self.cell2boundary[i].centroid.xy)
                area = self.cell2boundary[i].area
                girth = self.cell2boundary[i].length
                roundness = 4 * np.pi * area / np.power(girth, 2)
                cell_feature.append({"cell_id":index, "center_x":float(center_x), "center_y":float(center_y), "girth":girth, "area":area, "roundness":roundness, "mRNA_count":mRNA_count[i], "cell_type":CT[i]})
                cell_old2new[i] = index
                index += 1
                use_cell.append(True)
            else:
                use_cell.append(False)
        self.df_cell_feature = pd.DataFrame(cell_feature)


        cell_boundary = []
        for i in range(len(self.cell2boundary)):
            if use_cell[i]:
                df_this = pd.DataFrame(np.asarray(self.cell2boundary[i].exterior.xy).T)
                df_this.columns = ['x', 'y']
                df_this['cell_id'] = cell_old2new[i]
                cell_boundary.append(df_this)
        
        self.cell_boundary = pd.concat(cell_boundary, axis=0)

        
        transcripts_classification = []
        for i in range(len(transcripts_assignment)):
            if transcripts_assignment[i] in cell_old2new:
                tag = cell_old2new[transcripts_assignment[i]]
            else:
                tag = 0
            transcripts_classification.append({"x":self.coordinate[i,0], "y":self.coordinate[i,1], "mRNA_class":tag})
        self.transcripts_classification = pd.DataFrame(transcripts_classification)

        
        self.cell_expression = cell_expression[use_cell,:]
        self.cell_expression.obs_names = self.df_cell_feature['cell_id']
        
        self.cell_expression_predict = seg_expression_predict / (torch.sum(seg_expression_predict, dim=1) + 1e-10)[:,None] * seg_expression_predict.shape[1]
        self.cell_expression_predict = self.cell_expression_predict / (torch.sum(self.cell_expression_predict, dim=1) + 1e-10)[:,None] * self.cell_expression_predict.shape[1]
        self.cell_expression_predict = pd.DataFrame(self.cell_expression_predict.numpy(), columns=list(self.sc_data_all_gene.var_names))
        self.cell_expression_predict = sc.AnnData(self.cell_expression_predict)
        self.cell_expression_predict = self.cell_expression_predict[use_cell, :]
        self.cell_expression_predict.obs_names = self.df_cell_feature['cell_id']
    
        self.output_result()
        
        
    def output_result(self):
        
        print("\nOutput result ...")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Creat output dir " + self.output_dir)


        self.df_cell_feature.to_csv(os.path.join(self.output_dir, "cell_feature.csv"), sep='\t', index=False)
        self.cell_boundary.to_csv(os.path.join(self.output_dir, "cell_boundary.csv"), sep='\t', index=False)
        self.transcripts_classification.to_csv(os.path.join(self.output_dir, "transcripts_classification.csv"), sep='\t', index=False)
        sc.write(os.path.join(self.output_dir, "cell_expression.h5ad"), self.cell_expression)
        sc.write(os.path.join(self.output_dir, "cell_expression.predict.h5ad"), self.cell_expression_predict)


        
     




