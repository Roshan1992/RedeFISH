# Component Name: example
# Version:        20250207
# Language: python3
# Latest Revision Time: 2025/02/07
#
# License: To-be-decided
# Licensed Material - Property of CPNL.
#
# (c) Copyright CPNL. 2025
#
# Address:
# 28#, ZGC Science and Technology Park, Changping District, Beijing, China
#
# Author: Zhong Yunshan
# E-Mail: 327922729@qq.com
#
# Description: Example of RedeFISH V1.1.0
#
# Change History:
# Date         Author            Description
# 2025/02/07   Zhong Yunshan     Release V1.1.0
# ------------------------------------------------------------------------------------


import scanpy as sc
import numpy as np
import pandas as pd
import RedeFISH
import time
import os





device='cuda'
r = 30
st_data = pd.read_csv("example_data/st_data.csv")
sc_data = sc.read("example_data/sc_data.h5ad")


sc_data.var_names_make_unique()

coordinate = np.asarray([st_data['x'], st_data['y']]).T

if __name__ == "__main__":
    
    time1 = time.time()
    
    
    if device == 'cuda':
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
        gpu_id = str(np.argmin([int(x.split()[2]) for x in open('tmp','r').readlines()]))
        os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
        os.system('rm tmp')
        print("Select GPU:", gpu_id)
    
    
    model = RedeFISH.center_selection(r, st_data, sc_data, number_gene_for_cluster=3)
    estimated_cell_locations = model.predict_cell_location()
    
    model = RedeFISH.alignment(coordinate, 
                                  list(st_data['gene']),
                                  estimated_cell_locations,
                                  sc_data,
                                  quantile_cutoff = 0.999,
                                  output_dir = "test_output", 
                                  device = device).to(device)
    

    model.train(epochs=500)
    
        
    
  
            
