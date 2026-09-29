import os
import pickle
import numpy as np
import pandas as pd
import torch
import sys
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
CURRENT_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..','..'))
sys.path.append(PROJECT_ROOT)
import common_function.feature_engineering_functions as fe

def run_data_pipeline(raw_csv_path):
    # 1. Load Raw Data
    print("Loading raw dataset...")
    df = pd.read_csv(raw_csv_path)
    
    # 2. Apply Your Feature Engineering Pipeline
    print("Applying feature engineering...")
    df = fe.sort_detectors_by_intensity(df)
    df = fe.add_weighted_centroid(df)
    df = fe.add_intensity_ratios(df)
    df = fe.add_sensor_distances(df)
    df = fe.add_relative_coordinates(df)
    df = fe.apply_log(df, drop=False)
    
    # 3. Strict Data Splitting (80% Train, 15% Val, 5% Test)
    print("Splitting dataset...")
    df_train, df_temp = train_test_split(df, test_size=0.20, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.25, random_state=42)
    print(f"Train size: {len(df_train)} | Val size: {len(df_val)} | Test size: {len(df_test)}")
    
    # 4. Extract Node Features & Targets for Scaling
    # Node features per sensor: [x, y, I, log_I, rel_x, rel_y]
    def extract_raw_arrays(dataframe):
        nodes = []
        targets = []
        edges = []
        
        for _, row in dataframe.iterrows():
            nf = [
                [row['det1_x'], row['det1_y'], row['det1_I'], row['det1_I_log'], row['rel_det1_x'], row['rel_det1_y']],
                [row['det2_x'], row['det2_y'], row['det2_I'], row['det2_I_log'], row['rel_det2_x'], row['rel_det2_y']],
                [row['det3_x'], row['det3_y'], row['det3_I'], row['det3_I_log'], row['rel_det3_x'], row['rel_det3_y']]
            ]
            nodes.append(nf)
            targets.append([row['source_x'], row['source_y'], np.log1p(row['I_0'])])
            
            # Using 1/dist for geometric edges
            edges.append([
                1.0 / (row['dist_12'] + 1e-5), 1.0 / (row['dist_12'] + 1e-5),
                1.0 / (row['dist_23'] + 1e-5), 1.0 / (row['dist_23'] + 1e-5),
                1.0 / (row['dist_13'] + 1e-5), 1.0 / (row['dist_13'] + 1e-5)
            ])
            
        return np.array(nodes), np.array(targets), np.array(edges)

    print("Extracting feature matrices...")
    X_train_nodes, y_train, E_train = extract_raw_arrays(df_train)
    X_val_nodes, y_val, E_val = extract_raw_arrays(df_val)
    X_test_nodes, y_test, E_test = extract_raw_arrays(df_test)
    
    # 5. Leak-Free Scaling Setup
    print("Fitting scalers on training set only...")
    scaler_x = StandardScaler().fit(X_train_nodes.reshape(-1, X_train_nodes.shape[-1]))
    scaler_y = StandardScaler().fit(y_train)
    # ? DELETE THE scaler_e LINE
    
    # Transform all splits
    def scale_data(nodes, targets, edges):
        N = nodes.shape[0]
        scaled_n = scaler_x.transform(nodes.reshape(-1, nodes.shape[-1])).reshape(N, 3, -1)
        scaled_t = scaler_y.transform(targets)
        # ? JUST PASS THE RAW EDGES (No scaling applied)
        return scaled_n, scaled_t, edges

    X_train_s, y_train_s, E_train_s = scale_data(X_train_nodes, y_train, E_train)
    X_val_s, y_val_s, E_val_s = scale_data(X_val_nodes, y_val, E_val)
    X_test_s, y_test_s, E_test_s = scale_data(X_test_nodes, y_test, E_test)
    
    # 6. Build PyTorch Geometric Graphs
    # Fully connected 3-node topology
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], 
                               [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    
    def to_graph_list(nodes, targets, edges):
        graph_list = []
        for i in range(len(nodes)):
            x = torch.tensor(nodes[i], dtype=torch.float)
            y = torch.tensor([targets[i]], dtype=torch.float)
            edge_attr = torch.tensor(edges[i], dtype=torch.float)
            
            graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        return graph_list

    print("Converting processed arrays to graph datasets...")
    train_graphs = to_graph_list(X_train_s, y_train_s, E_train_s)
    val_graphs = to_graph_list(X_val_s, y_val_s, E_val_s)
    var_test_graphs = to_graph_list(X_test_s, y_test_s, E_test_s)
    
    # 7. Save Everything to Disk
    # 7. Save Everything to Disk
    os.makedirs('graph_data', exist_ok=True)
    torch.save(train_graphs, 'graph_data/train_graphs.pt')
    torch.save(val_graphs, 'graph_data/val_graphs.pt')
    torch.save(var_test_graphs, 'graph_data/test_graphs.pt')
    df_test.to_pickle('graph_data/df_test_raw.pkl')
    
    # ? REMOVE scaler_e FROM THE DICTIONARY
    with open('graph_data/scalers.pkl', 'wb') as f:
        pickle.dump({'scaler_x': scaler_x, 'scaler_y': scaler_y}, f)
        
    print("Data pipeline complete. Graph artifacts saved securely.")

if __name__ == "__main__":
    run_data_pipeline("/content/drive/MyDrive/radiation_source_localization/v1.8/dataset/dataset.csv")