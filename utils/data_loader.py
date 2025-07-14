import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

class MetroDataset(Dataset):
    def __init__(self, data, adj, time_delay, seq_len=4, pred_len=1):
        """
        data: 输入数据 [num_timesteps, num_nodes, num_features]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        time_delay: 时间延迟矩阵 [num_nodes, num_nodes]
        """
        self.data = data
        self.adj = adj
        self.time_delay = time_delay
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = data.shape[1]
        
    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        # 提取输入序列 [seq_len, num_nodes, features]
        x = self.data[index:index+self.seq_len]
        
        # 提取目标 (换乘站未来流量) [pred_len, num_transfer]
        # 注意: 换乘站ID为最后4个
        y = self.data[index+self.seq_len:index+self.seq_len+self.pred_len, -4:]
        
        # 添加时间延迟信息
        return {
            "x": torch.FloatTensor(x),
            "y": torch.FloatTensor(y),
            "adj": self.adj,
            "time_delay": self.time_delay
        }

def load_and_process_data():
    cfg = Config()
    processed_path = cfg.processed_path
    
    # 检查是否已处理
    if os.path.exists(os.path.join(processed_path, "graph_data.pt")):
        return torch.load(os.path.join(processed_path, "graph_data.pt"))
    
    # 加载AFC数据
    afc_data = pd.read_csv(os.path.join(cfg.data_path, "afc_data.csv"))
    afc_data["timestamp"] = pd.to_datetime(afc_data["timestamp"])
    
    # 按15分钟间隔重采样
    afc_data.set_index("timestamp", inplace=True)
    flow_data = afc_data.groupby("station_id").resample("15T").sum()["entry_flow"].unstack(level=0)
    
    # 处理缺失值
    flow_data = flow_data.fillna(method="ffill").fillna(0)
    
    # 转换为 [timesteps, num_nodes] 形状
    flow_matrix = flow_data.values.T  # [num_nodes, timesteps]
    flow_matrix = flow_matrix.reshape((flow_matrix.shape[0], flow_matrix.shape[1], 1))
    
    # 加载图数据
    builder = GraphBuilder(os.path.join(cfg.data_path, "metro_network.json"))
    adj, dist, interval = builder.build_transfer_centric_graph()
    time_delay = builder.calculate_time_delay(dist, interval)
    
    # 保存处理后的数据
    os.makedirs(processed_path, exist_ok=True)
    torch.save({
        "flow_matrix": torch.FloatTensor(flow_matrix),
        "adj": adj,
        "time_delay": time_delay
    }, os.path.join(processed_path, "graph_data.pt"))
    
    return {
        "flow_matrix": torch.FloatTensor(flow_matrix),
        "adj": adj,
        "time_delay": time_delay
    }

def create_data_loaders(data_dict, seq_len=4, pred_len=1, test_ratio=0.2):
    flow_matrix = data_dict["flow_matrix"]
    adj = data_dict["adj"]
    time_delay = data_dict["time_delay"]
    
    # 划分数据集 (工作日和周末)
    weekday_idx = []
    weekend_idx = []
    
    # 模拟时间索引 (实际应用中应根据实际日期)
    num_timesteps = flow_matrix.shape[1]
    for i in range(num_timesteps):
        if i % 7 < 5:  # 前5天为工作日
            weekday_idx.append(i)
        else:
            weekend_idx.append(i)
    
    # 创建数据集
    weekday_data = flow_matrix[:, weekday_idx, :].permute(1, 0, 2)  # [timesteps, nodes, features]
    weekend_data = flow_matrix[:, weekend_idx, :].permute(1, 0, 2)
    
    # 分割训练测试集
    split_idx = int(len(weekday_idx) * (1 - test_ratio))
    train_weekday = weekday_data[:split_idx]
    test_weekday = weekday_data[split_idx:]
    
    split_idx = int(len(weekend_idx) * (1 - test_ratio))
    train_weekend = weekend_data[:split_idx]
    test_weekend = weekend_data[split_idx:]
    
    # 创建数据集对象
    train_weekday_ds = MetroDataset(train_weekday, adj, time_delay, seq_len, pred_len)
    test_weekday_ds = MetroDataset(test_weekday, adj, time_delay, seq_len, pred_len)
    train_weekend_ds = MetroDataset(train_weekend, adj, time_delay, seq_len, pred_len)
    test_weekend_ds = MetroDataset(test_weekend, adj, time_delay, seq_len, pred_len)
    
    # 创建DataLoader
    train_weekday_loader = DataLoader(train_weekday_ds, batch_size=Config().batch_size, shuffle=True)
    test_weekday_loader = DataLoader(test_weekday_ds, batch_size=Config().batch_size, shuffle=False)
    train_weekend_loader = DataLoader(train_weekend_ds, batch_size=Config().batch_size, shuffle=True)
    test_weekend_loader = DataLoader(test_weekend_ds, batch_size=Config().batch_size, shuffle=False)
    
    return {
        "train_weekday": train_weekday_loader,
        "test_weekday": test_weekday_loader,
        "train_weekend": train_weekend_loader,
        "test_weekend": test_weekend_loader
    }

if __name__ == "__main__":
    data_dict = load_and_process_data()
    loaders = create_data_loaders(data_dict)
    print("Data loaders created successfully!")