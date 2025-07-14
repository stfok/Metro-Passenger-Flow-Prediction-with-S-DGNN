import json
import numpy as np
import torch
from config import Config

class GraphBuilder:
    def __init__(self, network_file):
        cfg = Config()
        with open(network_file, "r") as f:
            self.network = json.load(f)
        
        self.stations = self.network["stations"]
        self.lines = self.network["lines"]
        self.transfer_ids = self.network["transfer_stations"]
        self.num_nodes = cfg.num_nodes
        self.num_transfer = cfg.num_transfer
        self.metro_speed = cfg.metro_speed
        
    def build_original_graph(self):
        """原始地铁拓扑图"""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        
        # 构建线路连接
        for line_id, stations in self.lines.items():
            for i in range(len(stations) - 1):
                s1, s2 = stations[i], stations[i+1]
                adj[s1][s2] = 1.0
                adj[s2][s1] = 1.0
                
        return torch.tensor(adj, dtype=torch.float)
    
    def build_transfer_centric_graph(self):
        """以换乘站为中心的新拓扑图"""
        # 初始化邻接矩阵和边属性
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        edge_weights = np.zeros((self.num_nodes, self.num_nodes))
        distances = np.zeros((self.num_nodes, self.num_nodes))
        intervals = np.zeros((self.num_nodes, self.num_nodes))
        
        # 普通站到换乘站连接
        for station in self.stations:
            s_id = station["id"]
            if s_id in self.transfer_ids:
                continue
                
            for t_id in self.transfer_ids:
                # 检查是否在同一线路
                common_line = False
                for line_id, stations in self.lines.items():
                    if s_id in stations and t_id in stations:
                        common_line = True
                        # 计算距离 (模拟)
                        idx1 = stations.index(s_id)
                        idx2 = stations.index(t_id)
                        distance = abs(idx1 - idx2) * 1.5  # 1.5km/站
                        break
                
                if common_line:
                    adj[s_id][t_id] = 1.0
                    distances[s_id][t_id] = distance
                    intervals[s_id][t_id] = 5.0  # 5分钟发车间隔
        
        # 换乘站之间连接
        for i, t1 in enumerate(self.transfer_ids):
            for j, t2 in enumerate(self.transfer_ids):
                if i == j:
                    continue
                    
                # 检查是否在同一线路
                common_line = False
                for line_id, stations in self.lines.items():
                    if t1 in stations and t2 in stations:
                        common_line = True
                        idx1 = stations.index(t1)
                        idx2 = stations.index(t2)
                        distance = abs(idx1 - idx2) * 1.5
                        break
                
                if common_line:
                    adj[t1][t2] = 1.0
                    adj[t2][t1] = 1.0
                    distances[t1][t2] = distance
                    distances[t2][t1] = distance
                    intervals[t1][t2] = 3.0  # 换乘站发车间隔更短
                    intervals[t2][t1] = 3.0
        
        # 转换为Tensor
        adj = torch.tensor(adj, dtype=torch.float)
        distances = torch.tensor(distances, dtype=torch.float)
        intervals = torch.tensor(intervals, dtype=torch.float)
        
        return adj, distances, intervals
    
    def calculate_time_delay(self, distances, intervals):
        """计算时间延迟"""
        # T = 距离/速度 + 发车间隔/2
        time_delay = distances / (self.metro_speed / 60) + intervals / 2
        return time_delay

if __name__ == "__main__":
    builder = GraphBuilder("data/metro_network.json")
    orig_adj = builder.build_original_graph()
    adj, dist, interval = builder.build_transfer_centric_graph()
    time_delay = builder.calculate_time_delay(dist, interval)
    
    print("Original Adj Shape:", orig_adj.shape)
    print("Transfer-Centric Adj Shape:", adj.shape)
    print("Time Delay Matrix:\n", time_delay[84:88, :4])