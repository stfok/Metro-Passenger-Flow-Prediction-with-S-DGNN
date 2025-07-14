import torch
import torch.nn as nn
from config import Config

class DGNN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DGNN, self).__init__()
        self.cfg = Config()
        self.linear1 = nn.Linear(in_feats, out_feats)
        self.linear2 = nn.Linear(out_feats, out_feats)
        
    def forward(self, x, adj, time_delay, t):
        """
        x: 输入特征 [batch, num_nodes, in_feats]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        time_delay: 时间延迟矩阵 [num_nodes, num_nodes]
        t: 当前时间步索引
        """
        batch_size, num_nodes, _ = x.shape
        
        # 应用时间延迟调整
        adjusted_x = torch.zeros_like(x)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] > 0:
                    # 计算延迟时间步
                    delay_steps = int(time_delay[i, j] // self.cfg.time_interval)
                    
                    # 获取延迟后的特征
                    if t >= delay_steps:
                        delayed_feat = x[:, j, :]  # 这里简化处理
                    else:
                        delayed_feat = torch.zeros_like(x[:, j, :])
                    
                    # 加权聚合
                    adjusted_x[:, i, :] += adj[i, j] * delayed_feat
        
        # 第一层图卷积
        x = torch.relu(self.linear1(adjusted_x))
        # 第二层图卷积
        x = torch.sigmoid(self.linear2(x))
        
        return x