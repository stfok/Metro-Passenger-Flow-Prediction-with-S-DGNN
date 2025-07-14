import torch
import torch.nn as nn
from .dgnn import DGNN
from config import Config

class SDGNN(nn.Module):
    def __init__(self, num_nodes, num_transfer, in_feats=1, hidden_units=100, seq_len=4):
        super(SDGNN, self).__init__()
        self.cfg = Config()
        self.num_nodes = num_nodes
        self.num_transfer = num_transfer
        self.seq_len = seq_len
        self.hidden_units = hidden_units
        
        # 空间模块 (DGNN)
        self.dgnn = DGNN(in_feats, hidden_units)
        
        # 时间模块 (GRU)
        self.gru = nn.GRU(
            input_size=hidden_units * num_nodes,
            hidden_size=hidden_units,
            num_layers=1,
            batch_first=True
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_units, 64),
            nn.ReLU(),
            nn.Linear(64, num_transfer)
        )
        
    def forward(self, x, adj, time_delay):
        """
        x: 输入序列 [batch, seq_len, num_nodes, features]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        time_delay: 时间延迟矩阵 [num_nodes, num_nodes]
        """
        batch_size = x.size(0)
        
        # 空间特征提取
        spatial_features = []
        for t in range(self.seq_len):
            # 处理每个时间步
            x_t = x[:, t, :, :]
            feat = self.dgnn(x_t, adj, time_delay, t)
            spatial_features.append(feat)
        
        # 组合空间特征 [batch, seq_len, num_nodes * hidden_units]
        spatial_seq = torch.stack(spatial_features, dim=1)
        spatial_seq = spatial_seq.reshape(batch_size, self.seq_len, -1)
        
        # 时间特征提取
        _, h_n = self.gru(spatial_seq)
        h_n = h_n.squeeze(0)
        
        # 输出预测 (换乘站流量)
        output = self.fc_out(h_n)
        return output.unsqueeze(1)  # [batch, 1, num_transfer]

if __name__ == "__main__":
    # 测试模型
    cfg = Config()
    model = SDGNN(cfg.num_nodes, cfg.num_transfer)
    
    # 模拟输入
    x = torch.randn(32, cfg.seq_len, cfg.num_nodes, 1)
    adj = torch.randn(cfg.num_nodes, cfg.num_nodes)
    time_delay = torch.randn(cfg.num_nodes, cfg.num_nodes)
    
    out = model(x, adj, time_delay)
    print("Output shape:", out.shape)  # 应为 [32, 1, 4]