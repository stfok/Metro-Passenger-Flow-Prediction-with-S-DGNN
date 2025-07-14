import torch

class Config:
    # 模型参数
    num_nodes = 88  # 84个普通站 + 4个换乘站
    num_transfer = 4
    hidden_units = 100
    seq_len = 4      # 时间序列长度（4*15分钟）
    output_len = 1   # 预测未来15分钟
    
    # 训练参数
    batch_size = 32
    epochs = 100
    lr = 0.001
    
    # 路径
    data_path = "data/"
    processed_path = "data/processed/"
    model_save_path = "model/s_dgnn.pth"
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 其他
    time_interval = 15  # 分钟
    metro_speed = 30    # km/h

config = Config()