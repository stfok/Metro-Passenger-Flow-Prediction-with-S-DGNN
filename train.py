import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from model.s_dgnn import SDGNN
from config import Config
from utils.metrics import calculate_metrics

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        adj = batch["adj"].to(device)
        time_delay = batch["time_delay"].to(device)
        
        optimizer.zero_grad()
        output = model(x, adj, time_delay)
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 50 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_mae, total_rmse, total_mape = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            adj = batch["adj"].to(device)
            time_delay = batch["time_delay"].to(device)
            
            output = model(x, adj, time_delay)
            mae, rmse, mape = calculate_metrics(output, y)
            
            total_mae += mae
            total_rmse += rmse
            total_mape += mape
    
    num_batches = len(data_loader)
    return total_mae/num_batches, total_rmse/num_batches, total_mape/num_batches

def main():
    cfg = Config()
    device = cfg.device
    
    # 创建模型
    model = SDGNN(cfg.num_nodes, cfg.num_transfer).to(device)
    
    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 加载数据
    train_weekday = torch.load(f"{cfg.processed_path}/train_weekday.pt")
    test_weekday = torch.load(f"{cfg.processed_path}/test_weekday.pt")
    train_weekend = torch.load(f"{cfg.processed_path}/train_weekend.pt")
    test_weekend = torch.load(f"{cfg.processed_path}/test_weekend.pt")
    
    # 创建DataLoader
    train_weekday_loader = torch.utils.data.DataLoader(
        train_weekday, batch_size=cfg.batch_size, shuffle=True)
    test_weekday_loader = torch.utils.data.DataLoader(
        test_weekday, batch_size=cfg.batch_size, shuffle=False)
    train_weekend_loader = torch.utils.data.DataLoader(
        train_weekend, batch_size=cfg.batch_size, shuffle=True)
    test_weekend_loader = torch.utils.data.DataLoader(
        test_weekend, batch_size=cfg.batch_size, shuffle=False)
    
    print("Starting training...")
    best_mae = float("inf")
    
    for epoch in range(cfg.epochs):
        start_time = time.time()
        
        # 训练工作日模型
        train_loss = train_model(model, train_weekday_loader, optimizer, criterion, device)
        
        # 评估
        val_mae, val_rmse, val_mape = evaluate_model(model, test_weekday_loader, device)
        
        # 保存最佳模型
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), cfg.model_save_path)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{cfg.epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val MAPE: {val_mape:.2f}%")
    
    print("Training completed!")
    
    # 在周末数据上微调
    print("Fine-tuning on weekend data...")
    for param_group in optimizer.param_groups:
        param_group["lr"] = cfg.lr * 0.1  # 使用更小的学习率
    
    for epoch in range(10):  # 少量epoch
        train_loss = train_model(model, train_weekend_loader, optimizer, criterion, device)
        val_mae, val_rmse, val_mape = evaluate_model(model, test_weekend_loader, device)
        print(f"Weekend Epoch {epoch+1}/10 | Loss: {train_loss:.4f} | MAE: {val_mae:.4f}")

if __name__ == "__main__":
    main()