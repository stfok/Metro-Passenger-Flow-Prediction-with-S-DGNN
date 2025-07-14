import torch
from model.s_dgnn import SDGNN
from config import Config
from utils.metrics import calculate_metrics
import matplotlib.pyplot as plt

def main():
    cfg = Config()
    device = cfg.device
    
    # 加载模型
    model = SDGNN(cfg.num_nodes, cfg.num_transfer).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path))
    model.eval()
    
    # 加载测试数据
    test_weekday = torch.load(f"{cfg.processed_path}/test_weekday.pt")
    test_weekend = torch.load(f"{cfg.processed_path}/test_weekend.pt")
    
    test_weekday_loader = torch.utils.data.DataLoader(
        test_weekday, batch_size=cfg.batch_size, shuffle=False)
    test_weekend_loader = torch.utils.data.DataLoader(
        test_weekend, batch_size=cfg.batch_size, shuffle=False)
    
    # 评估工作日模型
    print("Evaluating on weekday data...")
    mae_wd, rmse_wd, mape_wd = evaluate_model(model, test_weekday_loader, device)
    print(f"Weekday Results: MAE={mae_wd:.4f}, RMSE={rmse_wd:.4f}, MAPE={mape_wd:.2f}%")
    
    # 评估周末模型
    print("Evaluating on weekend data...")
    mae_we, rmse_we, mape_we = evaluate_model(model, test_weekend_loader, device)
    print(f"Weekend Results: MAE={mae_we:.4f}, RMSE={rmse_we:.4f}, MAPE={mape_we:.2f}%")
    
    # 可视化部分结果
    visualize_results(model, test_weekday_loader, device, "Weekday")
    visualize_results(model, test_weekend_loader, device, "Weekend")

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

def visualize_results(model, data_loader, device, title):
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        batch = next(iter(data_loader))
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        adj = batch["adj"].to(device)
        time_delay = batch["time_delay"].to(device)
        
        output = model(x, adj, time_delay)
        
        # 转换为CPU numpy
        y = y.cpu().numpy()
        output = output.cpu().numpy()
        
        # 绘制结果
        plt.figure(figsize=(15, 10))
        for i in range(4):  # 4个换乘站
            plt.subplot(2, 2, i+1)
            plt.plot(y[0, :, i], label="True")
            plt.plot(output[0, :, i], label="Predicted")
            plt.title(f"Transfer Station T{i+1} - {title}")
            plt.xlabel("Time Step")
            plt.ylabel("Passenger Flow")
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"results_{title.lower()}.png")
        plt.close()

if __name__ == "__main__":
    main()