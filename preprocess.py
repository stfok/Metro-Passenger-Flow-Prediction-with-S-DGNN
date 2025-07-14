from utils.data_loader import load_and_process_data, create_data_loaders
from config import Config

def main():
    cfg = Config()
    
    # 加载并处理数据
    print("Loading and processing data...")
    data_dict = load_and_process_data()
    
    # 创建数据加载器
    print("Creating data loaders...")
    loaders = create_data_loaders(data_dict)
    
    # 保存数据加载器
    torch.save(loaders["train_weekday"].dataset, f"{cfg.processed_path}/train_weekday.pt")
    torch.save(loaders["test_weekday"].dataset, f"{cfg.processed_path}/test_weekday.pt")
    torch.save(loaders["train_weekend"].dataset, f"{cfg.processed_path}/train_weekend.pt")
    torch.save(loaders["test_weekend"].dataset, f"{cfg.processed_path}/test_weekend.pt")
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()