import torch
import matplotlib.pyplot as plt
from model.s_dgnn import SDGNN
from config import Config

def visualize_topology():
    cfg = Config()
    builder = GraphBuilder(os.path.join(cfg.data_path, "metro_network.json"))
    
    # 获取原始拓扑
    orig_adj = builder.build_original_graph().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.spy(orig_adj, markersize=1)
    plt.title("Original Metro Topology")
    
    # 获取以换乘站为中心的拓扑
    adj, _, _ = builder.build_transfer_centric_graph()
    adj = adj.numpy()
    
    plt.subplot(122)
    plt.spy(adj, markersize=1)
    plt.title("Transfer-Centric Topology")
    plt.savefig("topology_comparison.png")
    plt.close()

def visualize_flow_patterns():
    # 加载AFC数据
    afc_data = pd.read_csv(os.path.join(cfg.data_path, "afc_data.csv"))
    afc_data["timestamp"] = pd.to_datetime(afc_data["timestamp"])
    
    # 分析换乘站流量
    transfer_ids = [84, 85, 86, 87]
    transfer_data = afc_data[afc_data["station_id"].isin(transfer_ids)]
    
    plt.figure(figsize=(15, 10))
    for i, t_id in enumerate(transfer_ids):
        station_data = transfer_data[transfer_data["station_id"] == t_id]
        daily_flow = station_data.groupby(station_data["timestamp"].dt.hour)["entry_flow"].mean()
        
        plt.subplot(2, 2, i+1)
        daily_flow.plot(kind="bar")
        plt.title(f"Transfer Station T{t_id-83} Hourly Flow")
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Passenger Flow")
    
    plt.tight_layout()
    plt.savefig("transfer_station_flow.png")
    plt.close()

if __name__ == "__main__":
    visualize_topology()
    visualize_flow_patterns()