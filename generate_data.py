import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from config import Config

def generate_metro_network():
    """生成地铁网络拓扑"""
    cfg = Config()
    
    # 创建站点 (0-83: 普通站, 84-87: 换乘站)
    stations = [{"id": i, "type": "general"} for i in range(84)]
    transfer_stations = [
        {"id": 84, "name": "T1", "lines": [1, 2]},
        {"id": 85, "name": "T2", "lines": [2, 3]},
        {"id": 86, "name": "T3", "lines": [3, 4]},
        {"id": 87, "name": "T4", "lines": [4, 1]}
    ]
    stations.extend(transfer_stations)
    
    # 线路定义 (每条线路20个站)
    lines = {
        1: list(range(0, 20)) + [84] + list(range(20, 40)) + [87],
        2: list(range(40, 60)) + [84] + list(range(60, 80)) + [85],
        3: [80, 81, 82, 83, 85] + list(range(60, 70)) + [86],
        4: [87] + list(range(70, 80)) + [86] + list(range(20, 30))
    }
    
    # 保存拓扑
    network = {
        "stations": stations,
        "lines": lines,
        "transfer_stations": [s["id"] for s in transfer_stations]
    }
    
    with open(os.path.join(cfg.data_path, "metro_network.json"), "w") as f:
        json.dump(network, f, indent=2)
    
    return network

def generate_afc_data(days=30):
    """生成模拟AFC数据"""
    cfg = Config()
    np.random.seed(42)
    
    # 创建时间序列 (每天6:00-22:00)
    start_date = datetime(2023, 1, 1)
    timestamps = []
    for day in range(days):
        current = start_date + timedelta(days=day)
        for hour in range(6, 22):
            for minute in range(0, 60, 5):  # 每5分钟一个记录
                timestamps.append(current.replace(hour=hour, minute=minute))
    
    # 生成进站记录
    data = []
    num_records = len(timestamps) * cfg.num_nodes
    
    # 工作日/周末模式
    for ts in timestamps:
        is_weekend = ts.weekday() >= 5
        for station_id in range(cfg.num_nodes):
            # 基础客流量 + 高峰时段加成
            base_flow = np.random.poisson(20 if station_id < 84 else 50)
            
            # 早晚高峰增强
            if 7 <= ts.hour <= 9:
                base_flow *= np.random.uniform(2.0, 3.0)
            elif 17 <= ts.hour <= 19:
                base_flow *= np.random.uniform(1.8, 2.5)
            
            # 周末模式
            if is_weekend:
                base_flow *= np.random.uniform(0.8, 1.2)
                if 10 <= ts.hour <= 16:
                    base_flow *= np.random.uniform(1.2, 1.5)
            
            # 换乘站额外流量
            if station_id >= 84:
                base_flow *= np.random.uniform(1.5, 2.0)
            
            flow = max(1, int(base_flow))
            
            data.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "station_id": station_id,
                "entry_flow": flow
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(cfg.data_path, "afc_data.csv"), index=False)
    return df

if __name__ == "__main__":
    generate_metro_network()
    generate_afc_data()