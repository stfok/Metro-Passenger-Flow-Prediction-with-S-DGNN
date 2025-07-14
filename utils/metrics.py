import torch

def masked_mae(preds, labels, null_val=0.0):
    mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=0.0):
    mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))

def masked_mape(preds, labels, null_val=0.0):
    mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100

def calculate_metrics(preds, labels):
    """
    计算MAE, RMSE, MAPE
    preds: [batch, pred_len, num_transfer]
    labels: [batch, pred_len, num_transfer]
    """
    mae = masked_mae(preds, labels)
    rmse = masked_rmse(preds, labels)
    mape = masked_mape(preds, labels)
    return mae.item(), rmse.item(), mape.item()