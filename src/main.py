from train import train_model
from model import UNet
from data_loader import ReservoirDataset
from utils import visualize_prediction
from torch.utils.data import DataLoader
import torch


train_directory = "./train_data"
train_dataset = ReservoirDataset(train_directory, target_var="pressure_buildup")
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_directory = "./val_data"
val_dataset = ReservoirDataset(val_directory, target_var="pressure_buildup")
val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=9, out_channels=24)
training = train_model(
    model,
    train_data_loader,
    val_data_loader,
    device=device_cuda,
)
