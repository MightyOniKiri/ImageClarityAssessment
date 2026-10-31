import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

from model import train
from data_loader import class_names, device

def main():
    epochs = 50
    train_model = train()
    model_ft = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    
    model_ft = model_ft.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer_fn, step_size=10, gamma=0.1)
    model_ft = train_model.train_model(model_ft, loss_fn, scheduler, optimizer_fn, epochs)
    train_model.draw_loss()
    train_model.draw_acc()
    
if __name__=='__main__':
    main()