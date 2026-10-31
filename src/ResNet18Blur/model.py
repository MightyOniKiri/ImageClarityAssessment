import torch
import time
import os
import matplotlib.pyplot as plt
from pathlib import Path
import csv

from data_loader import data_sets, dataset_sizes, class_names, device

class train():
    def __init__(self):
        self.history_result = []
        self.best_acc = 0
    
    def train_model(self, model, loss_fn, scheduler, optimizer, epochs=10):
        train_start_time = time.perf_counter()
        
        best_model_path = os.path.join(os.path.dirname(__file__), 'best_model_params.pt')
        
        for epoch in range(epochs):
            start_time = time.perf_counter()
            epoch_result = {'epoch': epoch + 1}
            print(f'Epoch {epoch + 1}/{epochs}')
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    
                running_loss = 0.0
                running_corrects = 0.0
                
                for input_datas, labels in data_sets[phase]:
                    input_datas = input_datas.to(device)
                    labels = labels.to(device)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(input_datas)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_fn(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()    
                        
                    running_loss += loss.item() * input_datas.size(0)
                    running_corrects += torch.sum(preds == labels)
                    
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                epoch_result[f'{phase}_loss'] = epoch_loss
                epoch_result[f'{phase}_acc'] = epoch_acc.item()
                    
                print(f"epoch:{epoch + 1}, accuracy:{epoch_acc:4f}, loss:{epoch_loss:4f}")
                
                if phase == 'val':
                    self.history_result.append(epoch_result)
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_path)
                
            time_epoch_end = time.perf_counter()
            time_elapsed = time_epoch_end - start_time
            print(f"epoch{epoch + 1}, time:{time_elapsed:.2f} seconds")
        
        train_total_time = time.perf_counter() - train_start_time
        print()
        print(f"Training total time:{train_total_time:.2f} seconds")
        print(f"Best accuracy:{self.best_acc:4f}")
                
        # 保留最佳轮次的参数
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

        # 写入文件并保存训练数据
        save_csv_path = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'ResNet18' / 'training_history.csv'
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss','val_acc'])
            writer.writeheader()
            writer.writerows(self.history_result)
            
        return model
    
    def draw_loss(self):
        train_losses = [row["train_loss"] for row in self.history_result]
        val_losses = [row["val_loss"] for row in self.history_result]
        epochs = [row["epoch"] for row in self.history_result]

        plt.figure()
        plt.plot(epochs, train_losses, label='train')
        plt.plot(epochs, val_losses, label='val')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and validation loss')
        plt.grid(True)
        plt.legend()

        # 保存到项目目录
        save_path = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'ResNet18' / 'loss_curve.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    def draw_acc(self):
        train_acc = [row["train_acc"] for row in self.history_result]
        val_acc = [row["val_acc"] for row in self.history_result]
        epochs = [row["epoch"] for row in self.history_result]
        
        plt.figure()
        plt.plot(epochs, train_acc, label='train')
        plt.plot(epochs, val_acc, label='val')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and validation accuracy')
        plt.grid(True)
        plt.legend()

        # 保存到项目目录
        save_path = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'ResNet18' / 'acc_curve.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()