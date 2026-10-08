from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from data_loader_d import data_loader
import time
from deep_model import Trans,lstm,TextCNN
def binary_acc(preds, y):
    score_p, prediction = torch.max(preds, 1)
    # print("------pred-----")
    # print(prediction)
    score_t, target = torch.max(y, 1)
    correct = torch.eq(prediction, target).float()
    acc = correct.sum() /( len(correct))
    return acc


def train(model, iterator, optimizer, criteon):

    avg_loss = []
    avg_acc = []
    model.train()
    for i, (data,label) in enumerate(iterator):
        data=Variable(data).float()
        label=Variable(label).float()
        pred = model(data)
        loss = criteon(pred, label)
        acc = binary_acc(pred, label).item()
        avg_loss.append(loss.item())
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc


def evaluate(model, iterator, criteon):

    avg_loss = []
    avg_acc = []
    model.eval()

    with torch.no_grad():
        for (data,label) in iterator:
            data = Variable(data).float()
            label = Variable(label).float()
            pred = model(data)
            loss = criteon(pred,label)
            acc = binary_acc(pred, label).item()
            avg_loss.append(loss.item())
            avg_acc.append(acc)

    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc
def main():
    mat_path = "MCAD_AFQ_competition.mat"
    accs=[]
    acc_all=0
    for i in range(5):
        train_dataset = data_loader(mat_path, "./dataset_txt/train_dataset_" + str(i) + ".txt",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))
        train_iterator = DataLoader(train_dataset, batch_size=32,
                                    shuffle=True, num_workers=4)

        dev_dataset = data_loader(mat_path, "./dataset_txt/val_dataset_" + str(i) + ".txt",mode="val",
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

        dev_iterator = DataLoader(dev_dataset, batch_size=32,
                                  shuffle=False, num_workers=4)
        model = lstm()
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.001)
        criteon = nn.BCELoss()
        best_valid_acc = float('-inf')
        for epoch in range(30):
            start_time = time.time()
            train_loss, train_acc = train(model, train_iterator, optimizer, criteon)
            dev_loss, dev_acc = evaluate(model, dev_iterator, criteon)
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            if dev_acc > best_valid_acc:
                best_valid_acc = dev_acc
                torch.save(model.state_dict(), str(epoch)+'_transformer-model.pt')
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc * 100:.2f}%')
        acc_all+=best_valid_acc
        accs.append(best_valid_acc)
    print(acc_all/5)
    print(accs)

if __name__=="__main__":
    main()