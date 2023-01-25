import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")  # some images trigger annoying "corrupted data" warnings...

# SOURCE: https://www.pluralsight.com/guides/introduction-to-resnet
# SOURCE: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


if __name__ == '__main__':
    batch_size = 64
    learning_rate = 1e-2
    n_epochs = 50
    NEW_MODEL = True

    transforms = transforms.Compose([
        transforms.Resize((80, 80)),  # 256
        # transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # data_set = datasets.ImageFolder(root="PetImages", transform=transforms)
    data_set = datasets.ImageFolder(root="images/dice/all_types", transform=transforms)

    # train_dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    train_length = int(0.80 * len(data_set))
    test_length = len(data_set) - train_length
    train_data, test_data = torch.utils.data.random_split(data_set, [train_length, test_length])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:big_red cuda:dice....etc.
        print("Running on the GPU")
    else:
        # this might take a very long time w/o a gpu
        device = torch.device("cpu")
        print("Running on the CPU")

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Number of classes:", len(data_set.classes))
    print("Baseline accuracy:", 100 / len(data_set.classes))
    print(data_set.class_to_idx)

    if NEW_MODEL:
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, len(data_set.classes))
    else:
        net = models.resnet18()
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, len(data_set.classes))
        net.load_state_dict(torch.load('resnet.pt'))

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    valid_loss_min = np.Inf
    # these lists hold one value per epoch
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'\nEpoch {epoch}')
        net.train()
        time.sleep(0.1)
        for data_, target_ in train_dataloader:  # tqdm
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / len(train_dataloader))
        print(f'train-loss: {(running_loss / len(train_dataloader)):.4f}, train-acc: {(100 * correct / total):.4f}\n')

        batch_loss = 0
        correct_t = 0
        total_t = 0
        with torch.no_grad():
            net.eval()
            time.sleep(0.1)
            for data_t, target_t in test_dataloader:  # tqdm
                data_t, target_t = data_t.to(device), target_t.to(device)

                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)

                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)

            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(test_dataloader))
            print(f'validation loss: {batch_loss / len(test_dataloader):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            if batch_loss < valid_loss_min:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'all_cubes.pt')
                print('Improvement-Detected, saved model')
