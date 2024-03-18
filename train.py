import time
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import load_data
from model import initialize_model


def main():
    data_dir = 'dataset/90animals/'
    model_name = 'cnn'  # you can choose resnet/cnn/vgg
    feature_extract = True
    num_epochs = 20
    num_classes = 90
    batch_size = 16
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("use device:{}".format(device))
    image_datasets = load_data(data_dir)
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
    model, input_size = initialize_model(model_name, num_classes, feature_extract)
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    mylog = open('logs/' + model_name + '.log', 'w')
    # start train
    s_time = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=mylog)
        print('-' * 10, file=mylog)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                optimizer.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), file=mylog)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            mylog.flush()
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({'state_dict': model.state_dict()}, "weight/" + model_name + "_best.pth")
    time_elapsed = time.time() - s_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=mylog)
    print('Best val Acc: {:4f}'.format(best_acc), file=mylog)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    mylog.close()


if __name__ == '__main__':
    main()
