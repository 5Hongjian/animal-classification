import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import initialize_model
from TTA_method import *
from torchvision.utils import save_image


def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def evaluate(model, dataloader, device, save_dir='img_save'):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    total = 0
    os.makedirs(save_dir, exist_ok=True)

    # Initialize a counter for misclassified images
    misclassified_count = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

        # Find misclassified images and save them
        misclassified_indices = preds != labels
        misclassified_imgs = inputs[misclassified_indices]
        misclassified_true_labels = labels[misclassified_indices]
        misclassified_preds = preds[misclassified_indices]

        for img, true_label, pred_label in zip(misclassified_imgs, misclassified_true_labels, misclassified_preds):
            file_name = f'misclassified_{misclassified_count}_True{true_label.item()}_Pred{pred_label.item()}.png'
            img_path = os.path.join(save_dir, file_name)
            save_image(img.cpu(), img_path)
            misclassified_count += 1

    accuracy = running_corrects.double() / total
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Total images: {total}, Misclassified images: {misclassified_count}')


def tta_evaluate(model, dataloader, device, tta_transforms):
    model.eval()  # Set model to evaluate mode
    best_accuracy = 0.0
    best_transform = None
    for tta_transform in tta_transforms:
        running_corrects = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Apply TTA
            tta_inputs = torch.stack([tta_transform(img) for img in inputs])
            # Forward pass with TTA
            with torch.no_grad():
                tta_outputs = model(tta_inputs)
                _, tta_preds = torch.max(tta_outputs, 1)
            running_corrects += torch.sum(tta_preds == labels.data)
            total += labels.size(0)
        accuracy = running_corrects.double() / total
        print(f'TTA Accuracy (Transform: {tta_transform.__name__}): {accuracy:.4f}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_transform = tta_transform.__name__ if tta_transform else "Original"
    print(f'Best TTA Method: {best_transform} with Accuracy: {best_accuracy:.4f}')

def main():
    data_dir = 'dataset/90animals/val'
    model_name = 'cnn'  # you can choose resnet/cnn/vgg
    weight_path = 'weight/cnn_best.pth'
    num_classes = 90
    batch_size = 160
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Data augmentation and normalization for validation
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Define TTA transforms list
    tta_transforms = [
        tta_horizontal_flip,
        tta_vertical_flip,
        tta_rotation_90,
        tta_rotation_180,
        tta_rotation_270,
        tta_color_jitter,
    ]
    # Load the data
    dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize and load the model
    model, input_size = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    model = model.to(device)
    model = load_checkpoint(weight_path, model)
    print("Loaded model success")
    # Evaluate the model
    evaluate(model, dataloader, device)
    # Evaluate the model with TTA
    tta_evaluate(model, dataloader, device, tta_transforms)


if __name__ == '__main__':
    main()
