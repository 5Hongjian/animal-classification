# animal-classification
a simple task of classification

# Install
Install additional packages for better process.
```c
pip install tqdm
```

# Usage
1. train model
    run `train.py` to train model. You can change the model by changing `model_name`, select from "resnet/cnn/vgg".

    I have trained 3 models, and the weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1soJiN8eX4Ao98_7kOqxRhVeojJkjrIyq?usp=sharing)

    Put the weights in the `weight` folder.
    
    All train process details is put in logs dir.

2. val model

    run `val.py` to val model. You can change the model by changing `model_name`, select from "resnet/cnn/vgg".

    The Accuracy result is below:
    ```c
    cnn : 0.0111
    vgg : 0.7907
    resnet : 0.8667
    ```
    The result of SimpleCNN created by gpt4 is so low, maybe the simple model can't be used to complex images.

    To update the result, you can use the TTA method, the result is below:
    ```
    TTA Accuracy (Transform: tta_horizontal_flip): 0.8694
    TTA Accuracy (Transform: tta_vertical_flip): 0.6917
    TTA Accuracy (Transform: tta_rotation_90): 0.6963
    TTA Accuracy (Transform: tta_rotation_180): 0.6759
    TTA Accuracy (Transform: tta_rotation_270): 0.7148
    TTA Accuracy (Transform: tta_color_jitter): 0.3731
    ```
    It shows that the tta_horizontal_flip is effective, although the improvement is not significant.
    
    All misclassified images can be found in the img_save folder.
