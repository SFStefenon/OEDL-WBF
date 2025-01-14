# Optimized ensemble of deep learning models based on weighted boxes fusion (OEDL-WBF)

The proposed OEDL-WBF presented in this repository employs the Optuna framework based on a tree-structured Parzen estimator for hyperparameter optimization, to find the best setup for the object detection model. Subsequently, the hypertuned model is trained using the optimal hyperparameters and applying WBF for more accurate bounding box predictions. In the outcome, the insulators are identified with an optimized prediction. Interpretative results are presented, and calculated as a function of how the network learned the pattern that results in the prediction.

---

## To perform multi-criteria optimization the **Optuna framework** is considered

There are two ways to compute the Optuna, the first is locally on your machine (cluster or PC) and the second is using Google Colab.

> If you want to use a local machine, you can follow this Python-based [algorithm](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/yolov8_insulator_exp1.py). Using a Cluster the study is gonna be saved and you can evaluate latter using [Colab](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/Experiment_Results/Optuna_Results.ipynb).

> If you decide to use Colab, please go ahead and try it on using this [notebook](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Google_Colab_Computing/YOLOv8_Optuna.ipynb)! You will be asked to confirm your access to the drive where the data are saved.

OBS: Since the analysis is using a deep learning-based model, depending on your dataset, a high processing time will be required for the model to be trained (considering the defined number of epochs). The file that calls the dataset must be in the same main folder of the model.

---

## For architecture optimization the **weighted box fusion** is used

To apply the WBF you will need to train the model several times and save the weights. After that the WFB ensemble the YOLO's outputs to have a better prediction. You can find the algorithm [here](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Weighted_Box_Fusion/WBF_yolo.ipynb).

---

## For interpretability the **EigenCam** is applied

Examples of the results of the method can be found [here](https://github.com/SFStefenon/WBF-HE-YOLO/tree/main/EigenCam).

---

## Compute YOLOv8 in your machine

The first step is to download the YOLOv8. I recommend doing that from the official developer [Ultralytics](https://github.com/ultralytics/ultralytics).
This version is based on PyTorch, and it is available for your machine or Google Colab.

The second step in computing YOLO in your machine is to create the environment. Follow the example of how to do it:

```
# Enter in the folder of your project
cd ~~~ 

# Check the environments available
conda env list

# Create a new environment
conda create --name yolo-env python=3.9

# Activate the environment to install packages
conda activate yolo-env

# Install the requirements
pip install -r requirements.txt

# If something goes wrong and you need to remove the environment
conda env remove -n yolo-env
```

The third step is to organize your dataset.

```
# Example of how to upload your dataset in a Cluster
scp -r C:/Users/user_name/Desktop/dataset/ cluster:/home/user_name/dataset/
```

### Organize Your Dataset

Depending on the version of YOLO you will need to organize your data differently.

For YOLOv5, YOLOv7, and YOLOv8.
```    
dataset/train/images
dataset/train/labels
dataset/valid/images
dataset/valid/labels
```

Works especially for YOLOv6.
```    
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

When you have organized your data, you will need to set the model to load it.
Here you can call the 'data/mydata.yaml'

This file is going to be in the `data` folder inside the YOLO model.

Depending on how you decided to organize the data, the `mydata.yaml` may look like this:
```
train: ../dataset/train/images
val: ../dataset/valid/images
```

The model will load the labels automatically based on their names.

---

### Used Database

The model presented in this repository was evaluated using the dataset released by Dexter Lewis and Pratik Kulkarni, which can be found at [competition-insulator-defect-detection](https://dx.doi.org/10.21227/vkdw-x769) (accessed on March 25, 2023).

**We encourage you to perform comparative analyses with your dataset!**

---

### Create Your Custom Dataset

To create a custom dataset with the goal of object detection it is necessary to use image labeling software.

I recommend using the [labelImg](https://github.com/heartexlabs/labelImg), it's based on Python, so it's light and easy to use.
LabelImg is a graphical image annotation tool written in Python.

After you download the algorithm you can run the `labelImg.py`

In the `data/predefined_classes.txt` you can define the classes that you are going to use. 
This will allocate the spaces in the memory, therefore the number in the annotation will follow this order.

Later on, the classes that you have used have to match with `mydata.yaml`, which you previously created as this example:

```
nc: 4 
names: ['broken_shell', 'flashover_damage', 'good_insulator', 'insulator_string']
```

---
Thank you.

Dr. **Stefano Frizzo Stefenon**.

Regina, Canada, January 14, 2025.
