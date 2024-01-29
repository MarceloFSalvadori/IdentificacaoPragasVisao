import os

import time

import splitfolders

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from PlantDiseaseClassificationUtils import showImagesFolder
from torch.utils.data import DataLoader

#Check if CUDA API is available and if GPU is being used instead of CPU.
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.device(0))
#print(torch.cuda.get_device_name(0))

Training_Dataset_Path = "C:\\Users\\marce\\OneDrive\\Área de Trabalho\\DataFullSplitted\\train"
Validation_Dataset_Path = "C:\\Users\\marce\\OneDrive\\Área de Trabalho\\DataFullSplitted\\val"
Testing_Dataset_Path = "C:\\Users\\marce\\OneDrive\\Área de Trabalho\\DataFullSplitted\\test"

#augmentation_Dataset_Path = "C:\\Users\\marce\\OneDrive\\Área de Trabalho\\DataAug\\aug"

Validation_Losses = []
Training_Losses = []

Validation_accuracy = []
Training_accuracy = []
Testing_accuracy = []


#splitfolders.ratio('C:\\Users\\marce\\OneDrive\\Área de Trabalho\\DataFull', output='C:\\Users\\marce\\OneDrive\\Área de Trabalho\\DataFullSplitted', seed=1337, ratio=(.6, .20, .20), group_prefix=None, move=False)

#print(os.listdir("C:\\Users\\marce\\OneDrive\\Área de Trabalho\\TCC1\\Implementation\\Dataset\\Banana\\Training"))

#Defines
Batch_Size = 16


Augment_Data = True
# Transformation for the training dataset

 # Transformation for the validation dataset    
Validation_Transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
    ])

    # Transformation for the testing dataset
Testing_Transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

if (Augment_Data == False):
    Training_Transform = transforms.Compose([
        transforms.Resize((300,300)),  #Resize all images to be the same size
        transforms.ToTensor(),
        
    ])
        
else:
    Training_Transform = transforms.Compose([
        #transforms.Resize((300,300)),  #Resize all images to be the same size
        transforms.RandomHorizontalFlip(p = 0.5), #Randomly horizontaly flips images with a 50% probability rate
        #transforms.ColorJitter(saturation=[0,25]),
        #transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,2),hue=(-0.1,0.1)),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 45),
        transforms.RandomGrayscale(p = 0.05), #Kept low due to color information(specifically green) being valuable during training
        transforms.ToTensor(),
        
    ])
        

#Set up the training, validation and testing datasets. Receives the path for the images and applies the transformation previously defined
Training_Dataset = torchvision.datasets.ImageFolder(root= Training_Dataset_Path, transform= Training_Transform) 
Validation_Dataset = torchvision.datasets.ImageFolder(root= Validation_Dataset_Path, transform= Validation_Transform)
Testing_Dataset = torchvision.datasets.ImageFolder(root=Testing_Dataset_Path, transform=Testing_Transform)

#augmentation_Dataset = torchvision.datasets.ImageFolder(root=augmentation_Dataset_Path, transform=Training_Transform)

#Create iterable dataset with DataLoader. Batch size is the size of each batch created from the dataset. Shuffle is on so every time the data is loaded, the samples are shuffled.

Training_Loader = DataLoader(Training_Dataset, batch_size = Batch_Size, shuffle = True)
Validation_Loader = DataLoader(Validation_Dataset, batch_size = Batch_Size, shuffle = True)
Testing_Loader = DataLoader(Testing_Dataset, batch_size=Batch_Size, shuffle=True)


CNN_model = models.resnet18(weights = 'DEFAULT') #Defines the model used for the CNN training. ResNet-18 is a convolutional neural network that is 18 layers deep.
Number_Features = CNN_model.fc.in_features
Number_Classes = 2 #Defines the number of classes (categories of classification) used in the neural network. In this case, we have Healthy/Unhealthy so this variable is set to 2.
CNN_model.fc = nn.Linear(Number_Features , Number_Classes)
CNN_to_model = CNN_model.to('cuda:0') # Use the first (and only) avaialble GPU for the CNN model.

Loss_Function = nn.CrossEntropyLoss() # Loss function to determine the error between the prediction output and the provided target value.

Optimize = optim.SGD(CNN_model.parameters(), lr = 0.002, momentum = 0.9, weight_decay = 0.003)

def Evaluate_Model(CNN_Model, Data_Validation_Loader,Loss_Function,dataset:int):
    CNN_Model.eval()
    
    Images_Correct_Epoch = 0
    Total_Correct = 0
    Total = 0
    Current_Loss = 0
    
    torch.device("cuda:0")
    
    with torch.no_grad(): #In order not to compute the backward over the validation set
        for data in Data_Validation_Loader:
            Images, Labels = data
            Images = Images.to("cuda:0")
            Labels = Labels.to("cuda:0")
            Total += Labels.size(0) # Keep Track of total number of images
            
            Outputs = CNN_Model(Images)
            
            _, Predicted = torch.max(Outputs.data , 1)


            if dataset == 0:
                Loss = Loss_Function(Outputs,Labels)
                Current_Loss += Loss.item()
            
            
            
            #print(Predicted)
            #print(Labels)
            #print("######################")

            
            Images_Correct_Epoch += (Predicted == Labels).sum().item()
    
    if dataset == 0:
        Epoch_Loss = Current_Loss/len(Data_Validation_Loader)
        Validation_Losses.append(Epoch_Loss)
        print("Validation Loss is : %f", Epoch_Loss)

    Epoch_Accuracy = (100*Images_Correct_Epoch)/Total
    
    
    if dataset == 0:
        print( "Validation Dataset : %d images out of %d correctly (%.3f%%)." % (Images_Correct_Epoch, Total, Epoch_Accuracy))
    else:
        print( "Testing Dataset : %d images out of %d correctly (%.3f%%)." % (Images_Correct_Epoch, Total, Epoch_Accuracy))       
    
    return Epoch_Accuracy
            

def CNN_Training (CNN_Model, Data_Training_Loader, Data_Validation_Loader,Data_Testing_Loader, Optimizer, Loss_Function, Number_Epochs):
    
    #Timer to measure average epoch duration
    start = time.time()
    
    performance_Counter = 0

    #Keep track of the best model in order to save it
    best_accuracy = 0
    Epoch_num = 0

    torch.device("cuda:0")
    
    for epoch in range (Number_Epochs):
        print("Epoch Number %d" % (epoch + 1))
        CNN_model.train() # Inform the model that the model is being trained, differentiating training from validation/evaluation
        Current_Correct = 0
        Current_Loss = 0
        Total = 0
        
        for data in Data_Training_Loader:
            Images, Labels = data
            Images = Images.to("cuda:0")
            Labels = Labels.to("cuda:0")
            Total += Labels.size(0) # Keep Track of total number of images
            
            Optimizer.zero_grad()
            
            Outputs = CNN_Model(Images)
            
            _, Predicted = torch.max(Outputs.data,1)
            Loss = Loss_Function(Outputs,Labels)
            Loss.backward()
            
            Optimizer.step()
            Current_Loss += Loss.item()
            Current_Correct += (Labels == Predicted).sum().item()
        
        Epoch_Loss = Current_Loss/len(Data_Training_Loader)  #Epoch loss is equal to the Current Loss divided by the number of batches in the Training Loader
        Epoch_Accuracy = (100*Current_Correct)/Total
        
        Training_accuracy.append(Epoch_Accuracy)
        Training_Losses.append (Epoch_Loss)


        print( "Training Datase: %d Images out of %d correctly (%.3f%%). Epoch loss is : %.3f " % (Current_Correct, Total, Epoch_Accuracy, Epoch_Loss))
        
        #Validation evaluation
        current_accuracy_validation = Evaluate_Model(CNN_Model, Data_Validation_Loader,Loss_Function,0)
        Validation_accuracy.append(current_accuracy_validation)

        #Testing evaluation
        current_accuracy_testing = Evaluate_Model(CNN_Model, Data_Testing_Loader,Loss_Function,1)
        Testing_accuracy.append(current_accuracy_testing)

        if (current_accuracy_testing > best_accuracy):
            Epoch_num = epoch
            best_accuracy = current_accuracy_testing
            torch.save(CNN_Model.state_dict(),'bestModelCheckpointFinal.pht')
            
        if epoch > 1 :
            
            if Validation_Losses[epoch] > Validation_Losses[epoch - 1]:
                
                performance_Counter += 1
                
            else:
                
                performance_Counter = 0
        
        if performance_Counter == 3:
            break                


    print("Best testing accuracy, at epoch %d , is : %f" % (Epoch_num, best_accuracy))    
    print("Finished")

    end = time.time()
    totalTimeMin = (end - start)/60
    print(totalTimeMin)

    return CNN_Model        
    
CNN_Training(CNN_model, Training_Loader, Validation_Loader,Testing_Loader, Optimize, Loss_Function, 50)


plt.figure(figsize=(10,5))
plt.title("Accuracy")
plt.plot(Training_accuracy,label="Training Accuracy")
plt.plot(Validation_accuracy,label="Validation Accuracy")
plt.plot(Testing_accuracy,label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.figure(figsize=(10,5))
plt.title("Loss")
plt.plot(Training_Losses,label="Training Loss")
plt.plot(Validation_Losses,label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#showImagesFolder(augmentation_Dataset,20)
