import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import boto3
import copy
import argparse
import os
import tarfile
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#rom torch_snippets import Report
#from torch_snippets import *

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion):
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects.double() / len(test_loader)
    


def train(model, train_loader, validation_loader, criterion, optimizer):
    epochs=5
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    #log = Report(epochs)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for pos,(inputs, labels) in enumerate(image_dataset[phase]):
                tot=len(image_dataset[phase])
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    #log.record(pos=(pos+1)/tot, train_loss=loss, end='\r') # impersistent data
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
                with torch.no_grad():
                    for pos,(inputs, labels) in enumerate(image_dataset[phase]):
                        tot=len(image_dataset[phase])
                        outputs = model(inputs)
                        valid_loss = criterion(outputs, labels)
                        #log.record(pos=(pos+1)/tot, valid_loss=valid_loss, end='\r') # impersistent data

        if loss_counter==1:
            break
        if epoch==0:
            break
    return model
    
def net(pretrained = False):
    model = models.resnet50(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

batch_size=2
learning_rate=1e-4
train_loader, test_loader, validation_loader=create_data_loaders('dogImages',batch_size)
# Get trained Model from s3 
model_downloaded = os.path.exists("./Downloaded_Models/model.pth")
if(!model_downloaded):
    s3 = boto3.client('s3')
    with open('model.tar.gz', 'wb') as f:
        s3.download_fileobj('sagemaker-us-east-1-666074078756', 'dog-pytorch-2023-03-23-02-40-18-342/output/model.tar.gz', f)

model_downloaded = os.path.exists("./Downloaded_Models/model.pth")
assert model_downloaded == True, "Model was not Downloaded"

extracted_model_exists = os.path.exists("./Downloaded_Models/model.pth")
if(!extracted_model_exists):
    # open file
    file = tarfile.open('model.tar.gz')
    # extracting file
    file.extractall('./Downloaded_Models')
    file.close()
    
extracted_model_exists = os.path.exists("./Downloaded_Models/model.pth")
assert extracted_model_exists == True, "Model was not Extracted"

model=net(pretrained = False)
model.load_state_dict(torch.load("./Downloaded_Models/model.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

logger.info("Starting Model Training")
model=train(model, train_loader, validation_loader, criterion, optimizer)
torch.save(model.state_dict(), 'TrainedModels/model.pth')
print('saved')

