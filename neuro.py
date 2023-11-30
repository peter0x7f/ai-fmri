# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from torchvision import transforms, datasets
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# #import libs

# REBUILD_DATA = True 

# def img_format(image_data, fold_name): 
# 	img_format.IMG_SIZE=100
# 	size = img_format.IMG_SIZE
# 	img_format.ADHD = "FMRI/ADHD"
# 	img_format.Depression = "FMRI/Depression"
# 	img_format.Anxiety = "FMRI/Anxiety"
# 	img_format.Autism = "FMRI/Autism"
# 	img_format.Schitzophrenia = "FMRI/Schitzophrenia"
# 	img_format.Sclerosis = "FMRI/Sclerosis"
# 	img_format.Alzheirmers = "FMRI/Alzheirmers"
# 	#img_data = {Not_Detected: 0, Detected: 1}
# 	ADHDcount = 0
# 	Anxeitycount = 0

# 	for label in img_data:
# 		print(label)
# 		for f in tqdm(os.listdir(label)):
# 		#iterate through images
# 		 	try:
# 				path = os.path.join(label, f)
# 				#img = cv2.imread(path, "cv2. for fmris")
# 				img = cv2.resize(img, (size, size))
# 				train_data.append([np.array(img), np.eye(2)[img_data]])
# 		        if label == self.ADHD:
#                     self.ADHDcount += 1
#                 elif label == self.Anxiety:
#                     self.Anxeitycount += 1

# 			except Exception as e:
# 				pass
#     np.random.shuffle(image_data)
#     np.save(fold_name, image_data)
# 			#format images for better reading. 
# 	np.random.shuffle(self.img_format)
# 	np.save("training_data.npy", self.img_format)
# 	print('ADHD:',dogsvcats.catcount)
# 	print('Anxiety:',dogsvcats.dogcount)

# if REBUILD_DATA:
#     dogsvcats = DogsVSCats()
#     dogsvcats.make_training_data()
# X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
# X = X/255.0
# y = torch.Tensor([i[1] for i in training_data])
# plt.imshow(X[0], cmap="gray")
# print(y[0])
# img_format()#activating global variables

# def train_data():
# 	#detect if illness is shown in fmri with certainty from 0.00 to 1.00



# class ADHD_brain:
# 	ADHD="path/"
# 	ADHD_data=[]
# 	#use fmri in python to take dataset, image it, then save it. 

# class Depression_brain:
# 	Depression="path/"
# 	Depression_data=[]

# class Anxiety_brain:
# 	Anxiety="path/"
# 	Anxiety_data=[]

# class Autism:
# 	Autism="path/"
# 	Autism_data=[]

# class Schitzophrenia:
# 	Schitzophrenia="path/"
# 	Schitzophrenia_data=[]

# class Sclerosis:
# 	Sclerosis="path/"
# 	Sclerosis_data=[]

# class Alzheirmers:
# 	Alzheirmers="path/"
# 	Alzheirmers_data=[]

# class Gui:
# 	#visual model for program. 
# 	#show statics of accuracy, show visual of mental illness, make button to upload photo.
import os 
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class BrainModel(nn.Module):
    def __init__(self):
        super(BrainModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.25)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the training data
train_data = # Load the fMRI images of brains with mental disorders
train_labels = # Load the labels for the training data

# Convert the data to PyTorch tensors
train_data = torch.Tensor(train_data)
train_labels = torch.LongTensor(train_labels)

# Create the model
model = BrainModel()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')