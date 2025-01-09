import torch
import os
from torchvision import transforms as T, datasets, models
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable


def data_transforms(phase=None):
    if phase == TRAIN:

        data_T = T.Compose([

            T.Resize(size=(256, 256)),
            T.RandomRotation(degrees=(-20, +20)),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif phase == TEST:

        data_T = T.Compose([

            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return data_T


data_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray"
TEST = 'test'
TRAIN = 'train'

trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN), transform=data_transforms(TRAIN))
testset = datasets.ImageFolder(os.path.join(data_dir, TEST), transform=data_transforms(TEST))

class_names = trainset.classes
print(class_names)
print(trainset.class_to_idx)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

images, labels = iter(trainloader).next()

for i, (images, labels) in enumerate(trainloader):
    if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

class classify(nn.Module):
    def __init__(self, num_classes=2):
        super(classify, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=num_classes)

        # Feed forward function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 112 * 112)
        output = self.fc(output)

        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = classify()
# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = nn.CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

Losses = []
for i in range(4):
    running_loss = 0
    for images, labels in trainloader:

        # Changing images to cuda for gpu
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # Training pass
        # Sets the gradient to zero
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        # accumulates the loss for mini batch
        loss.backward()

        # And optimizes its weights here
        optimizer.step()
        Losses.append(loss)

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(i + 1, running_loss / len(trainloader)))

correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        img = images[i].view(1, 3, 224, 224)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))
