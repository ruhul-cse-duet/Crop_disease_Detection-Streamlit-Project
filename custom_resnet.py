
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# for moving data to device (CPU or GPU)
def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def prediction_img(img):
    class ImageClassificationBase(nn.Module):

        def training_step(self, batch):
            images, labels = batch
            # images, labels = images.to(DEVICE), labels.to(DEVICE) # move to GPU
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            # images, labels = images.to(DEVICE), labels.to(DEVICE) # move to GPU
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    # convolution block with BatchNormalization
    def ConvBlock(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)

    # resnet architecture ........................................
    class CNN_NeuralNet(ImageClassificationBase):
        def __init__(self, in_channels, num_diseases):
            super().__init__()

            self.conv1 = ConvBlock(in_channels, 64)
            self.conv2 = ConvBlock(64, 128, pool=True)
            self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

            self.conv3 = ConvBlock(128, 256, pool=True)
            self.conv4 = ConvBlock(256, 512, pool=True)
            # self.conv5 = ConvBlock(256, 256, pool=True)
            # self.conv6 = ConvBlock(256, 512, pool=True)
            # self.conv7 = ConvBlock(512, 512, pool=True)

            self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

            # self.classifier = nn.Sequential(nn.MaxPool2d(4),
            #                                nn.Flatten(),
            #                                nn.Linear(512, num_diseases))

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Safe replacement
                nn.Flatten(),
                nn.Linear(512, num_diseases)
            )

        def forward(self, x):  # x is the loaded batch
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            # out = self.conv5(out)
            # out = self.conv6(out)
            # out = self.conv7(out)
            out = self.res2(out) + out
            out = self.classifier(out)

            return out



    #model = CNN_NeuralNet(3,38)
    model = to_device(CNN_NeuralNet(3, 38), device)

    model.load_state_dict(torch.load('resnet_Model.pth', weights_only=True))


    model.eval()
    # 5. Make prediction
    with torch.no_grad():
        #img = to_device(img.unsqueeze(0), device)
        output = model(img)
        predicted = output.argmax(dim=1).item()
        # _, predicted = torch.max(output, 1)

    return predicted
