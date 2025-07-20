
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prediction_image(img):
    class channelAttention(nn.Module):
        def __init__(self, input_feature, ratio=8):
            super(channelAttention,self).__init__()

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(input_feature, input_feature//ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(input_feature//ratio, input_feature, 1, bias=False),
                #nn.Sigmoid()
            )

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            output = avg_out + max_out
            #return output
            return self.sigmoid(output)


    #........................................................................................
    class spatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super(spatialAttention, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(2,8,kernel_size, padding=kernel_size//2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(8,1, kernel_size=1)
            )

            self.sigmoid = nn.Sigmoid()


        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out = torch.max(x, dim=1, keepdim=True)[0]
            x_concat = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(x_concat)

            return self.sigmoid(out) * x


    # Full CBAM Block...................................................................
    class CBAM(nn.Module):
        def __init__(self, channels, ratio=8, kernel_size=7):
            super(CBAM, self).__init__()
            self.ca = channelAttention(channels, ratio)
            self.sa = spatialAttention(kernel_size)

        def forward(self, x):
            x = x * self.ca(x)
            x = x * self.sa(x)
            return x

    class CBAM_CNN(nn.Module):
        def __init__(self):
            super(CBAM_CNN, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                CBAM(32),
                nn.MaxPool2d(2)
            )
            self.cbam1 = CBAM(32)
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                CBAM(64),
                nn.MaxPool2d(2)
            )
            self.cbam2 = CBAM(64)

            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                CBAM(128),
                nn.MaxPool2d(2)
            )
            self.cbam3 = CBAM(128)

            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                CBAM(256),
                nn.MaxPool2d(2)
            )
            # Now compute flatten size
            dummy_input = torch.zeros(1, 3, 224, 224)
            x = self.conv1(dummy_input)
            x = self.cbam1(x)
            x = self.conv2(x)
            x = self.cbam2(x)
            x = self.conv3(x)
            x = self.cbam3(x)
            x = self.conv4(x)
            flatten_size = x.view(1, -1).size(1)

            self.fc = nn.Linear(flatten_size, 38)  # Adjust size based on pooling and input

        def forward(self, x):
            x = self.conv1(x)   # [B, 32, H/2, W/2]
            x = self.conv2(x)   # [B, 64, H/4, W/4]
            x = self.conv3(x)   # [B, 128, H/8, W/8]
            x = self.conv4(x)   # [B, 256, 1, 1]
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc(x)
            return x


    model = CBAM_CNN()

    model.load_state_dict(torch.load('crops_cbam_Model.pth', weights_only=True))
    model = model.to(device)

    model.eval()
    # 5. Make prediction
    with torch.no_grad():
        output = model(img)
        predicted = output.argmax(dim=1).item()
        # _, predicted = torch.max(output, 1)

    return predicted
