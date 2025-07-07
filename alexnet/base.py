import torch
from torch import nn
from .local_response_normalization import LocalReponseNormalization

class AlexNet(nn.Module):
    def __init__(self, ):
        super(AlexNet, self).__init__()

        self.lrn = LocalReponseNormalization()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=256, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=384, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=384, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=4096)
        self.output = nn.Linear(in_features=self.fc2.out_features, out_features=1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers with ReLU activation and max pooling
        
        # First convolutional layer with stride 4
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.lrn(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)

        # Second convolutional layer
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.lrn(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        
        # Third, fourth, and fifth convolutional layers connected without any intervening
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)

        # Max pooling after the last convolutional layer
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)

        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

        return self.output(x)