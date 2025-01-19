import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, deep_supervision=False, features=[64, 128, 256, 512, 1024]):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.features = features

        # Encoder blocks
        self.encoders = nn.ModuleList([DoubleConv(in_channels, features[0])])
        for idx in range(1, len(features)):
            self.encoders.append(DoubleConv(features[idx-1], features[idx]))

        # Decoder blocks
        self.decoders = nn.ModuleDict()
        self.upsamplers = nn.ModuleDict()
        for i in range(len(features) - 1):
            for j in range(1, len(features) - i):
                key = f"x_{j}_{i + 1}"
                self.decoders[key] = DoubleConv(features[j-1] + features[j], features[j-1])
                self.upsamplers[key] = nn.ConvTranspose2d(features[j], features[j-1], kernel_size=2, stride=2)

        # Deep supervision heads
        self.heads = nn.ModuleList()
        for i in range(1, len(features)):
            self.heads.append(nn.Conv2d(features[0], num_classes, kernel_size=1))

        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
            x = self.pool(x)

        nested_outputs = {}
        for i in range(len(self.features) - 1, 0, -1):
            for j in range(1, len(self.features) - i + 1):
                if j == 1:
                    upsampled = self.upsamplers[f"x_{j}_{i}"](encoder_features[i])
                    concatenated = torch.cat([upsampled, encoder_features[i-1]], dim=1)
                else:
                    upsampled = self.upsamplers[f"x_{j}_{i}"](nested_outputs[f"x_{j-1}_{i}"])
                    concatenated = torch.cat([upsampled, encoder_features[i-j]], dim=1)
                nested_outputs[f"x_{j}_{i}"] = self.decoders[f"x_{j}_{i}"](concatenated)

        if self.deep_supervision:
            outputs = [self.heads[j-1](nested_outputs[f"x_{j}_{1}"]) for j in range(1, len(self.features))]
            return outputs
        else:
            return self.heads[-1](nested_outputs[f"x_{len(self.features)-1}_1"])

# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetPlusPlus(in_channels=3, num_classes=1, deep_supervision=True).to(device)
    x = torch.randn((1, 3, 224, 224)).to(device)
    outputs = model(x)
    print("Output shapes with deep supervision:")
    for out in outputs:
        print(out.shape)

    model = UNetPlusPlus(in_channels=3, num_classes=1, deep_supervision=False).to(device)
    outputs = model(x)
    print("Output shape without deep supervision:", outputs.shape)
