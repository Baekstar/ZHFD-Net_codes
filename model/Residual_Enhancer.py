import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_activation=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.use_activation = use_activation
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.use_activation:
            x = self.activation(x)
        return x

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvLayer(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1))
        self.conv1x1 = ConvLayer(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(outputs, 1))
            outputs.append(out)
        out = self.conv1x1(torch.cat(outputs, 1))
        return out + x  # Residual connection

class DFREModule(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DFREModule, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth_rate, num_layers)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_rate, num_layers)
        self.final_conv = ConvLayer(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.final_conv(out)
        return out

# Example usage:
if __name__ == '__main__':
    input_tensor = torch.rand((1, 64, 256, 256))  # Example tensor of shape (batch_size, channels, height, width)
    dfre = DFREModule(64)  # Initialize DFRE with 64 input channels
    output = dfre(input_tensor)
    print(output.shape)  # Output tensor shape



class DualDepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualDepthwiseSeparableConvModule, self).__init__()
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.depthwise5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=out_channels)
        self.depthwise7 = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=out_channels)
        self.pointwise2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.pointwise1(x))
        x = self.relu(self.depthwise5(x))
        x = self.relu(self.depthwise7(x))
        x = self.pointwise2(x)
        x += residual
        x = self.relu(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size, in_channels, height, width = 1, 32, 224, 224
    dummy_input = torch.randn(batch_size, in_channels, height, width)
    model = DualDepthwiseSeparableConvModule(in_channels, in_channels)
    output = model(dummy_input)
    print(output.shape)