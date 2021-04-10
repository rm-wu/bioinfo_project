import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

PADDING = 0


class DoubleConv(nn.Module):
    """
    Module for UNet's double convolution operation.
    -----------------------------------------------
        Note:   The actual implementation makes use of nn.BatchNorm
                and a different padding wrt to the original paper
    """
    def __init__(self, input_ch, output_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch,
                      kernel_size=3, stride=1, padding=PADDING, bias=False),
            # TODO: bias=False, because here I'm using nn.BatchNorm
            #  and while the paper the padding is 0 here it is 1 (same output_ch)
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


def center_crop(tensor_1, tensor_2):
    """
    Crops the center elements of tensor_1 to match the shape of tensor_2
    """
    dim = tensor_1.shape[-1] - tensor_2.shape[-1]
    if dim < 0:
        raise Exception("tensor_2 must be smaller than tensor_1")
    if dim % 2 == 0:
        dim = dim // 2
        tensor = tensor_1[:, :, dim:-dim, dim:-dim]
    else:
        # TODO: update it??
        dim = dim // 2
        tensor = tensor_1[:, :, dim+1:-dim, dim+1:-dim]
    assert tensor.shape == tensor_2.shape
    return tensor


class UNet(nn.Module):
    """
    Implementation of the UNet architecture
    """
    def __init__(self, input_ch=3, output_ch=1, feat_ch=None):
        super(UNet, self).__init__()

        if feat_ch is None:
            feat_ch = [64, 128, 256, 512]

        self.down_stream = nn.ModuleList()
        self.up_stream_conv = nn.ModuleList()
        self.up_stream_transp = nn.ModuleList()
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # UNet down stream
        for feat in feat_ch:
            self.down_stream.append(
                DoubleConv(input_ch, feat)
            )
            input_ch = feat

        self.bottom = DoubleConv(feat_ch[-1], feat_ch[-1] * 2)  # 512 --> 1024

        # TODO: Use ConvTranspose or UpScale?
        # UNet up stream
        # Note: in_channels are always halved
        for feat in reversed(feat_ch):
            self.up_stream_transp.append(
                nn.ConvTranspose2d(feat * 2, feat,
                                   kernel_size=2, stride=2)
            )
            self.up_stream_conv.append(
                DoubleConv(feat*2, feat)
            )
        # Output
        self.output_layer = nn.Conv2d(feat_ch[0], output_ch, kernel_size=1)

    def forward(self, x):
        intermediate_steps = []

        # Down Stream
        for down_layer in self.down_stream:
            x = down_layer(x)
            print(x.shape)
            intermediate_steps.append(x)
            x = self.pool_layer(x)
            print(x.shape)
        print(f"intermediate_steps:{len(intermediate_steps)}")
        x = self.bottom(x)
        print(x.shape)
        intermediate_steps.reverse()

        # Up Stream
        for feat, conv, transp in zip(intermediate_steps,
                                      self.up_stream_conv,
                                      self.up_stream_transp):
            x = transp(x)
            feat_c = center_crop(feat, x)
            # TODO: problems when the difference between the dimensions is odd
            print(f"x: {x.shape}\tfeat_c : {feat_c.shape}")

            x = torch.cat((x, feat_c), 1)
            print(x.shape)
            x = conv(x)
            print(x.shape)

        return self.output_layer(x)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



if __name__ == '__main__':
    x = torch.rand((3, 1, 668, 668))  # The input tensor must be divisible by 2 * #(down_layers)
    model = UNet(input_ch=1, output_ch=1)
    preds = model(x)
    print(x.shape, preds.shape)
