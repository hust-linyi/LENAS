import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()
        self.down_1 = SingleConv(in_ch, list_ch[1], kernel_size=5, stride=1, padding=2)
        self.encoder_1 = nn.Sequential(
            SingleConv(list_ch[1], list_ch[1], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[1], list_ch[1], kernel_size=5, stride=1, padding=2)
        )
        self.down_2 = SingleConv(list_ch[1], list_ch[2], kernel_size=5, stride=2, padding=2)
        self.encoder_2 = nn.Sequential(
            SingleConv(list_ch[2], list_ch[2], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[2], list_ch[2], kernel_size=5, stride=1, padding=2)
        )
        self.down_3 = SingleConv(list_ch[2], list_ch[3], kernel_size=5, stride=2, padding=2)
        self.encoder_3 = nn.Sequential(
            SingleConv(list_ch[3], list_ch[3], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[3], list_ch[3], kernel_size=5, stride=1, padding=2)
        )
        self.down_4 = SingleConv(list_ch[3], list_ch[4], kernel_size=5, stride=2, padding=2)
        self.encoder_4 = nn.Sequential(
            SingleConv(list_ch[4], list_ch[4], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[4], list_ch[4], kernel_size=5, stride=1, padding=2)
        )
        self.down_5 = SingleConv(list_ch[4], list_ch[5], kernel_size=5, stride=2, padding=2)
        self.encoder_5 = nn.Sequential(
            SingleConv(list_ch[5], list_ch[5], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[5], list_ch[5], kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        out_encoder_1 = self.down_1(x)
        out_encoder_1 = sum(out_encoder_1, self.encoder_1(out_encoder_1))
        out_encoder_2 = self.down_2(out_encoder_1)
        out_encoder_2 = sum(out_encoder_2, self.encoder_2(out_encoder_2))
        out_encoder_3 = self.down_3(out_encoder_2)
        out_encoder_3 = sum(out_encoder_3, self.encoder_3(out_encoder_3))
        out_encoder_4 = self.down_4(out_encoder_3)
        out_encoder_4 = sum(out_encoder_4, self.encoder_4(out_encoder_4))
        out_encoder_5 = self.down_5(out_encoder_4)
        out_encoder_5 = sum(out_encoder_5, self.encoder_5(out_encoder_5))

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]


class Decoder(nn.Module):
    def __init__(self, list_ch):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[4], list_ch[4], kernel_size=5, stride=1, padding=2)
        )
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[3], list_ch[3], kernel_size=5, stride=1, padding=2)
        )
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[2], list_ch[2], kernel_size=5, stride=1, padding=2)
        )
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=5, stride=1, padding=2),
            SingleConv(list_ch[1], list_ch[1], kernel_size=5, stride=1, padding=2)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        out_decoder_up4 = self.upconv_4(out_encoder_5)
        out_decoder_4 = self.decoder_conv_4(torch.cat((out_decoder_up4, out_encoder_4), dim=1))
        out_decoder_up3 = self.upconv_3(sum(out_decoder_up4, out_decoder_4))
        out_decoder_3 = self.decoder_conv_3(torch.cat((out_decoder_up3, out_encoder_3), dim=1))
        out_decoder_up2 = self.upconv_2(sum(out_decoder_up3, out_decoder_3))
        out_decoder_2 = self.decoder_conv_2(torch.cat((out_decoder_up2, out_encoder_2), dim=1))
        out_decoder_up1 = self.upconv_1(sum(out_decoder_up2, out_decoder_2))
        out_decoder_1 = self.decoder_conv_1(torch.cat((out_decoder_up1, out_encoder_1), dim=1))

        return out_decoder_1


class BaseUNet(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(list_ch)

        # init
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B):
        super(Model, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_A)

        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)

        output_A = self.conv_out_A(out_net_A)
        return output_A


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Model().to(device)
    model = Model(in_ch=9, out_ch=1,
                                list_ch_A=[-1, 16, 32, 64, 128, 256],
                                list_ch_B=[-1, 32, 64, 128, 256, 512]).to(device)

    input = torch.randn(1, 9, 128, 128, 128) # BCDHW 
    input = input.to(device)
    out = model(input) 
    print("output.shape:", out.shape) # 4, 1, 8, 256, 256