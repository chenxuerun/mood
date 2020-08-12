import torch
import torch.nn as nn

class CAE_pytorch(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, rep_dim = 256):
        super(CAE_pytorch, self).__init__()
        # nf = 16
        nf = (16, 64, 256, 1024)
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf[0], kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf[0])
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf[0], out_channels=nf[1], kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf[1])
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf[1], out_channels=nf[2], kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf[2])
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_conv4 = nn.Conv2d(in_channels=nf[2], out_channels=nf[3], kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(num_features=nf[3])
        self.enc_act4 = nn.ReLU(inplace=True)

        # self.enc_conv5 = nn.Conv2d(in_channels=nf[3], out_channels=nf[4], kernel_size=8, stride=2, padding=0)
        # self.enc_bn5 = nn.BatchNorm2d(num_features=nf[3])
        # self.enc_act5 = nn.ReLU(inplace=True)

        # self.enc_fc = nn.Linear(nf * 4 * 16 * 16, rep_dim)
        # self.rep_act = nn.Tanh()

        # Decoder
        # self.dec_fc = nn.Linear(rep_dim, nf * 4 * 16 * 16)
        # self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 16 * 16)
        # self.dec_act0 = nn.ReLU(inplace=True)
        
        # self.dec_conv00 = nn.ConvTranspose2d(in_channels=nf[4], out_channels=nf[3], kernel_size=8, stride=2, padding=0, output_padding=0)
        # self.dec_bn00 = nn.BatchNorm2d(num_features=nf[3])
        # self.dec_act00 = nn.ReLU(inplace=True)

        self.dec_conv0 = nn.ConvTranspose2d(in_channels=nf[3], out_channels=nf[2], kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dec_bn0 = nn.BatchNorm2d(num_features=nf[2])
        self.dec_act0 = nn.ReLU(inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf[2], out_channels=nf[1], kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf[1])
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf[1], out_channels=nf[0], kernel_size=4, stride=2, padding=1, output_padding=0)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf[0])
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf[0], out_channels=out_channels, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.output_act = nn.Sigmoid()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        # x = self.enc_act5(self.enc_bn5(self.enc_conv5(x)))
        # rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return x

    def decode(self, rep):
        # x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        # x = x.view(-1, self.nf * 4, 16, 16)
        # x = self.dec_act00(self.dec_bn00(self.dec_conv00(rep)))
        x = self.dec_act0(self.dec_bn0(self.dec_conv0(rep)))
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

class CAE_pytorch_2(nn.Module):
    def __init__(self, in_channels = 1, rep_dim = 256):
        super(CAE_pytorch_2, self).__init__()
        # nf = 16
        nf = (64, 256, 512, 1024, 1024)
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf[0], kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf[0])
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf[0], out_channels=nf[1], kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf[1])
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf[1], out_channels=nf[2], kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf[2])
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_conv4 = nn.Conv2d(in_channels=nf[2], out_channels=nf[3], kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(num_features=nf[3])
        self.enc_act4 = nn.ReLU(inplace=True)

        self.enc_conv5 = nn.Conv2d(in_channels=nf[3], out_channels=nf[4], kernel_size=3, stride=2, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(num_features=nf[3])
        self.enc_act5 = nn.ReLU(inplace=True)

        # Decoder
        
        self.dec_conv00 = nn.ConvTranspose2d(in_channels=nf[4], out_channels=nf[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn00 = nn.BatchNorm2d(num_features=nf[3])
        self.dec_act00 = nn.ReLU(inplace=True)

        self.dec_conv0 = nn.ConvTranspose2d(in_channels=nf[3], out_channels=nf[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn0 = nn.BatchNorm2d(num_features=nf[2])
        self.dec_act0 = nn.ReLU(inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf[2], out_channels=nf[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf[1])
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf[1], out_channels=nf[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf[0])
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf[0], out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Sigmoid()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        x = self.enc_act5(self.enc_bn5(self.enc_conv5(x)))
        # rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return x

    def decode(self, rep):
        # x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        # x = x.view(-1, self.nf * 4, 16, 16)
        x = self.dec_act00(self.dec_bn00(self.dec_conv00(rep)))
        x = self.dec_act0(self.dec_bn0(self.dec_conv0(x)))
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
        # return torch.sigmoid(x)



class RSRAE(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super(RSRAE, self).__init__()
        nf = 64
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_fc = nn.Linear(nf * 4 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()

        # Robust Subspace Recovery
        d = 10
        self.A = nn.Parameter(torch.randn(rep_dim, d))

        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 4 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 4 *4)
        self.dec_act0 = nn.ReLU(inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        h = self.encode(x)
        hs = h @ self.A
        hr = self.A @ hs.t()
        hr = hr.t()
        rec = self.decode(hr / (torch.norm(hr, dim=1).unsqueeze(-1).repeat(1, hr.size(1)) + 1e-10))

        hh = h.detach()
        hs2 = hh @ self.A
        hr2 = self.A @ hs2.t()
        hr2 = hr2.t()

        AA = self.A.t() @ self.A
        return rec, hh, hr2, AA

class RSRAE_plus(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super(RSRAE_plus, self).__init__()
        nf = 64
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_fc = nn.Linear(nf * 4 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()

        # Robust Subspace Recovery
        d = 10
        self.A = nn.Parameter(torch.randn(rep_dim, d))

        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 4 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 4 *4)
        self.dec_act0 = nn.ReLU(inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        h = self.encode(x)
        hs = h @ self.A
        hr = self.A @ hs.t()
        hr = hr.t()
        AA = self.A.t() @ self.A
        rec = self.decode(hr / (torch.norm(hr, dim=1).unsqueeze(-1).repeat(1, hr.size(1)) + 1e-10))
        return rec, h, hr, AA

class L21Loss(nn.Module):
    def __init__(self):
        super(L21Loss, self).__init__()

    def forward(self, x, y):
        res = x - y
        res = torch.norm(res, dim=1, p=2)
        return res.sum()


class RSR1Loss(nn.Module):
    def __init__(self):
        super(RSR1Loss, self).__init__()

    def forward(self, hr, h):
        c = L21Loss()
        return c(hr, h)


class RSR2Loss(nn.Module):
    def __init__(self):
        super(RSR2Loss, self).__init__()

    def forward(self, AA):
        I = torch.ones_like(AA)
        res = AA - I
        return torch.norm(res)

# model = CAE_pytorch()
# print(model)


