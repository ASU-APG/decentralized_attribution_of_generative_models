import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.convT1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.batch1 = nn.BatchNorm2d(ngf * 8)
        self.relu1 = nn.ReLU(True)

        self.convT2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(ngf * 4)
        self.relu2 = nn.ReLU(True)

        self.convT3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(ngf * 2)
        self.relu3 = nn.ReLU(True)

        self.convT4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.batch4 = nn.BatchNorm2d(ngf)
        self.relu4 = nn.ReLU(True)

        self.convT5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

        """
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        """

    def forward(self, input):
        x = self.convT1(input)
        x = self.batch1(x)
        x = self.relu1(x)

        x = self.convT2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        x = self.convT3(x)
        x = self.batch3(x)
        x = self.relu3(x)

        x = self.convT4(x)
        x = self.batch4(x)
        x = self.relu4(x)

        x = self.convT5(x)
        x = self.tanh(x)

        return x