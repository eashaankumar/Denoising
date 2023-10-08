import torch
import os


class CNN_240p_Denoiser_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def DepthPoint(in_channel, out_channel, kernel_size):
            N = []
            N.append(torch.nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, groups=in_channel))
            N.append(torch.nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            return torch.nn.Sequential(*N)

        def EncoderInitBlock():
            N = []
            N.append(DepthPoint(3, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 3, (2,1,1)))
            N.append(torch.nn.ReLU())
            return torch.nn.Sequential(*N)
        
        self.block1 = EncoderInitBlock()


    def forward(self, x):
        return self.block1(x)

    def make_input_tensor(self, buffer):
        # convert grayscale to rgb
        buffer['depth'] = buffer['depth'].repeat(1, 3, 1, 1)
        buffer['k'] = buffer['k'].repeat(1, 3, 1, 1)
        # add depth dimension
        tensor = torch.cat([buffer['noisy'][:, :, None, :, :], 
                            buffer['normals'][:, :, None, :, :], 
                            buffer['depth'][:, :, None, :, :], 
                            buffer['albedo'][:, :, None, :, :], 
                            buffer['shape'][:, :, None, :, :], 
                            buffer['emission'][:, :, None, :, :], 
                            buffer['k'][:, :, None, :, :]], dim=2)
        return tensor 

class CNN_240p_Denoiser_Expanded_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def DepthPoint(in_channel, out_channel, kernel_size):
            N = []
            N.append(torch.nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, groups=in_channel))
            N.append(torch.nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            return torch.nn.Sequential(*N)

        def DepthPoint2D(in_channel, out_channel, kernel_size):
            N = []
            N.append(torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, groups=in_channel))
            N.append(torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            return torch.nn.Sequential(*N)

        def DepthPointTranspose2D(in_channel, out_channel, kernel_size, output_padding, dilation, stride):
            N = []
            N.append(torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, groups=in_channel, output_padding=output_padding, dilation=dilation, stride=stride))
            N.append(torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            return torch.nn.Sequential(*N)

        def EncoderInitBlock():
            N = []
            N.append(DepthPoint(3, 32, (3,3,3)))
            N.append(torch.nn.BatchNorm3d(32))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (3,3,3)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 128, (3,3,3)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.AvgPool3d(kernel_size=(3,3,3)))
            return torch.nn.Sequential(*N)

        def DecoderBlock():
            """
            1. Lose the depth channel
            2. Blow up image
            3. Refine image
            """
            N = []
            N.append(DepthPointTranspose2D(in_channel=128, out_channel=32, kernel_size=3, output_padding=1, dilation=(2,2), stride=3))
            N.append(torch.nn.ReLU())
            N.append(DepthPointTranspose2D(in_channel=32, out_channel=16, kernel_size=4, output_padding=0, dilation=(1,1), stride=1))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint2D(in_channel=16, out_channel=3, kernel_size=1))
            N.append(torch.nn.ReLU())
            return torch.nn.Sequential(*N)

        self.encoder = EncoderInitBlock()
        self.decoder = DecoderBlock()

    def forward(self, x):
        encoded = self.encoder(x)
        # return encoded
        return self.decoder(encoded.view(-1, 128, 78, 140))
        

    def make_input_tensor(self, buffer):
        buffer_list = [] 
        channel_dim = 1
        depth_dim = 2
        ior_rough_smooth_extCoef_metal = torch.cat([
            buffer['ior'][:, 0:1, None, :, :],
            buffer['extcoMetal'][:, 0:1, None, :, :],
            buffer['roughSmooth'][:, 0:1, None, :, :]
        ], dim=channel_dim)
        assert ior_rough_smooth_extCoef_metal.shape[channel_dim] == 3
        
        return torch.cat([
                    buffer['noisy'][:, :, None, :, :], 
                    buffer['normals'][:, :, None, :, :], 
                    buffer['depth'][:, :, None, :, :], 
                    buffer['albedo'][:, :, None, :, :], 
                    buffer['shape'][:, :, None, :, :], 
                    buffer['emission'][:, :, None, :, :], 
                    buffer['k'][:, :, None, :, :],
                    buffer['specular'][:, :, None, :, :],
                    ior_rough_smooth_extCoef_metal
                    ], dim=2)

if __name__=='__main__':
    from trainer.data import load_data
    dataset = load_data('/workspace/CNN_240p_Denoiser/val', batch_size=8, num_dataset_threads=3, data_count=10)
    model = CNN_240p_Denoiser_Expanded_3d()
    sd_size = (240,426)
    for b in dataset:
        buffers = b
        break
    X = model.make_input_tensor(buffers)
    print(X.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    model = model.to(device)

    Y = model(X)
    print(Y.shape)
    pass