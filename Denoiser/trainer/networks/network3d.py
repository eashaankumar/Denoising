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
    dataset = load_data('/workspace/CNN_240p_Denoiser/val', batch_size=16, num_dataset_threads=3, data_count=10)
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