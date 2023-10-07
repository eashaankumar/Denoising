import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import os
from tqdm import tqdm
import threading

def load_thread(iter, data, desiredKeys, file_type, trans, dataset_path, data_range):
    data.extend([0] * (len(iter[0])))
    assert (len(data) == len(iter[0]))
    with tqdm(iter[0], f"{iter[1]} {len(data)}") as tepoch:
        name = dataset_path.split(os.sep)[-2:]
        tepoch.set_description(f"{name[0]}-{name[1]}-{data_range}")
        for i, subdir in enumerate(tepoch):
            subdirPath = os.path.join(dataset_path, subdir)
            images = os.listdir(subdirPath)
            tepoch.set_postfix(image=f"{images}")
            data[i] = (subdirPath,images,subdir)
            pass


class CNN_SD_Denoiser_Dataset(Dataset):

   
    def __init__(self, dataset_path: str, file_type, num_dataset_threads=3, data_count=-1):
        # import the modules
        import os
        from os import listdir
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.data = []
        self.desiredKeys = ['albedo', 'converged', 'depth', 'emission', 'extcoMetal', 'ior', 'noisy', 'normals', 'roughSmooth', 'shape', 'k', 'specular']
        
        self.threads = []
        self.threadsData = []
        self.file_type = file_type

        print(f"Loading {dataset_path} on {num_dataset_threads} threads")

        total_data = os.listdir(dataset_path)
        if (data_count > 0 and data_count < len(total_data)):
            total_data = total_data[:data_count]
        partition_size = int(len(total_data) / num_dataset_threads)
        start = 0
        for i in range(num_dataset_threads):
            tData = []
            end = start+partition_size
            if (i == num_dataset_threads-1):
                end = len(total_data)
            t = threading.Thread(target=load_thread, args=(
                (total_data[start:end], "datapoints"), tData, self.desiredKeys, file_type, self.trans, dataset_path, (start, end))
            )
            t.start()
            self.threads.append(t)
            self.threadsData.append(tData)
            start += partition_size
            print("Thread started")
        
        for i in range(num_dataset_threads):
            self.threads[i].join()

        for i in range(num_dataset_threads):
            self.data.extend(self.threadsData[i])
        
        assert (len(self.data) == len(total_data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        buffers = {}
        subdirPath,images,subdir = self.data[index]
        for image in images:
            img_path = os.path.join(subdirPath, image)
             
            if (image.endswith(f'noisy-{subdir}.{self.file_type}')):
                buffers['noisy'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'albedo-{subdir}.{self.file_type}')):
                buffers['albedo'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'converged-{subdir}.{self.file_type}')):
                buffers['converged'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'depth-{subdir}.{self.file_type}')):
                buffers['depth'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'emission-{subdir}.{self.file_type}')):
                buffers['emission'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'normals-{subdir}.{self.file_type}')):
                buffers['normals'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'shape-{subdir}.{self.file_type}')):
                buffers['shape'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'k-{subdir}.{self.file_type}')):
                buffers['k'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'extcoMetal-{subdir}.{self.file_type}')):
                buffers['extcoMetal'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'ior-{subdir}.{self.file_type}')):
                buffers['ior'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'roughSmooth-{subdir}.{self.file_type}')):
                buffers['roughSmooth'] = self.trans(Image.open(img_path))
            elif (image.endswith(f'specular-{subdir}.{self.file_type}')):
                buffers['specular'] = self.trans(Image.open(img_path))
            pass
        keys = buffers.keys()
        assert len(keys) == len(self.desiredKeys)
        for key in keys:
            assert key in self.desiredKeys

        return buffers

def load_data(dataset_path, num_workers=0, batch_size=128, num_dataset_threads=3, data_count=-1):
    dataset = CNN_SD_Denoiser_Dataset(dataset_path, file_type='jpg', num_dataset_threads=num_dataset_threads, data_count=data_count)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
    
if __name__ == '__main__':
    import argparse
    print("Showing dataset")
    parser = argparse.ArgumentParser("trainer.data")
    parser.add_argument("--path", help="path to data directory", type=str)
    parser.add_argument("--type", help="image file extension (png, jpeg, etc...)", default='png')
    args = parser.parse_args()
    dataset = CNN_SD_Denoiser_Dataset(dataset_path= args.path, file_type=args.type)
    print(dataset[0])