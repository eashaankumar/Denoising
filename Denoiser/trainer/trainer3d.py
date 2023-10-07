from .networks.network3d import CNN_240p_Denoiser_Expanded_3d
from .data import load_data
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from inputimeout import inputimeout, TimeoutOccurred
from .utils import *

class DenoisingAutoencoderTrainer:
    def __init__(self) -> None:
        self.network = None
        self.train_data = None
        self.val_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.sd_size = (240,426)
        self.optim = None
        self.loss = None
        self.writer = None
        self.tensorboard_name = None
        self.make_input_tensor = None
        pass

    def load_model_from_path(self, path):
        model_name = path
        self.network = load_model(model_type=CNN_240p_Denoiser_Expanded_3d, model_name=model_name)
        self.make_input_tensor = self.network.make_input_tensor
        self.network = torch.nn.DataParallel(self.network, device_ids=[0, 1])
        self.network.to(self.device)
        self.__init_params()
    
    def save_model(self, path):
        model_name = path
        save_model(self.network.module, model_type=CNN_240p_Denoiser_Expanded_3d, model_name=model_name)

    def create_new_model(self):
        self.network = CNN_240p_Denoiser_Expanded_3d()
        self.make_input_tensor = self.network.make_input_tensor
        self.network = torch.nn.DataParallel(self.network, device_ids=[0, 1])
        self.network.to(self.device)
        self.__init_params()

    def release_model(self):
        self.network.to('cpu')
        self.network = None

    def __init_params(self):
        self.optim = torch.optim.Adam(self.network.parameters(), 0.001)
        self.loss = torch.nn.MSELoss()

    def load_writer(self, experiment_dir, tensorboard_name):
        self.writer = SummaryWriter(os.path.join(experiment_dir, 'tf_logs', tensorboard_name))
        self.tensorboard_name = tensorboard_name
        pass

    def __get_lr(self):
        for g in self.optim.param_groups:
            olr = g['lr']
        return olr

    def record_loss(self, train_loss, val_loss, step):
        if self.writer:
            
            self.writer.add_scalars(f'{self.tensorboard_name}/Loss/lr={self.__get_lr()}/b={self.train_data.batch_size}/length={len(self.train_data.dataset)}', {
                'Training Loss': train_loss,
                'Validation Loss': val_loss,
            }, step)
            #TODO: Add more measurements
            pass
        pass

    def record_imgs(self, tru, pred, step):
        if self.writer:
            assert len(tru.shape) == 4
            assert tru.shape == pred.shape
            final = torch.zeros(tru.shape[0] + pred.shape[0], tru.shape[1], tru.shape[2], tru.shape[3])
            j = 0
            for i in range(tru.shape[0]):
                final[j] = tru[i]
                final[j+1] = pred[i]
                j += 2
            self.writer.add_images(f'{self.tensorboard_name}/Images/lr={self.__get_lr()}/b={self.train_data.batch_size}/length={len(self.train_data.dataset)}', final, step)
        pass
    
    def load_data(self, rootpath, batch_size):
        self.train_data = load_data(os.path.join(rootpath, 'train'), batch_size=batch_size, num_workers=10,
                           num_dataset_threads=1, data_count=20000)
        os.system('CLS')
        self.val_data = load_data(os.path.join(rootpath, 'val'), batch_size=batch_size, num_workers=10,
                            num_dataset_threads=1, data_count=2000)
        os.system('CLS')
        pass

    def train(self) -> float:
        total_train_loss = 0
        self.network.train()
        count = 0
        with tqdm(iter(self.train_data), unit="batches") as trainloop:
            for i, buffers in enumerate(trainloop):
                count += 1
                trainloop.set_description(f"Train Batch {i}")
                input_tensor = self.make_input_tensor(buffers).to(self.device)
                ground_truth = buffers['converged'][:, :, None, :, :].to(self.device)
                res = self.network(input_tensor)
                input_tensor = input_tensor.detach()
                del input_tensor
                l = self.loss(res, ground_truth[:,:,0,:,:])
                ground_truth = ground_truth.detach()
                del ground_truth
                self.optim.zero_grad()
                l.backward()
                self.optim.step()
                curr_loss = l.cpu().data
                total_train_loss += curr_loss
                trainloop.set_postfix(loss=f"{curr_loss}")
        return total_train_loss / count

    def validate(self):
        total_val_loss = 0
        self.network.eval()
        debug_images = None
        count = 0
        with tqdm(iter(self.val_data), unit="batches") as valloop:
            for i, buffers in enumerate(valloop):
                count += 1
                # self.loss_plotter.update_display()
                valloop.set_description(f"Val Batch {i}")
                input_tensor = self.make_input_tensor(buffers).to(self.device)
                ground_truth = buffers['converged'][:, :, None, :, :].to(self.device)
                res = self.network(input_tensor)
                if (i == 0):
                    debug_images = ground_truth[:8, :, 0, :, :], res[:8, :, :, :]
                    pass
                input_tensor = input_tensor.detach()
                del input_tensor
                l = self.loss(res, ground_truth[:,:,0,:,:])
                ground_truth = ground_truth.detach()
                del ground_truth
                curr_loss = l.cpu().data
                total_val_loss += curr_loss
                valloop.set_postfix(loss=f"{curr_loss}")
        return total_val_loss / count, debug_images

class TrainingMenu:
    def __init__(self) -> None:
        pass

    def get_menu_options(self):
        return  f"\n\n{bcolors.BOLD}Menu:{bcolors.ENDC}\n" + \
                f"{bcolors.OKBLUE}[1]: Change lr"+ \
                f"{bcolors.ENDC}\n" + \
                f"Pick an item: "
    
    def confirm(self, timeout=5, yes='y') -> bool:
        o = inputimeout(prompt=f"Confirm [{yes}/n]", timeout=timeout)
        if (o == yes):
            return True
        return False
    
    def _handle_learning_rate(self, optim, timeout=5):
        for g in optim.param_groups:
            olr = g['lr']
            print(f'old lr: {olr}')
        new_lr = float(inputimeout(prompt='Enter new lr: ', timeout=20))
        print(f'new lr={new_lr}')
        if (not self.confirm(timeout=20)):
            return self.menu(timeout=timeout)
        for g in optim.param_groups:
            old_lr = g['lr']
            g['lr'] = new_lr
            print(f"{bcolors.OKGREEN}Old LR: {old_lr} New LR: {new_lr}{bcolors.ENDC}")

    def menu(self, optim, timeout=5):
        try:
            menu = inputimeout(prompt=self.get_menu_options(), timeout=timeout)
            if (menu == "1"):
                self._handle_learning_rate(optim, timeout=timeout)
                return
            else:
                print(f'{bcolors.FAIL}invalid choice{bcolors.ENDC}')
                self.menu(timeout=timeout, optim=optim)
                return
        except TimeoutOccurred:
            return


if __name__ == '__main__':
    print("Showing Trainer")

    parser = argparse.ArgumentParser("trainer.data")
    parser.add_argument("--rootpath", help="path to data root directory", type=str, required=True)
    parser.add_argument("--experiment", help="path to experiment root directory", type=str, required=True)
    parser.add_argument("--modelversion", help="version number to save model as", type=str, required=True)
    parser.add_argument("--load_model", help="is this model saved and to be loaded from?", type=str, required=False)
    parser.add_argument("--tensorboard", help="name of tensorboard for this run", type=str, required=True, default='tensorboard')
    parser.add_argument("--batch_size", help="number of batches", type=int, required=True, default=8)
    args = parser.parse_args()

    trainer = DenoisingAutoencoderTrainer()
    if (args.load_model):
        print("loading model")
        trainer.load_model_from_path(os.path.join(args.experiment, f'cnn_240p_den_{args.load_model}'))
    elif (args.load_model is None):
        print("new model")
        trainer.create_new_model()

    else:
        raise Exception(f"Invalid --load_model arg {args.load_model}")
        

    # for name, param in network.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)


    
    num_epochs = 200


    trainer.load_data(args.rootpath, args.batch_size)
    trainer.load_writer(args.experiment, args.tensorboard)

    inputManager = TrainingMenu()

    with tqdm(range(num_epochs), unit="epochs", mininterval=0.01) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
            # manage stuff
            if (epoch % 1 == 0):
                # save
                #if (epoch > 0):
                #save model
                model_name = os.path.join(args.experiment, f'cnn_240p_den_{args.modelversion}')
                trainer.save_model(path=model_name)
                #trainer.release_model()
                #trainer.load_model_from_path(path=model_name)
                pass

                inputManager.menu(optim=trainer.optim, timeout=10)
                print("Continuing...")
                
            train_loss = trainer.train()

            val_loss, valid_images = trainer.validate()

            print(f"Train Loss: {train_loss} Val_loss: {val_loss}")
            # trainer.record_loss(train_loss, val_loss, epoch)
            # if (epoch % 5 == 0):
            #     trainer.record_imgs(tru=valid_images[0], pred=valid_images[1], step=epoch)
            
