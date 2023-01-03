import os
import numpy as np
import tqdm
from PIL import Image
import argparse
import pickle

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader

from network import LF_network

class LFPatchDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True, file_dir = './patch_data/lowtoy_patches/'):
        if is_train:
            self.file_dir = f'{file_dir}/train'
            self.file_list = []
            for f in sorted(os.listdir(self.file_dir)):
                self.file_list.append(f'{file_dir}/train/{f}')
            self.batch_num = len(self.file_list)
        else:
            self.batch_num = 1
            self.file_list = [f'{file_dir}/patch_val.pkl']*1
    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        filename_ = self.file_list[idx]
        with open(filename_, 'rb') as f:
            ret_di = pickle.load(f)

        lab_t = torch.from_numpy(ret_di['y']).float()
        inp_G_t = torch.from_numpy(ret_di['x']).float()
        
        return inp_G_t, lab_t

def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(255**2 / mse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Root directory")
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name")
    parser.add_argument("--trainset_dir", type=str, default="lowtoy")
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--img_W", type=int, default=240)
    parser.add_argument("--img_H", type=int, default=320)

    # test mode -> only output some novel views
    parser.add_argument("--test_mode", action='store_true')
    # experiments 
    parser.add_argument("--encoding", type=str, default="DiscreteFouier")
    # pe 
    parser.add_argument("--multires_xy", type=int, default=10)
    parser.add_argument("--multires_uv", type=int, default=6)
    # Gegenbauer
    parser.add_argument("--in_feature_ratio", type=float, default=1.)
    parser.add_argument("--alpha", type=float, default=0.5)

    args = parser.parse_args()

    device = ("cuda:0" if torch.cuda.is_available() else "cpu" ) 
    if device == "cpu":
        assert device == "cuda:0", "No GPU available"
    root_dir = args.root_dir
    exp_dir = f'{root_dir}/{args.exp_name}'
    print(f'Current experiment directory is: {exp_dir}')
    trainset_dir = f'{root_dir}/{args.trainset_dir}'

    num_epochs = args.num_epochs
    
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.isdir(f'{exp_dir}/valout'):
        os.makedirs(f'{exp_dir}/valout')
    if not os.path.isdir(f'{exp_dir}/testout'):
        os.makedirs(f'{exp_dir}/testout')
    val_im_shape = [240, 320]

    # check whether we have a trained model
    current_epoch = 0
    model = LF_network(
        encoding=args.encoding,
        alpha=args.alpha, 
        in_feature_ratio=args.in_feature_ratio,
        hidden_layers=6, 
        skips=[], 
        hidden_features=256,
        with_norm=True, 
        with_res=True,
        multires_uv=args.multires_uv,
        multires_xy=args.multires_xy,
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    if os.path.isfile(f'{exp_dir}/state.pth'):
        state = torch.load(f'{exp_dir}/state.pth')
        current_epoch = state['current_epoch']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        # scheduler.load_state_dict(state['scheduler'])
        print('Loaded trained state')
    print(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    model = model.to(device)

    trainset = LFPatchDataset(is_train=True, file_dir = trainset_dir)
    valset = LFPatchDataset(is_train=False, file_dir = trainset_dir)
    val_inp_t, _ = valset[0]

    bsize = 1
    train_loader = DataLoader(trainset, batch_size=bsize, drop_last=False, num_workers=8, pin_memory=True)
    iters = len(train_loader)

    # Frequency to save validation image
    val_freq = 600#iters * 2
    # Frequency to save the checkpoint
    save_freq = 1
    
    mse_losses, psnrs = [], []
    # training/validation loop
    if not args.test_mode:
        print('Starts training')
        start_epoch = current_epoch
        for epoch in range(start_epoch, start_epoch+num_epochs):
            e_psnr, e_loss, it = 0, 0, 0
            t = tqdm.tqdm(train_loader)
            # training
            for batch_idx, (inp_G_t, lab_t) in enumerate(t):
                optimizer.zero_grad()
                inp_G_t, lab_t = inp_G_t.view(-1, inp_G_t.shape[-1]).to(device), lab_t.view(-1, 3).to(device)
                
                # scale the input coordinates from integers to floats
                inp_G_t[..., :2] /= 15
                inp_G_t[..., 2]  /= (args.img_W-1)
                inp_G_t[..., 3]  /= (args.img_H-1)

                out = model(inp_G_t)
                mse_loss = torch.nn.functional.mse_loss(out, lab_t)
                loss = mse_loss
                loss.backward()
                optimizer.step()
                
                psnr = 10 * np.log10(1 / mse_loss.item())
                e_psnr += psnr
                e_loss += mse_loss.item()
                it += 1
                t.set_postfix(PSNR = psnr, EpochPSNR = e_psnr / it, EpochLoss = e_loss / it)
            # validation
            if True:
                model.eval()
                # create an image witdth * height
                y = np.linspace(0, val_im_shape[0]-1, val_im_shape[0])
                x = np.linspace(0, val_im_shape[1]-1, val_im_shape[1])
                i = [x, y]
                y, x = np.meshgrid(*i)
                img_grid = np.stack([x, y], axis=-1)
                # randomly sample 5 views
                valid_imgs = []
                cams = []
                for i in range(5):
                    # random generate virtual camera
                    cam = np.random.rand(2) * 15
                    cams.append(cam)
                    # concatenate the camera with the image grid
                    valid_imgs.append(np.concatenate([np.ones_like(img_grid)* cam, img_grid], axis=-1))
                with torch.no_grad():
                    for img, cam in zip(valid_imgs, cams):
                        img[:, :, :2] /= 15
                        img[:, :, 2] /= (args.img_W-1)
                        img[:, :, 3] /= (args.img_H-1)
                        img = torch.from_numpy(img).float().view(-1, 4).to(device)
                        out = model(img)
                        out = torch.clamp(out, 0, 1)
                        out_np = out.view(val_im_shape[0], val_im_shape[1], 3).cpu().numpy() * 255
                        out_im = Image.fromarray(np.uint8(out_np))
                        out_name = f'valout/valout_e_{epoch}_c_{cam[0]:.1f}_{cam[1]:.1f}.png'
                        out_im.save(f'{exp_dir}/{out_name}')
                model.train()
            
            scheduler.step()
            
            print('Epoch: %s Ave PSNR: %s Ave Loss: %s'%(epoch, e_psnr / it, e_loss / it))
            psnrs.append(e_psnr / it); mse_losses.append(e_loss / it)

            if epoch % save_freq == 0 and epoch != 0:
                state = {'current_epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}  
                torch.save(state, f'{exp_dir}/state.pth')

        np.savetxt(f'{exp_dir}/mse_stats.txt', mse_losses, delimiter=',')
        np.savetxt(f'{exp_dir}/psnr_stats.txt', psnrs, delimiter=',')
    # test mode
    else:
        model.eval()
        # create an image witdth * height
        y = np.linspace(0, val_im_shape[0]-1, val_im_shape[0])
        x = np.linspace(0, val_im_shape[1]-1, val_im_shape[1])
        i = [x, y]
        y, x = np.meshgrid(*i)
        img_grid = np.stack([x, y], axis=-1)
        # generate 16 training views
        test_imgs = []
        cams = []
        uv = np.linspace(0, 15, 16, endpoint=True)
        vv = np.linspace(0, 15, 16, endpoint=True)
        for u in uv:
            for v in vv:
                # random generate virtual camera
                cam = np.array([u,v], dtype=np.float32)
                cams.append(cam)
                # concatenate the camera with the image grid
                test_imgs.append(np.concatenate([np.ones_like(img_grid) * cam, img_grid], axis=-1))
        with torch.no_grad():
            it = 0
            for img, cam in zip(test_imgs, cams):
                it += 1 
                img[:, :, :2] /= 15
                img[:, :, 2] /= (args.img_W-1)
                img[:, :, 3] /= (args.img_H-1)
                img = torch.from_numpy(img).float().view(-1, 4).to(device)
                out = model(img)
                
                out = torch.clamp(out, 0, 1)
                out_np = out.view(val_im_shape[0], val_im_shape[1], 3).cpu().numpy() * 255
                out_im = Image.fromarray(np.uint8(out_np))
                out_name = f'testout/testout_e_{current_epoch}_c_{int(cam[0] + cam[1]*16) + 1:d}.png'
                out_im.save(f'{exp_dir}/{out_name}')

