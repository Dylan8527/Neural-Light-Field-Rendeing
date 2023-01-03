from cv2 import UMAT_MAGIC_VAL
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from network import LF_network
from PIL import Image
import torch

class Interpolator:
    def __init__(self,
        lf_network:LF_network,
        interpolator="Gaussian",
        aperture_size=0.1,
        aperture_camera_sample_num=3,
        focal_plane=1./9.,
        undersampled=1.,
    ):
        # We'd better recommond "Gaussian" interpolation.
        self.interpolator = interpolator
        assert interpolator in ["Gaussian"]
        # We use a aperture in a circle shape !
        self.aperture_size=aperture_size
        self.aperture_camera_sample_num = aperture_camera_sample_num # We sample # virtual cameras in this circle~
        # depth = z, for refoucsing
        self.focal_plane = focal_plane
        self.undersampled = undersampled #  
        # My Neural light field.
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu" ) 
        self.lf_net = lf_network.to(self.device) # On GPU for inference~ 
        self.lf_net.eval()
        # Some constant for light field
        self.U, self.V =  16,  16 # camera range
        self.S, self.T = 240, 320 #  pixel range

        
    # Task 1: Generate novel views with neural light field
    def get_LF_val(self, x):
        # x : [0, 1]
        with torch.no_grad():
            return self.lf_net(x)

    def get_single_LF_val(self, u, v, s, t):
        # (u,v) ~ camera's position : [0, 1]
        # (s,t) ~ pixel's position : [0, 1]
        x = torch.stack([u, v, s, t], dim=-1)
        return self.get_LF_val(x)

    def generate_training_views(self, write=False):
        # create an image S * T
        y = np.linspace(0, self.S-1, self.S)
        x = np.linspace(0, self.T-1, self.T)
        i = [x, y]
        y, x = np.meshgrid(*i)
        img_grid = np.stack([x, y], axis=-1)
        # generate 16 training views
        test_imgs = []
        cams = []
        uv = np.linspace(0, 15, 16, endpoint=True)
        vv = np.linspace(0, 15, 16, endpoint=True)
        results = []
        for u in uv:
            for v in vv:
                # random generate virtual camera
                cam = np.array([v,u], dtype=np.float32)
                cams.append(cam)
                # concatenate the camera with the image grid
                test_imgs.append(np.concatenate([np.ones_like(img_grid) * cam, img_grid], axis=-1))
        for img, cam in zip(test_imgs, cams):
            img[:, :, :2] /= 15
            img[:, :, 2] /= (self.S-1)
            img[:, :, 3] /= (self.T-1)
            img = torch.from_numpy(img).float().view(-1, 4).to(self.device)
            out = self.get_LF_val(img)
            
            out = torch.clamp(out, 0, 1)
            out_np = out.view(self.S, self.T, 3).cpu().numpy() * 255
            out_im = Image.fromarray(np.uint8(out_np))
            results.append(np.uint8(out_np))
            if write:
                out_name = f'train_{int(cam[0] + cam[1]*16) + 1:03d}.bmp'
                out_im.save(f'./output/Task1/train_views/{out_name}')
            else:
                plt.imshow(out_im)
                plt.show()
        # save results as video
        path = f'./output/Task1/train_views.avi'
        generate_video(results, path, fps=32)

    def generate_novel_views(self):
        # create an image S * T
        y = np.linspace(0, self.S-1, self.S)
        x = np.linspace(0, self.T-1, self.T)
        i = [x, y]
        y, x = np.meshgrid(*i)
        img_grid = np.stack([x, y], axis=-1)
        # center at (7.5, 7.5)
        center_u, center_v = 7.5, 7.5
        circles_num = 5
        radius = np.linspace(0.5, 7.5, 5)[::-1]
        rads = np.linspace(0, 2*np.pi, 60)
        results = []
        for r in radius:
            for rad in rads:
                du = r * np.cos(rad)
                dv = r * np.sin(rad)
                u = center_u + du
                v = center_v + dv
                cam = np.array([v,u], dtype=np.float32)
                img = np.concatenate([np.ones_like(img_grid) * cam, img_grid], axis=-1)
                img[:, :, :2] /= 15
                img[:, :, 2] /= (self.S-1)
                img[:, :, 3] /= (self.T-1)
                img = torch.from_numpy(img).float().view(-1, 4).to(self.device)
                out = self.get_LF_val(img)
                out = torch.clamp(out, 0, 1)
                out_np = out.view(self.S, self.T, 3).cpu().numpy() * 255
                out_im = np.uint8(out_np)

                tmp_img = Image.fromarray(out_im)
                out_name = f'novel_{u:.1f}_{v:.1f}.bmp'
                tmp_img.save(f'./output/Task1/novel_views/{out_name}')

                results.append(out_im)
        # save results as video
        path = f'./output/Task1/novel_views.avi'
        generate_video(results, path, fps=60)

        

    #TODO Task 2
    
    #TODO Task3 : Camera translational motion
    # actually we implement 
    def camera_moving_along_xy(self):
        # create an image S * T
        y = np.linspace(0, self.S-1, self.S)
        x = np.linspace(0, self.T-1, self.T)
        i = [x, y]
        y, x = np.meshgrid(*i)
        img_grid = np.stack([x, y], axis=-1)
        # generate 16 training views
        test_imgs = []
        cams = []
        uv = np.linspace(0, 15, 32, endpoint=True)
        vv = np.linspace(0, 15, 32, endpoint=True)
        results = []
        for u in uv:
            for v in vv:
                # random generate virtual camera
                cam = np.array([v,u], dtype=np.float32)
                cams.append(cam)
                # concatenate the camera with the image grid
                test_imgs.append(np.concatenate([np.ones_like(img_grid) * cam, img_grid], axis=-1))
        for img, cam in zip(test_imgs, cams):
            img[:, :, :2] /= 15
            img[:, :, 2] /= (self.S-1)
            img[:, :, 3] /= (self.T-1)
            img = torch.from_numpy(img).float().view(-1, 4).to(self.device)
            out = self.get_LF_val(img)
            
            out = torch.clamp(out, 0, 1)
            out_np = out.view(self.S, self.T, 3).cpu().numpy() * 255
            out_im = np.uint8(out_np)
            results.append(out_im)
        # save results as video
        path = f'./output/Task3/camera_moving_along_xy.avi'
        generate_video(results, path, fps=32)

    def camera_moving_along_z(self):
         # create an image S * T
        y = np.linspace(0, self.S-1, self.S) / (self.S-1)
        x = np.linspace(0, self.T-1, self.T) / (self.T-1)
        i = [x, y]
        y, x = np.meshgrid(*i)
        img_grid = np.stack([x, y], axis=-1)
        zv = np.linspace(-0.3, 1e-6, 180)
        results = []
        for z in zv:
            # z <= 0, moving backward~
            u = -(img_grid[..., 0:1] * 2 - 1) * z  + 0.5
            v = -(img_grid[..., 1:2] * 2 - 1) * z * self.S / self.T + 0.5
            img = np.concatenate([v, u, img_grid], axis = -1)
            img[...,0] = 1 - img[...,0]
            img = torch.from_numpy(img).float().view(-1, 4).to(self.device)
            out = self.get_LF_val(img)
            out = torch.clamp(out, 0, 1)
            out_np = out.view(self.S, self.T, 3).cpu().numpy() * 255
            out_im = np.uint8(out_np)
            # save image
            tmp_img = Image.fromarray(out_im)
            out_name = f'z_{z:.6f}.png'
            tmp_img.save(f'./output/Task3/camera_moving_along_z/{out_name}')
            results.append(out_im)
        # save results as video
        path = f'./output/Task3/camera_moving_along_z.avi'
        generate_video(results, path, fps=30)    

    #TODO Task4 : Refocusing and change aperture size.
    def refocusing(self):
        from trad_interpolator import Interpolater
        from dataio import Dataset
        # first generate training views by LFNet
        self.generate_training_views(write=True)
        path = './output/Task1/train_views/'
        data = Dataset(path)
        # Then generate novel views by traditional Methods
        # First, refocusing
        base_dir = "./output/Task4/refocusing"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        focal_planes = np.arange(0.05, 0.50, 0.005)
        results = []
        for focal_plane in focal_planes:
            interpolater = Interpolater(interpolator="quadra-linear", focal_plane=focal_plane, aperture_size=8)
            result = interpolater.interpolate(data, [7.5, 7.5])
            results.append(result)
            generate_image(result, base_dir + "/neur_focal_plane_{focal_plane:.3f}.png".format(focal_plane=focal_plane))
        path = "./output/Task4/"
        generate_video(results, path + "neur_refocusing.avi", fps=30)
        # Second, change aperture size
        base_dir = "./output/Task4/aperture_size"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        aperture_sizes = np.linspace(1, 8, 8, endpoint=True)
        for _aperture_size in aperture_sizes:
            aperture_size = int(_aperture_size)
            interpolater = Interpolater(interpolator="quadra-linear", focal_plane=1./9., aperture_size=aperture_size)
            result = interpolater.interpolate(data, [7.5, 7.5])
            generate_image(result, base_dir +  "/neur_aperture_size_{aperture_size:1d}_2.png".format(aperture_size=aperture_size))

        path = './data/lowtoy/'
        data = Dataset(path)
        results = []
        # then generate novel views by traditional Methods
        # First, refocusing
        base_dir = "./output/Task4/refocusing"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        focal_planes = np.arange(0.05, 0.50, 0.005)
        for focal_plane in focal_planes:
            interpolater = Interpolater(interpolator="quadra-linear", focal_plane=focal_plane, aperture_size=8)
            result = interpolater.interpolate(data, [7.5, 7.5])
            results.append(result)
            generate_image(result, base_dir + "/trad_focal_plane_{focal_plane:.3f}.png".format(focal_plane=focal_plane))
        path = "./output/Task4/"
        generate_video(results, path + "trad_refocusing.avi", fps=30)
        # Second, change aperture size
        base_dir = "./output/Task4/aperture_size"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        aperture_sizes = np.linspace(1, 8, 8, endpoint=True)
        for _aperture_size in aperture_sizes:
            aperture_size = int(_aperture_size)
            interpolater = Interpolater(interpolator="quadra-linear", focal_plane=1./9., aperture_size=aperture_size)
            result = interpolater.interpolate(data, [7.5, 7.5])
            generate_image(result, base_dir +  "/trad_aperture_size_{aperture_size:1d}_2.png".format(aperture_size=aperture_size))
    
def generate_video(results, path, fps=60):
    results = np.array(results)
    results = np.array(results, dtype=np.uint8)
    results = results[:, :, :, ::-1]
    # save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (results.shape[2], results.shape[1]))
    for i in range(results.shape[0]):
        out.write(results[i])
    out.release()

def generate_image(results, path):
    results = np.array(results)
    results = np.array(results, dtype=np.uint8)
    results = results[:, :, ::-1]
    cv2.imwrite(path, results)
if "__main__" == __name__:
    if not os.path.exists("./output"):
        os.makedirs("./output")
        # Task1
        os.makedirs("./output/Task1")
        os.makedirs("./output/Task1/train_views")
        os.makedirs("./output/Task1/novel_views")
        # Task2
        os.makedirs("./output/Task2")
        # Task3
        os.makedirs("./output/Task3")
        os.makedirs("./output/Task3/camera_moving_along_xy")
        os.makedirs("./output/Task3/camera_moving_along_z")
        # Task4
        os.makedirs("./output/Task4")

    # state_path = "/home/vrlab/chenqh/CS276-CP/programming_assignment/assignment1/part2/lowtoy_ggb_a_0.5_in_1/state.pth"
    state_path = "/home/vrlab/chenqh/CS276-CP/programming_assignment/assignment1/part2/lowtoy_pe_mxy_10_muv_10/state.pth"
    state = torch.load(state_path)
    lf_network = LF_network(
        encoding="DiscreteFourier",
        # encoding="Gegenbauer",
        alpha=0.5, 
        in_feature_ratio=1.0,
        hidden_layers=8, 
        skips=[], 
        hidden_features=512,
        with_norm=True, 
        with_res=True,
        multires_uv=10,
        multires_xy=10,
    )
    lf_network.load_state_dict(state['model'])
    
    interpolator = Interpolator(lf_network)
    # Task 1
    # interpolator.generate_training_views(write=True)
    # interpolator.generate_novel_views()

    # Task3 
    # interpolator.camera_moving_along_xy()
    # interpolator.camera_moving_along_z()

    # Task4
    interpolator.refocusing()
    
        

