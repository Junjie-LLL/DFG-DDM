import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.dfg_ddm import DFG_UNet
from torch.nn import functional as F
from .loss_util import weighted_loss
from pytorch_msssim import ssim


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm
_reduction_modes = ['none', 'mean', 'sum']

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

class AmplitudeLoss(nn.Module):
    def __init__(self):
        super(AmplitudeLoss, self).__init__()
    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        amp = torch.abs(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        amp1 = torch.abs(fre1)
        return l1_loss(amp, amp1, reduction='mean')

class PhaseLoss(nn.Module):
    def __init__(self):
        super(PhaseLoss, self).__init__()
    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        pha = torch.angle(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        pha1 = torch.angle(fre1)
        return l1_loss(pha, pha1, reduction='mean')
    
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])
    rgb = torch.abs(rgb)

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)


    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def rgb2lab(rgb,ab_norm = 110.,l_cent = 50.,l_norm = 100.):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-l_cent)/l_norm
    ab_rs = lab[:,1:,:,:]/ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

class LabLoss(nn.Module):
    def __init__(self):
        super(LabLoss, self).__init__()
        self.cri_pix = nn.L1Loss()

    def forward(self, output, target):
        output_lab = rgb2lab(output)
        target_lab = rgb2lab(target)
        loss = l1_loss(output_lab,target_lab)

        return loss
    
@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, eps=self.eps)
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)    # filter
        down = filtered[:,:,::2,::2]               # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2,::2] = down*4                  # upsample
        filtered = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
    
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def pixel_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    xx = torch.cat([x0[:, :3, :, :], x], dim=1)
    output = model(xx, t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) # L2损失函数mse


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DFG_UNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        # loss
        self.amplitudeLoss = AmplitudeLoss()
        self.phaseLoss = PhaseLoss()
        self.labLoss = LabLoss()
        # self.edgeLoss = EdgeLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                loss_noise = pixel_loss(self.model, x, t, e, b)
                if epoch > 50 and self.step % 200 == 0: 
                    # 生成结果，计算损失
                    with torch.no_grad():
                        for i, (x, y) in enumerate(val_loader):
                            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                            break
                        n = x.size(0)
                        x_cond = x[:, :3, :, :].to(self.device)
                        x_cond = data_transform(x_cond)
                        x_gt = x[:, 3:, :, :].to(self.device)
                        x_gt = data_transform(x_gt)
                        x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
                        x = self.sample_image(x_cond, x)
                        x = inverse_data_transform(x).to(self.device)
                        x_gt = inverse_data_transform(x_gt)
                    # EdgeLoss，LabLoss，AmplitudeLoss，PhaseLoss
                    # AmplitudeLoss
                    amplitude_loss = self.amplitudeLoss(x, x_gt)
                    phase_loss = self.phaseLoss(x, x_gt)

                    charbonnier_loss = self.charbonnier_loss(x,x_gt)
                    loss = loss_noise + 100*(amplitude_loss + phase_loss) + 100*charbonnier_loss

                    print(f"step: {self.step}, loss_noise: {loss_noise.item()}, amplitude_loss: {100*amplitude_loss.item()}, phase_loss: {100*phase_loss.item()}, charbonnier_loss: {100*charbonnier_loss.item()}")
                else:
                    loss = loss_noise
                    if self.step % 20 == 0:
                        print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm_data_fft_' + str(self.step)))

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size) + "_new_")
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            # x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, "fft_" + str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, "fft_" + str(step), f"{i}.png"))
