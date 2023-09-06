import kornia.filters as kn_filters
import kornia.morphology as kn_morph
import torchvision.transforms.functional as TF
import torch


def lp_loss(pred, grnd, mask, p=2):
    """TODO DESCRIPTION

    params:
        * pred (TODO): TODO
        * grnd (TODO): TODO
        * mask (TODO): TODO (must be B x 1 x H x W)
        * p (int) optional: TODO (default 2)

    returns:
        * (TODO): TODO
    """
    if p == 1:
        lp_term = torch.nn.functional.l1_loss(pred, grnd, reduction='none') * mask
    if p == 2:
        lp_term = torch.nn.functional.mse_loss(pred, grnd, reduction='none') * mask
                       
    return lp_term.sum() / (mask.sum() * lp_term.shape[1])


class MSGLoss():
    """TODO DESCRIPTION

    params:
        * scales (int) optional: TODO (default 4)
        * taps (list) optional: TODO (default [1,1,1,1])
        * k_size (list) optional: TODO (default [3,3,3,3])
        * device (str) optional: TODO (default None)
    """
    def __init__(self, scales=4, taps=[1, 1, 1, 1], k_size=[3, 3, 3, 3], device=None):
        """Create an instance of MSGLoss.

        params:
            * scales (int) optional: TODO (default 4)
            * taps (list) optional: TODO (default [1,1,1,1])
            * k_size (list) optional: TODO (default [3,3,3,3])
            * device (str) optional: TODO (default None)
        """
        self.n_scale = scales
        self.taps = taps
        self.k_size = k_size
        self.device = device

        assert len(self.taps) == self.n_scale, 'number of scales and number of taps must be the same'
        assert len(self.k_size) == self.n_scale, 'number of scales and number of kernels must be the same'

        self.imgDerivative = ImageDerivative()

        self.erod_kernels = [torch.ones(2 * t + 1, 2 * t + 1) for t in self.taps]

        if self.device is not None:
            self.to_device(self.device)


    def to_device(self, device):
        """TODO DESCRIPTION

        params:
            * device (str): TODO
        """
        self.imgDerivative.to_device(device)
        self.device = device
        self.erod_kernels = [kernel.to(device) for kernel in self.erod_kernels]


    def __call__(self, output, target, mask=None):
        """TODO DESCRIPTION

        params:
            * output (TODO): TODO
            * target (TODO): TODO
            * mask (TODO) optional: TODO (default None)

        returns:
            * (TODO): TODO
        """
        return self.forward(output, target, mask)


    def forward(self, output, target, mask):
        """TODO DESCRIPTION

        params:
            * output (TODO): TODO
            * target (TODO): TODO
            * mask (TODO): TODO

        returns:
            * loss (TODO): TODO
        """
        diff = output - target

        if mask is None:
            mask = torch.ones(diff.shape[0], 1, diff.shape[2], diff.shape[3])
            mask = mask.to(self.device)

        loss = 0
        for i in range(self.n_scale):
            # resize with antialias
            mask_resized = torch.floor(self.resize_aa(mask, i) + 0.001)

            # erosion to mask out pixels that are effected by unkowns
            mask_resized = kn_morph.erosion(mask_resized, self.erod_kernels[i])
            diff_resized = self.resize_aa(diff, i)

            # compute grads
            grad_mag = self.gradient_mag(diff_resized, i)

            # mean over channels
            grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)

            # average the per pixel diffs
            temp = mask_resized * grad_mag

            loss += torch.sum(mask_resized * grad_mag) / (torch.sum(mask_resized) * grad_mag.shape[1])

        loss /= self.n_scale
        return loss

    def resize_aa(self, img, scale):
        """TODO DESCRIPTION

        params:
            * img (TODO): TODO
            * scale (TODO): TODO

        returns:
            * (TODO): TODO
        """
        if scale == 0:
            return img

        # blurred = TF.gaussian_blur(img, self.k_size[scale])
        # scaled = blurred[:, :, ::2**scale, ::2**scale]
        # blurred = img

        # NOTE: interpolate is noticeably faster than blur and sub-sample
        scaled = torch.nn.functional.interpolate(img, scale_factor=1/(2**scale), mode='bilinear', align_corners=True, antialias=True)
        return scaled


    def gradient_mag(self, diff, scale):
        """TODO DESCRIPTION

        params:
            * diff (TODO): TODO
            * scale (TODO): TODO

        returns:
            * grad_magnitude (TODO): TODO
        """
        # B x C x H x W
        grad_x, grad_y = self.imgDerivative(diff, self.taps[scale])

        # B x C x H x W
        grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 0.001)

        return grad_magnitude


class ImageDerivative():
    """TODO DESCRIPTION

    params:
        * device (str) optional: TODO (default None)
    """
    def __init__(self, device=None):
        """Creates an instance of ImageDerivative

        params:
            * device (str) optional: TODO (default None)
        """
        # seperable kernel: first derivative, second prefiltering
        tap_3 = torch.tensor([[0.425287, -0.0000, -0.425287], [0.229879, 0.540242, 0.229879]])
        tap_5 = torch.tensor([[0.109604,  0.276691,  0.000000, -0.276691, -0.109604], [0.037659,  0.249153,  0.426375,  0.249153,  0.037659]])
        tap_7 = torch.tensor([0])
        tap_9 = torch.tensor([[0.0032, 0.0350, 0.1190, 0.1458, -0.0000, -0.1458, -0.1190, -0.0350, -0.0032], [0.0009, 0.0151, 0.0890, 0.2349, 0.3201, 0.2349, 0.0890, 0.0151, 0.0009]])
        tap_11 = torch.tensor([0])
        tap_13 = torch.tensor([[0.0001, 0.0019, 0.0142, 0.0509, 0.0963, 0.0878, 0.0000, -0.0878, -0.0963, -0.0509, -0.0142, -0.0019, -0.0001],
                               [0.0000, 0.0007, 0.0071, 0.0374, 0.1126, 0.2119, 0.2605, 0.2119, 0.1126, 0.0374, 0.0071, 0.0007, 0.0000]])

        self.kernels = [tap_3, tap_5, tap_7, tap_9, tap_11, tap_13]

        # sending them to device
        if device is not None:
            self.to_device(device)


    def to_device(self, device):
        """TODO DESCRIPTION

        params:
            * device (str): TODO
        """
        self.kernels = [kernel.to(device) for kernel in self.kernels]


    def __call__(self, img, t_id):
        """TODO DESCRIPTION

        params:
            * img (TODO): image with dimensions B x C x H x W
            * t_id (int): tap radius (for example t_id=1 will use the tap 3)

        returns:
            * (TODO): TODO
        """
        if t_id == 3 or t_id == 5:
            assert False, "Not Implemented"

        return self.forward(img, t_id)


    def forward(self, img, t_id=1):
        """TODO DESCRIPTION

        params:
            * img (TODO): image with dimensions B x C x H x W
            * t_id (int) optional: tap radius (for example t_id=1 will use the tap 3) (default 1)

        returns:
            * (tuple): TODO
        """
        kernel = self.kernels[t_id-1]

        p = kernel[1 : 2, ...]
        d1 = kernel[0 : 1, ...]

        # B x C x H x W
        grad_x = kn_filters.filter2d_separable(img, p, d1, border_type='reflect', normalized=False, padding='same')
        grad_y = kn_filters.filter2d_separable(img, d1, p, border_type='reflect', normalized=False, padding='same')

        return (grad_x, grad_y)
