import kornia.filters as kn_filters
import kornia.morphology as kn_morph
import torch

# @torch.jit.script
def compute_scale_and_shift(prediction, target, mask):
    """Computes the optimal scale and shift according to least-squares
    criteria between the prediction and the target in the masked area

    params:
        pred (torch.Tensor): network prediction tensor (B x H x W)
        grnd (torch.Tensor): ground truth tensor (B x H x W)
        mask (torch.Tensor): mask denoting valid pixels (must be B x H x W)

    returns:
        x_0 (torch.Tensor): scales (B)
        x_1 (torch.Tensor): shifts (B)
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 -
    # a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] -
                  a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] +
                  a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def compute_ssi_pred(pred, grnd, mask):
    """Returns the provided predictions shifted and scaled such that they 
    minimize the L2 difference with the ground truth in the masked area

    params:
        pred (torch.Tensor): network prediction tensor (B x H x W)
        grnd (torch.Tensor): ground truth tensor (B x H x W)
        mask (torch.Tensor): mask denoting valid pixels (must be B x H x W)

    returns:
        (TODO): the network prediction optimally shifted and scaled
    """
    scale, shift = compute_scale_and_shift(pred, grnd, mask)

    # NOTE: early in training this scale can be negative, so we can simply clip it
    # at zero. It could also probably just be set to one if less than 0, it's just
    # to help stabilize early training until the network is making reasonable preds

    scale = torch.nn.functional.relu(scale)
    # scale[scale <= 0] = 1.0
    # scale = torch.abs(scale)

    return (pred * scale.view(-1, 1, 1)) + shift.view(-1, 1, 1)


@torch.jit.script
def resize_aa(img, scale: int):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        scale (TODO): TODO

    returns:
        (TODO): TODO
    """
    if scale == 0:
        return img

    # blurred = TF.gaussian_blur(img, self.k_size[scale])
    # scaled = blurred[:, :, ::2**scale, ::2**scale]
    # blurred = img

    # NOTE: interpolate is noticeably faster than blur and sub-sample
    scaled = torch.nn.functional.interpolate(
        img,
        scale_factor=1/(2**scale),
        mode='bilinear',
        align_corners=True,
        antialias=True
    )
    return scaled

def lp_loss(pred, grnd, mask, p=2):
    """Performs a regular LP loss where P is specified. Can be used to
    compute both MSE (p=2) and L1 (p=1) loss functions

    params:
        pred (torch.Tensor): network prediction tensor (B x C x H x W)
        grnd (torch.Tensor): ground truth tensor (B x C x H x W)
        mask (torch.Tensor): mask denoting valid pixels (must be B x 1 x H x W)
        p (int) optional: degree of L norm (default 2)

    returns:
        (TODO): the mean LP loss between pixels in prediction and ground truth
    """
    if p == 1:
        lp_term = torch.nn.functional.l1_loss(pred, grnd, reduction='none') * mask
    else:
        lp_term = torch.nn.functional.mse_loss(pred, grnd, reduction='none') * mask

    return lp_term.sum() / (mask.sum() * lp_term.shape[1])


class MSGLoss():
    """Multi-scale Gradient Loss implementation

    params:
        scales (int) optional: TODO (default 4)
        taps (list) optional: TODO (default [1,1,1,1])
        k_size (list) optional: TODO (default [3,3,3,3])
        device (str) optional: TODO (default None)
    """
    def __init__(self, scales=4, taps=[1, 1, 1, 1], k_size=[3, 3, 3, 3], device=None):
        """Create an instance of MSGLoss.

        params:
            scales (int) optional: TODO (default 4)
            taps (list) optional: TODO (default [1,1,1,1])
            k_size (list) optional: TODO (default [3,3,3,3])
            device (str) optional: TODO (default None)
        """
        self.n_scale = scales
        self.taps = taps
        self.k_size = k_size
        self.device = device

        # pylint: disable-next=line-too-long
        assert len(self.taps) == self.n_scale, 'number of scales and number of taps must be the same'
        # pylint: disable-next=line-too-long
        assert len(self.k_size) == self.n_scale, 'number of scales and number of kernels must be the same'

        self.imgDerivative = ImageDerivative()

        self.erod_kernels = [torch.ones(2 * t + 1, 2 * t + 1) for t in self.taps]

        if self.device is not None:
            self.to_device(self.device)


    def to_device(self, device):
        """TODO DESCRIPTION

        params:
            device (str): TODO
        """
        self.imgDerivative.to_device(device)
        self.device = device
        self.erod_kernels = [kernel.to(device) for kernel in self.erod_kernels]


    def __call__(self, output, target, mask=None):
        """TODO DESCRIPTION

        params:
            output (TODO): TODO
            target (TODO): TODO
            mask (TODO) optional: TODO (default None)

        returns:
            (TODO): TODO
        """
        return self.forward(output, target, mask)

    def forward(self, output, target, mask):
        """TODO DESCRIPTION

        params:
            output (TODO): TODO
            target (TODO): TODO
            mask (TODO): TODO

        returns:
            loss (TODO): TODO
        """
        diff = output - target

        if mask is None:
            mask = torch.ones(diff.shape[0], 1, diff.shape[2], diff.shape[3])
            mask = mask.to(self.device)

        loss = 0
        for i in range(self.n_scale):
            # resize with antialias
            mask_resized = torch.floor(resize_aa(mask, i) + 0.001)

            # erosion to mask out pixels that are effected by unkowns
            mask_resized = kn_morph.erosion(mask_resized, self.erod_kernels[i])
            diff_resized = resize_aa(diff, i)

            # compute grads
            grad_mag = self.gradient_mag(diff_resized, i)

            # mean over channels
            grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)

            # average the per pixel diffs
            temp = mask_resized * grad_mag
            
            mask_sum = torch.sum(mask_resized)
            if mask_sum != 0:
                # pylint: disable-next=line-too-long
                loss += torch.sum(mask_resized * grad_mag) / (mask_sum * grad_mag.shape[1])

        loss /= self.n_scale
        return loss



    def gradient_mag(self, diff, scale):
        """TODO DESCRIPTION

        params:
            diff (TODO): TODO
            scale (TODO): TODO

        returns:
            grad_magnitude (TODO): TODO
        """
        # B x C x H x W
        grad_x, grad_y = self.imgDerivative(diff, self.taps[scale])

        # B x C x H x W
        grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-8)

        return grad_magnitude


class ImageDerivative():
    """TODO DESCRIPTION

    params:
        device (str) optional: TODO (default None)
    """
    def __init__(self, device=None):
        """Creates an instance of ImageDerivative

        params:
            device (str) optional: TODO (default None)
        """
        # seperable kernel: first derivative, second prefiltering
        tap_3 = torch.tensor([
            [0.425287, -0.0000, -0.425287],
            [0.229879, 0.540242, 0.229879]])
        tap_5 = torch.tensor([
            [0.109604,  0.276691,  0.000000, -0.276691, -0.109604],
            [0.037659,  0.249153,  0.426375,  0.249153,  0.037659]])
        tap_7 = torch.tensor([0])
        tap_9 = torch.tensor([
            [0.0032, 0.0350, 0.1190, 0.1458, -0.0000, -0.1458, -0.1190, -0.0350, -0.0032],
            [0.0009, 0.0151, 0.0890, 0.2349, 0.3201, 0.2349, 0.0890, 0.0151, 0.0009]])
        tap_11 = torch.tensor([0])
        tap_13 = torch.tensor([
            [0.0001, 0.0019, 0.0142, 0.0509, 0.0963, 0.0878, 0.0000,
             -0.0878, -0.0963, -0.0509, -0.0142, -0.0019, -0.0001],
            [0.0000, 0.0007, 0.0071, 0.0374, 0.1126, 0.2119, 0.2605,
             0.2119, 0.1126, 0.0374, 0.0071, 0.0007, 0.0000]])

        self.kernels = [tap_3, tap_5, tap_7, tap_9, tap_11, tap_13]

        # sending them to device
        if device is not None:
            self.to_device(device)


    def to_device(self, device):
        """TODO DESCRIPTION

        params:
            device (str): TODO
        """
        self.kernels = [kernel.to(device) for kernel in self.kernels]


    def __call__(self, img, t_id):
        """TODO DESCRIPTION

        params:
            img (TODO): image with dimensions B x C x H x W
            t_id (int): tap radius (for example t_id=1 will use the tap 3)

        returns:
            (TODO): TODO
        """
        if t_id in [3, 5]:
            assert False, "Not Implemented"

        return self.forward(img, t_id)


    def forward(self, img, t_id=1):
        """TODO DESCRIPTION

        params:
            img (TODO): image with dimensions B x C x H x W
            t_id (int) optional: tap radius (for example t_id=1 will use the tap 3) (default 1)

        returns:
            (tuple): TODO
        """
        kernel = self.kernels[t_id-1]

        p = kernel[1 : 2, ...]
        d1 = kernel[0 : 1, ...]

        # B x C x H x W
        grad_x = kn_filters.filter2d_separable(
            img,
            p,
            d1,
            border_type='reflect',
            normalized=False,
            padding='same')
        grad_y = kn_filters.filter2d_separable(
            img,
            d1,
            p,
            border_type='reflect',
            normalized=False,
            padding='same')

        return (grad_x, grad_y)
