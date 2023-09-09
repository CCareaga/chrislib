import functools
import torch
from torch import nn


def set_grad(nets, requires_grad):
    """TODO DESCRIPTION

    params:
        nets (TODO): TODO
        requires_grad (TODO): TODO
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Identity(nn.Module):
    """Do nothing.

    returns:
        x (nn.tensor): the tensor to perform a no-op forward pass on
    """
    def forward(self, x):
        """A no-op forward pass.

        returns:
            x (nn.tensor): the tensor to perform a no-op forward pass on
        """
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer. For BatchNorm, we use learnable affine parameters and track
    running statistics (mean/stddev). For InstanceNorm, we do not use learnable affine parameters.
    We do not track running statistics.

    params:
        norm_type (str) optional: the name of the normalization layer. Must be one of ["batch",
            "instance", "none"] (default "instance")

    returns:
        norm_layer (TODO): a normalization layer
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError(f'normalization layer {norm_type} is not found')
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights. We use 'normal' in the original pix2pix and CycleGAN paper. But
    xavier and kaiming might work better for some applications. Feel free to try yourself.

    params:
        net (TODO): network to be initialized
        init_type (str) optional: the name of an initialization method. Must be one of ["normal",
            "xavier", "kaiming", "orthogonal"] (default "normal")
        init_gain (float) optional: scaling factor for normal, xavier and orthogonal (default 0.02)
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        # pylint: disable-next=line-too-long
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the
    network weights

    params:
        net (TODO): the network to be initialized
        init_type (str) optional: the name of an initialization method. Must be one of ["normal",
            "xavier", "kaiming", "orthogonal"] (default "normal")
        gain (float) optional: scaling factor for normal, xavier and orthogonal (default 0.02)
        gpu_ids (int list) optional: which GPUs the network runs on: e.g., 0,1,2 (default [])

    returns:
        net (TODO): an initialized network
    """
    # NOTE (chris): I can do the handling of the GPU placement manually
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator

    params:
        input_nc (int): the number of channels in input images
        ndf (int) optional: the number of filters in the last conv layer (default 64)
        n_layers (int) optional: the number of conv layers in the discriminator (default 3)
        norm_layer (TODO) optional: normalization layer (default nn.BatchNorm2d)
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        params:
            input_nc (int): the number of channels in input images
            ndf (int) optional: the number of filters in the last conv layer (default 64)
            n_layers (int) optional: the number of conv layers in the discriminator (default 3)
            norm_layer (TODO) optional: normalization layer (default nn.BatchNorm2d)
        """
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)


    def forward(self, input_data):
        """Standard forward pass.

        params:
            input (TODO): TODO

        returns:
            (TODO): TODO
        """
        return self.model(input_data)


def define_D(
        input_nc,
        ndf,
        netD,
        n_layers_D=3,
        norm='batch',
        init_type='normal',
        init_gain=0.02,
        gpu_ids=[]):
    """Create a discriminator. The discriminator has been initialized by <init_net>. It uses Leaky
    RELU for non-linearity. Our current implementation provides three types of discriminators:

        * basic: 'PatchGAN' classifier described in the original pix2pix paper. It can classify
          whether 70×70 overlapping patches are real or fake. Such a patch-level discriminator
          architecture has fewer parameters than a full-image discriminator and can work on
          arbitrarily-sized images in a fully convolutional fashion.
        * n_layers: With this mode, you can specify the number of conv layers in the discriminator
          with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        * pixel: 1x1 PixelGAN discriminator can classify whether a pixel is real or not. It
          encourages greater color diversity but has no effect on spatial statistics.

    params:
        input_nc (int): the number of channels in input images
        ndf (int): the number of filters in the first conv layer
        netD (str): the architecture's name. Must be one of ["basic", "n_layers", "pixel"]
        n_layers_D (int) optional: the number of conv layers in the discriminator; effective when
            netD=='n_layers' (default 3)
        norm (str) optional: the type of normalization layers used in the network (default "batch")
        init_type (str) optional: the name of the initialization method (default "normal")
        init_gain (float) optional: scaling factor for normal, xavier and orthogonal (default 0.02)
        gpu_ids (int list) optional: which GPUs the network runs on: e.g., 0,1,2 (default [])

    returns:
        (TODO): a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    # NOTE (chris): for now just use the default PatchGAN
    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    # elif netD == 'n_layers':  # more options
    #     net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    # elif netD == 'pixel':     # classify if each pixel is real or fake
    #     net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    # else:
    #     raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """Define different GAN objectives. The GANLoss class abstracts away the need to create the
    target label tensor that has the same size as the input. Do not use sigmoid as the last layer
    of Discriminator. LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.

    params:
        gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and wgangp
        target_real_label (bool) optional: label for a real image (default 1.0)
        target_fake_label (bool) optional: label of a fake image (default 0.0)
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class. Note: Do not use sigmoid as the last layer of
        Discriminator. LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.

        params:
            gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and
                wgangp
            target_real_label (bool) optional: label for a real image (default 1.0)
            target_fake_label (bool) optional: label of a fake image (default 0.0)
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')


    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        params:
            prediction (tensor): tpyically the prediction from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        returns:
            (TODO): a label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)


    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        params:
            prediction (tensor): tpyically the prediction output from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        Returns:
            (TODO): the calculated loss
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
