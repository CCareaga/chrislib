import torch
from altered_midas.midas_net import MidasNet
from altered_midas.midas_net_custom import MidasNet_small


def load_models(ord_path, mrg_path, device='cuda'):
    """TODO DESCRIPTION

    params:
        ord_path (str): TODO
        mrg_path (str): TODO
        device (str) optional: the device to run the model on (default "cuda")

    returns:
        models (TODO): TODO
    """
    models = {}

    ord_model = MidasNet()
    ord_model.load_state_dict(torch.load(ord_path))
    ord_model.eval()
    ord_model = ord_model.to(device)

    mrg_model = MidasNet_small(exportable=False, input_channels=5, output_channels=1)
    mrg_model.load_state_dict(torch.load(mrg_path))
    mrg_model.eval()
    mrg_model = mrg_model.to(device)

    models['ordinal_model'] = ord_model
    models['real_model'] = mrg_model

    return models
