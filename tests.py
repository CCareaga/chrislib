import torch
from timeit import default_timer as timer
from skimage.transform import resize

from data_util import load_image
from loss import MSGLoss

test = load_image('/home/chris/research/intrinsic/data/bulk_input/unsplash/deep_cove1.jpg')
test = resize(test, (384, 384)) #[:, :, 0]
test = torch.from_numpy(test)

pred = test.permute(2, 0, 1).unsqueeze(0).repeat(8, 1, 1, 1)
grnd = torch.ones((8, 3, 384, 384))
mask = (torch.ones((8, 1, 384, 384))).float()

msg_loss = MSGLoss(scales=4, taps=[2, 1, 1, 1], k_size=[3, 5, 7, 9])

loss_8 = msg_loss(pred, grnd, mask)
loss_1 = msg_loss(pred[0].unsqueeze(0), grnd[0].unsqueeze(0), mask[0].unsqueeze(0))
print(loss_1, loss_8)

