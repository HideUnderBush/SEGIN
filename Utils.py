from torchvision import transforms
from PIL import Image, ImageOps
from torch.autograd import Variable
from torchvision.utils import make_grid
import numpy as np


def load_image(img_path, to_array=False, to_variable=False):
    img = Image.open(img_path).convert("RGB")
    #img = ImageOps.fit(img, (224,224), Image.ANTIALIAS)

    #scale = transforms.Scale((256,256))
    tensorize = transforms.ToTensor()
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    loader = transforms.Compose([
        #scale, tensorize, normalize
        tensorize, normalize
    ])
    img_tensor = loader(img)

    if to_array:
        img_tensor = img_tensor.unsqueeze(0)
    if to_variable:
        img_tensor = Variable(img_tensor)

    return img_tensor


def deprocess_image(tensor, is_th_variable=False, is_th_tensor=False, un_normalize=True):
    img = tensor
    if is_th_variable:
        img = tensor.data.numpy()
    if is_th_tensor:
        img = tensor.numpy()
    if un_normalize:
        img[:, :, 0] = (img[:, :, 0] * .228 + .485)
        img[:, :, 1] = (img[:, :, 1] * .224 + .456)
        img[:, :, 2] = (img[:, :, 2] * .225 + .406)
    return img


def get_viz_tensor(activations_tensor):
    """
    :param activations_tensor: pytorch variable of shape C * H * W
    :return: a numpy array of H * W * 3
    """
    reshaped_tensor = activations_tensor.contiguous().view(-1, 1, activations_tensor.size()[1], activations_tensor.size()[2])
    grid = make_grid(reshaped_tensor).numpy()
    grid = np.transpose(grid, (1, 2, 0))
    return grid


