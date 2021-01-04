import torch
import torchvision
import torch.tensor as Tensor
import training.utils as utils
import numpy as np
from torch.utils import data
from PIL import Image
import PIL


class PngToPIL(object):
    """Transform PIL PNG image to raw image"""
    def __call__(self, img) -> Image:
        return img.convert("RGB")


class BasePILConvert(object):
    """Base class for labels transform"""
    def __init__(self):
        self.label_map = np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, img: Image) -> np.ndarray:
        """Transform RGB image to class labels
        Args:
            img: PIL RGB image in Pascal VOC label format
        Returns:
            labels: numpy array with shape [ H x W ] with class indexes"""
        img = np.array(img).astype(int)
        labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.int16)
        for class_idx, color_state in enumerate(self.label_map):
            labels[np.where(np.all(img == color_state, axis=-1))[:2]] = class_idx
        return labels.astype(int)


class ToOneHot(BasePILConvert):
    """Transform PIL RGB image to tensor with class labels"""
    def __call__(self, img: Image) -> Tensor:
        return utils.labels_one_hot(
            torch.tensor(np.swapaxes(self.encode_segmap(img), 0, -1),
                         dtype=torch.int64).cuda())


class ToLabels(BasePILConvert):
    """Transform PIL RGB image to tensor with class labels in one-hot encoding"""
    def __call__(self, img: Image) -> Tensor:
        return torch.tensor(np.swapaxes(self.encode_segmap(img), 0, -1),
                            dtype=torch.int64).cuda()


# Load datasets
batch_size = 18
img_size = 256
label_size = img_size - 188  # Constant edge difference

# Input image transform normalize
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
# Labels transform
class_convert = torchvision.transforms.Compose(
    [
        PngToPIL(),
        torchvision.transforms.Resize((label_size, label_size), interpolation=Image.NEAREST),
        ToLabels()
    ]
)
# Load training set
trainset = torchvision.datasets.VOCSegmentation(root='./data', image_set='train',
                                                download=False, transform=transform,
                                                target_transform=class_convert)
trainloader = data.DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
# Load test set
testset = torchvision.datasets.VOCSegmentation(root='./data', image_set='val',
                                               download=False, transform=transform,
                                               target_transform=class_convert)
testloader = data.DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
