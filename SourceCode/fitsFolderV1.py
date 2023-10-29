import joblib
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor()             # 将图像转换为张量
])

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def fits_loader2(path:str) -> np.ndarray:
    a=[]
    try:
        a=joblib.load(path)#.transpose((1,2,0))
    except EOFError:
        print(path)
        print('Some Problems With Your Python')
    return a

def fits_loader3(path:str) -> np.ndarray:
    a=[]
    try:
        a=joblib.load(path).transpose((1,2,0))
    except EOFError:
        print(path)
        print('Some Problems With Your Python')
    return a,path

def is_valid_file(x: str) -> bool:
    return has_file_allowed_extension(x,'mat')  # type: ignore[arg-type]


def get_purify_fits_folder(path:str):
    return ImageFolder(path,loader=fits_loader2,is_valid_file=is_valid_file,transform=transform)

def get_path_fits_folder(path:str):
    return ImageFolder(path,loader=fits_loader3,is_valid_file=is_valid_file)
