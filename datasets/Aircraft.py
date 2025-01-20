import numpy as np
import torchvision.transforms as transforms

def dataset_setting(nSupport, img_size=32):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    mean = [0.4815, 0.4578, 0.4082]
    std = [0.2686, 0.2613, 0.2758]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                         transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                         transforms.RandomHorizontalFlip(),
                                         #lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    valTransform = transforms.Compose([#lambda x: np.asarray(x),
                                       transforms.Resize((248, 248)),
                                       transforms.CenterCrop((img_size, img_size)),
                                       transforms.ToTensor(),
                                       normalize
                                       ])
    
    inputW, inputH, nbCls = img_size, img_size, 64

    trainDir = None
    valDir = None
    testDir = './data/test_data/Aircraft_fewshot/test'
    episodeJson = None

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
