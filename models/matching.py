import sys
import torch
from .gmatcher import GMatcher
sys.path.append('../')
from utils.common import sift_forward


class Matching(torch.nn.Module):
    """ Image Matching Frontend """
    def __init__(self, config={}):
        super().__init__()
        self.gmodel = GMatcher(config)
        self.max_keypoints = config.get('max_keypoints', -1)

    def forward(self, data):
        pred = {}
        if 'keypoints0' not in data:
            pred0 = sift_forward({'image': data['image0'], 'max_keypoints': self.max_keypoints,
                                  'carhynet': data['carhynet']}, device=data['device'])
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = sift_forward({'image': data['image1'], 'max_keypoints': self.max_keypoints,
                                  'carhynet': data['carhynet']}, device=data['device'])
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        pred = {**pred, **self.gmodel(data)}
        return pred
