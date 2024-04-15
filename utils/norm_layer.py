import torch

class NormalizeLayer(torch.nn.Module):

    def __init__(self, means, sds):
        super(NormalizeLayer, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.means = torch.tensor(means).to(self.device)
        self.sds = torch.tensor(sds).to(self.device)

    def forward(self, input):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

# RESA_MEAN = [103.939, 116.779, 123.68]
# RESA_STDDEV = [1., 1., 1.]

RESA_MEAN = [0.4076, 0.4579, 0.4850]
RESA_STDDEV = [1., 1., 1.]

_IMAGENET_MEAN_u = [0.485, 0.406, 0.456]
_IMAGENET_STDDEV_u = [0.229, 0.225, 0.224]

def get_normalize_layer(model='ufld'):
    if model == 'resa':
        return NormalizeLayer(RESA_MEAN, RESA_STDDEV)
    elif model == 'ufld':
        return NormalizeLayer(_IMAGENET_MEAN_u, _IMAGENET_STDDEV_u)
    else:
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)

