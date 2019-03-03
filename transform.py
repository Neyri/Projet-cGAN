from torchvision import transforms, utils


class RandomSizedCrop(transforms.RandomSizedCrop):
    """Overload RandomSizedCrop to use it for our dataset"""

    def __init__(self, output_size):
        super(RandomSizedCrop, self).__init__()

    def __call__(self, sample):
