import numpy as np

from .augmentation import Augmentation, _transform_to_aug
from .transform import NoOpTransform
from .transform_impl import (
    XFlipTransform,
    YFlipTransform,
    ZRotationTransform,
    ScaleTransform,
    PointSamplingTransform,
)

__all__ = [
    "RandomApply",
    "RandomXFlip",
    "RandomYFlip",
    "RandomZRotation",
    "RandomScale",
    "RandomPointSampling",
]


class RandomApply(Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug, prob=0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        self.aug = _transform_to_aug(tfm_or_aug)
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob

    def get_transform(self, *args):
        do = self._rand_range() < self.prob
        if do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input):
        do = self._rand_range() < self.prob
        if do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomXFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of flip.
        """
        super().__init__()

        self._init(locals())

    def get_transform(self, points):
        do = self._rand_range() < self.prob
        if do:
            return XFlipTransform()
        else:
            return NoOpTransform()


class RandomYFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of flip.
        """
        super().__init__()

        self._init(locals())

    def get_transform(self, points):
        do = self._rand_range() < self.prob
        if do:
            return YFlipTransform()
        else:
            return NoOpTransform()


class RandomZRotation(Augmentation):
    """Rotation about the z-axis."""

    def __init__(self, angle, sample_style="range"):
        super().__init__()

        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)

        self._init(locals())

    def get_transform(self, points):
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
        else:
            angle = np.random.choice(self.angle)

        if angle % 360 == 0:
            return NoOpTransform()

        return ZRotationTransform(angle)


class RandomScale(Augmentation):
    def __init__(self, scale_factor, sample_style="range"):
        super().__init__()

        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(scale_factor, (float, int)):
            scale_factor = (scale_factor, scale_factor)

        self._init(locals())

    def get_transform(self, points):
        if self.is_range:
            scale_factor = np.random.uniform(self.scale_factor[0], self.scale_factor[1])
        else:
            scale_factor = np.random.choice(self.scale_factor)

        if scale_factor == 1:
            return NoOpTransform()

        return ScaleTransform(scale_factor)


class RandomPointSampling(Augmentation):
    def __init__(self, num_points):
        super().__init__()

        self.num_points = num_points

    def get_transform(self, points):
        choice = np.random.choice(
            points.shape[0], self.num_points, replace=points.shape[0] < self.num_points
        )
        return PointSamplingTransform(choice)
