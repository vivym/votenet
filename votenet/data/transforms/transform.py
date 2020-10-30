from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)

__all__ = [
    "RotationTransform",
]


class PointSamplingTransform(Transform):
    def __init__(self, num_points):
        super().__init()

        self.num_points = num_points

    def apply_points(self, points):


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])
