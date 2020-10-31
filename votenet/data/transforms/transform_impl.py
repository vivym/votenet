import numpy as np

from .transform import Transform

__all__ = [
    "XFlipTransform",
    "YFlipTransform",
    "ZRotationTransform",
    "ScaleTransform",
    "PointSamplingTransform",
]


class XFlipTransform(Transform):
    def apply_points(self, points: np.ndarray):
        points[:, 0] *= -1
        return points

    def apply_box(self, box: np.ndarray):
        assert box.shape[1] == 6 or box.shape[1] == 7

        box[:, 0] *= -1

        if box.shape[1] == 7:
            box[:, 6] = np.pi - box[:, 6]

        return box


class YFlipTransform(Transform):
    def apply_points(self, points: np.ndarray):
        points[:, 1] *= -1
        return points

    def apply_box(self, box: np.ndarray):
        assert box.shape[1] == 6 or box.shape[1] == 7

        box[:, 1] *= -1

        if box.shape[1] == 7:
            box[:, 6] *= -1

        return box


class ZRotationTransform(Transform):
    """Rotation about the z-axis."""

    def __init__(self, angle: float):
        super().__init__()

        angle = np.deg2rad(angle)
        c = np.cos(angle)
        s = np.sin(angle)
        self.angle = angle
        self.rot_mat = np.array([[c, -s, 0],
                                 [s, c, 0],
                                 [0, 0, 1]])

    def apply_points(self, points: np.ndarray):
        points[:, :3] = np.dot(points[:, :3], np.transpose(self.rot_mat))
        return points

    def apply_box(self, box: np.ndarray):
        assert box.shape[1] == 6 or box.shape[1] == 7
        batch_size = box.shape[0]

        box[:, :3] = np.dot(box[:, :3], np.transpose(self.rot_mat))

        if box.shape[1] == 6:  # axis aligned box
            dx, dy = box[:, 3] / 2., box[:, 4] / 2.
            new_x = np.zeros((batch_size, 4))
            new_y = np.zeros((batch_size, 4))

            for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
                crnrs = np.zeros((batch_size, 3))
                crnrs[:, 0] = crnr[0] * dx
                crnrs[:, 1] = crnr[1] * dy
                crnrs = np.dot(crnrs, np.transpose(self.rot_mat))
                new_x[:, i] = crnrs[:, 0]
                new_y[:, i] = crnrs[:, 1]

            new_x = 2. * np.max(new_x, 1)
            new_y = 2. * np.max(new_y, 1)
            box[:, 3] = new_x
            box[:, 4] = new_y
        else:
            box[:, 6] -= self.angle

        return box


class ScaleTransform(Transform):
    def __init__(self, scale_factor: float):
        super().__init__()

        self.scale_factor = scale_factor

    def apply_points(self, points: np.ndarray):
        points[:, :3] *= self.scale_factor
        return points

    def apply_box(self, box: np.ndarray):
        box[:, :6] *= self.scale_factor
        return box


class PointSamplingTransform(Transform):
    def __init__(self, choice: np.ndarray):
        super().__init__()

        self.choice = choice

    def apply_points(self, points: np.ndarray):
        return points[self.choice]

    def apply_box(self, box: np.ndarray):
        return box
