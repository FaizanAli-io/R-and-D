import numpy as np


class LSC:
    def __init__(self, points, scale):
        self.points = points
        self.scale = scale

    def compute(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        n = len(self.points)

        sxsq = np.sum(x ** 2)
        sysq = np.sum(y ** 2)
        sxy = np.sum(x * y)
        sx = np.sum(x)
        sy = np.sum(y)

        left = np.array([
            [sxsq, sxy, sx],
            [sxy, sysq, sy],
            [sx, sy, n],
        ])

        right = np.array([
            np.sum(x * (x ** 2 + y ** 2)),
            np.sum(y * (x ** 2 + y ** 2)),
            np.sum(x ** 2 + y ** 2),
        ])

        A, B, C = np.dot(np.linalg.inv(left), right)

        k, m = A / 2, B / 2
        r = (np.sqrt(4 * C + A ** 2 + B ** 2)) / 2

        return k, m, r

    def search(self, a, b, arr, tar):
        m = (a + b) // 2

        if tar <= arr[a] or tar >= arr[b]:
            if tar - arr[a] < arr[b] - tar:
                return a
            return b

        if arr[m] == tar:
            return m

        if abs(a - b) == 1:
            if arr[a] <= tar <= arr[b]:
                if tar - arr[a] < arr[b] - tar:
                    return a
                return b

        elif arr[m] > tar:
            return self.search(a, m, arr, tar)

        elif arr[m] < tar:
            return self.search(m, b, arr, tar)

    def choose(self, rad, cop):
        radii = [row[1] * self.scale for row in cop]
        match = self.search(0, len(radii) - 1, radii, rad)
        return cop[match]
