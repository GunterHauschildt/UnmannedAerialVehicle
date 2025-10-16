from typing import List, Tuple, Optional, Iterator

import cv2 as cv
import numpy as np


class MissionPlanner:
    """
    MissionPlanner builds and iterates a path over a map image.
    Inputs
    - map_file_name: Path to the map image.
    - points: List[Tuple[int, int]]
        Waypoint centers (u, v) in pixel coordinates. The path is constructed by interpolating between
        consecutive points with a velocity between those points
    - max_velocity: float
        The maximum velocity (in Euclidean space)
    Returns (via next_crop())
    - crop_rgb: np.ndarray of shape (self._input_size, self._input_size, 3), dtype=uint8, RGB order.
    - center: Tuple[int, int] (u, v) used as the crop center, after clamping.
    """

    def __init__(
        self,
        map_file_name: str,
        points: List[Tuple[int, int]],
        max_velocity: float,
        template_size: int,
        input_size: int
    ) -> None:

        self._template_size = template_size
        self._input_size = input_size

        self._map_bgr = cv.imread(map_file_name)
        if self._map_bgr is None:
            raise ValueError(f"Failed to read map image from path: {map_file_name}")

        self._max_velocity = max_velocity
        self._velocity = np.array([0., 0.]).astype(np.float32)
        self._acceleration = np.array([0.1, 0.1]).astype(np.float32)

        self._build_path(points)
        self._idx = 0

        self._mission_plan_map = None

    def _build_path(self, pts: List[Tuple[int, int]]) -> None:

        if len(pts) < 2:
            raise ValueError(f"Need at least a start point and an end point.")

        path: List[np.ndarray] = []
        path.append(np.array(pts[0]))

        for i in range(len(pts) - 1):

            self._is_stopping = [False, False]
            p1 = np.array(pts[i + 1], dtype=np.float32)

            direction = np.array(p1 - path[-1])
            direction = direction / np.linalg.norm(direction)

            max_velocity = direction * self._max_velocity
            min_velocity = max_velocity / 100.

            while np.any(np.abs(path[-1] - p1) > 1.):

                # --- per-axis deceleration ---
                remaining = p1 - path[-1]

                for axis in range(2):
                    # compute per-axis stopping distance
                    stopping_distance = (self._velocity[axis] ** 2) / (2 * self._acceleration[axis])

                    # if we still have room to accelerate safely:
                    if stopping_distance < abs(remaining[axis]) and not self._is_stopping[axis]:
                        # accelerate forward
                        self._velocity[axis] += self._acceleration[axis]
                        # clamp to max
                        self._velocity[axis] = min(self._velocity[axis], max_velocity[axis])

                    else:
                        self._is_stopping[axis] = True
                        # time to brake
                        self._velocity[axis] -= self._acceleration[axis]
                        # don't go past some minimum for rounding error
                        self._velocity[axis] = max(self._velocity[axis], min_velocity[axis])

                # Update position. Don't round yet.
                path.append(path[-1] + self._velocity)
                # print(f"Velocity: {self._velocity}, Path: {self._path[-1]}")

        self._path = [np.round(p).astype(np.int32) for p in path]

    def reset(self) -> None:
        self._idx = 0

    def path_length(self) -> int:
        return len(self._path)

    def get_marked_up_map(self) -> np.ndarray:
        """Return a copy of the map with the mission path drawn (BGR)."""
        if self._mission_plan_map is None:
            self._mission_plan_map = self._draw_mission_plan_map()
        return self._mission_plan_map

    def get_map(self) -> np.ndarray:
        """Return the underlying raw BGR map image (no overlays)."""
        return self._map_bgr

    def get_path(self) -> List[np.ndarray]:
        return self._path

    def has_next(self) -> bool:
        return self._idx < len(self._path)

    def next_crop(self) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Returns the next centered crop.

        Output
        - crop_rgb: (self._input_size, self._input_size, 3) uint8, RGB order
        - center: (u, v) int, crop center used (after clamping)

        Returns None when the path is fully consumed.
        """
        if not self.has_next():
            return None

        u, v = self._path[self._idx]
        self._idx += 1

        tl_u = u - self._input_size // 2
        tl_v = v - self._input_size // 2
        br_u = tl_u + self._input_size
        br_v = tl_v + self._input_size

        # Crop in BGR then convert to RGB for return
        crop_bgr = self._map_bgr[tl_v:br_v, tl_u:br_u]
        crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
        return crop_rgb, (u, v)

    def iter_crops(self) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
        self.reset()
        while self.has_next():
            out = self.next_crop()
            if out is None:
                break
            yield out

    def crop(self, num: int) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
        self._idx = num
        return self.next_crop()
