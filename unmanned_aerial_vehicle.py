"""
Unmanned Aerial Vehicle control utilities.

This class encapsulates:
- current_position and current_velocity state
- A PID controller for position error
- A Kalman filter matching the previous configuration in slam.py
- RandomMockDistortion: mock an image from an UAV
- Helper to compute center estimates from LK output (matching slam.py logic)
"""
from typing import Optional, Tuple, Sequence
import cv2 as cv
import numpy as np


class UnmannedAerialVehicle:
    def __init__(
            self,
            initial_position: Sequence[float],
            initial_velocity: Sequence[float],
            pid_gains: Tuple[float, float, float],
            map_image: np.ndarray,
            template_size: int,
            input_size: int,
            kalman_strength: float = 8.0,
            random_warp_size: int = 8,
            color_swap: Optional[Tuple[int, int, int]] = None
    ) -> None:
        self.current_position = np.array(initial_position, dtype=np.float32)
        self.desired_velocity = np.array(initial_velocity, dtype=np.float32)
        self._dt = 1

        self._template_size = template_size
        self._input_size = input_size

        self._max_random_warp_size = random_warp_size
        self._color_swap = color_swap if color_swap is not None else [1, 2, 0]

        # Map image (BGR)
        self._map_bgr = map_image.copy()

        # PID controller state
        self.kp, self.ki, self.kd = pid_gains
        self._integral = np.zeros(2, dtype=np.float32)
        self._last_error = np.zeros(2, dtype=np.float32)

        # Kalman filter configuration matching slam.py
        self.kalman = cv.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            np.float32,
        )
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * kalman_strength
        self.kalman.processNoiseCov = (
                np.array(
                    [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]],
                    np.float32,
                )
                * 1.0 / kalman_strength
        )
        # State init
        self.kalman.statePre = np.array(
            [[self.current_position[0]], [self.current_position[1]], [0.0], [0.0]],
            np.float32,
        )

    def set_pid_gains(self, kp: float, ki: float, kd: float) -> None:
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)

    def reset_pid(self) -> None:
        self._integral[:] = 0.0
        self._last_error[:] = 0.0

    def pid_update(self, error: np.ndarray) -> np.ndarray:
        """
        Classic PID controller update for 2D error.
        error: (ex, ey)
        dt: time delta
        Returns control signal (ux, uy)
        """
        e = np.array(error, dtype=np.float32)
        self._integral += e * self._dt
        derivative = (e - self._last_error) / self._dt
        self._last_error = e
        u = self.kp * e + self.ki * self._integral + self.kd * derivative
        return u

    def kalman_correct(self, measurement: np.ndarray) -> None:
        """
        Correct with measurement (mx, my) and predict next position.
        """
        m = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.kalman.correct(m)
        return

    def kalman_predict(self) -> np.ndarray:
        """
        Returns predicted (u, v) as np.ndarray of shape (2,)
        """
        prediction = self.kalman.predict()  # (4,1)
        return np.squeeze(prediction, axis=-1)[0:2]

    def get_downward_facing_image(self) -> np.ndarray:
        return UnmannedAerialVehicle.random_distortion(
            self._map_bgr,
            self.current_position.astype(np.int32),
            (self._template_size, self._template_size),
            (self._input_size, self._input_size),
            self._max_random_warp_size,
            self._color_swap
        )

    @staticmethod
    def random_distortion(map_img: np.ndarray,
                          center: np.ndarray,
                          t_size: Tuple[int, int],
                          i_size: Tuple[int, int],
                          max_rand: int,
                          color_swap: Tuple[int, int, int]) -> np.ndarray:
        """
        - map_img: BGR HxWx3
        - center: (u, v) int pixel center for square region
        - t_size: (H, W) template size to crop after homography warp
        - i_size: (H, W) of input window (expects square). Uses i_size[0] for both height and width
        Returns distorted BGR template (t_size[0] x t_size[1] x 3)
        """
        c = np.asarray(center, dtype=np.int32)
        c_u, c_v = int(c[0]), int(c[1])
        sq_h = int(i_size[0]) // 2
        sq_w = int(i_size[0]) // 2
        square_img = map_img[c_v - sq_h:c_v + sq_h, c_u - sq_w:c_u + sq_w, :].copy()

        tl = (np.asarray(i_size, dtype=np.int32) - np.asarray(t_size, dtype=np.int32)) // 2
        br = np.asarray(t_size, dtype=np.int32) + (
                    np.asarray(i_size, dtype=np.int32) - np.asarray(t_size, dtype=np.int32)) // 2

        tl_rand_a = tl - max_rand
        tl_rand_b = tl + max_rand
        br_rand_a = br - max_rand
        br_rand_b = br + max_rand

        top_left_box_u = np.random.randint(int(tl_rand_a[0]), int(tl_rand_b[0]) + 1)
        top_left_box_v = np.random.randint(int(tl_rand_a[1]), int(tl_rand_b[1]) + 1)

        top_right_box_u = np.random.randint(int(br_rand_a[0]), int(br_rand_b[0]) + 1)
        top_right_box_v = np.random.randint(int(tl_rand_a[1]), int(tl_rand_b[1]) + 1)

        bottom_left_box_u = np.random.randint(int(tl_rand_a[0]), int(tl_rand_b[0]) + 1)
        bottom_left_box_v = np.random.randint(int(br_rand_a[1]), int(br_rand_b[1]) + 1)

        bottom_right_box_u = np.random.randint(int(br_rand_a[0]), int(br_rand_b[0]) + 1)
        bottom_right_box_v = np.random.randint(int(br_rand_a[1]), int(br_rand_b[1]) + 1)

        # prepare source and target four points
        src_points = np.array(
            [
                [top_left_box_u, top_left_box_v],
                [top_right_box_u, top_right_box_v],
                [bottom_left_box_u, bottom_left_box_v],
                [bottom_right_box_u, bottom_right_box_v],
            ],
            dtype=np.float32,
        )

        tgt_points = np.array(
            [[tl[0], tl[1]], [br[0], tl[1]], [tl[0], br[1]], [br[0], br[1]]],
            dtype=np.float32,
        )

        src_points = np.reshape(src_points, [4, 1, 2])
        tgt_points = np.reshape(tgt_points, [4, 1, 2])

        # find homography
        h_matrix, status = cv.findHomography(src_points, tgt_points, 0)
        try:
            warped = cv.warpPerspective(
                square_img,
                h_matrix,
                (int(i_size[1]), int(i_size[0])),
            )
            warped = warped[int(tl[1]):int(br[1]), int(tl[0]):int(br[0]), :]
        except (Exception,):
            warped = square_img

        # shift the colors to introduce color distortion.
        warped = warped[:, :, color_swap]
        return warped

    def step(self, control: np.ndarray) -> np.ndarray:
        """
        Integrate velocity to position.
        control: (ux, uy) applied as velocity increment
        """
        u = np.array(control, dtype=np.float32)
        # Treat control as desired velocity delta
        self.desired_velocity = u * self._dt
        return self.current_position + self.desired_velocity * self._dt
