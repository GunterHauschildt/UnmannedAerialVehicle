import os
import numpy as np
import cv2 as cv
from mission_plan import MissionPlanner as MissionPlan
from unmanned_aerial_vehicle import UnmannedAerialVehicle as UAV
from deep_lukas_kanade import DeepLukasKanade
import DLK.net
import argparse
from typing import Optional
import platform


class Draw:
    def __init__(self, map_draw: np.ndarray,
                 path: list[np.ndarray],
                 y_draw_size: int,
                 template_size: int,
                 input_size: int,
                 video_out: Optional[str] = None):

        self._blank = 64
        self._text_size = 40

        y = max(y_draw_size, template_size + self._blank + self._text_size + input_size)
        self._scale = y / map_draw.shape[0]
        self._map_draw = cv.resize(map_draw,
                                   (round(map_draw.shape[1] * self._scale),
                                    round(map_draw.shape[0] * self._scale)))

        # if running wsl and mobaxterm on laptop with a 2nd monitor, this
        # puts the display on (my) 2nd monitor
        if in_wsl := "microsoft" in platform.uname().release.lower():
            self._x0 = 2100
            self._y0 = 100
        else:
            self._x0 = 100
            self._y0 = 100

        path = np.round(np.array(path).astype(np.float32) * self._scale).astype(np.int32).reshape(
            -1, 1, 2)
        self._map_draw = cv.polylines(self._map_draw, [path], isClosed=False, color=(0, 200, 200),
                                      thickness=2,
                                      lineType=cv.LINE_AA)

        self._shape = (self._map_draw.shape[0],
                       self._map_draw.shape[1] + input_size + self._blank,
                       3)

        self._video_stream_out = None
        if video_out is not None:
            self._video_stream_out = cv.VideoWriter(
                  video_out if video_out.endswith('.mp4') else video_out + '.mp4',
                  cv.VideoWriter.fourcc(*'mp4v'),
                  4,
                  (self._shape[1], self._shape[0])
            )

    def draw(self,
             template_tensor,
             input_tensor,
             x,
             y,
             obj1,
             obj2,
             obj3
             ):

        def scale_pts(pts):
            return np.round(np.array(pts).astype(np.float32) * self._scale).astype(np.int32)

        obj1 = scale_pts(obj1)
        obj2 = scale_pts(obj2)
        obj3 = scale_pts(obj3)

        # Map with estimated and predicted centers
        map_draw = self._map_draw.copy()
        map_draw = cv.circle(map_draw, obj1, 10, (0, 255, 0), 3)
        map_draw = cv.circle(map_draw, obj2, 10, (255, 0, 0), 2)
        map_draw = cv.circle(map_draw, obj3, 10, (0, 0, 255), 1)

        # UAV image
        template_draw = (template_tensor[:, :, ::-1] * 255.0).astype(dtype=np.uint8)

        # Input image (with homography)
        input_draw = (input_tensor[:, :, ::-1] * 255.0).astype(dtype=np.uint8)
        x = np.round(x).astype(np.int32)
        y = np.round(y).astype(np.int32)
        for i in range(4):
            input_draw = cv.line(
                input_draw,
                np.array((x[0][i % 4], y[0][i % 4])),
                np.array((x[0][(i + 1) % 4], y[0][(i + 1) % 4])),
                (255,),
                2,
            )

        # Concat them all
        draw = np.zeros(self._shape, dtype=np.uint8)

        yms = 0
        yme = map_draw.shape[0]
        xms = 0
        xme = map_draw.shape[1]
        draw[yms:yme, xms:xme, :] = map_draw

        yts = self._text_size
        yte = yts + template_draw.shape[0]
        xts = map_draw.shape[1] + (input_draw.shape[1] - template_draw.shape[1]) // 2
        xte = xts + template_draw.shape[1]
        draw[yts:yte, xts:xte, :] = template_draw

        yis = yte + self._blank
        yie = yis + input_draw.shape[0]
        xis = map_draw.shape[1] + self._blank // 2
        xie = xis + input_draw.shape[1]
        draw[yis:yie, xis:xie, :] = input_draw

        cv.putText(draw, "UAV", (xts, yts - 10), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
        cv.putText(draw, "HOMOGRAPHY", (xis, yis - 10), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
        cv.putText(draw, "MISSION PLAN", (10, yts - 10), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

        win_name = "Deep Lucas Kanade"
        cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
        cv.moveWindow(win_name, self._x0, self._y0)
        cv.imshow(win_name, draw)
        cv.waitKey(1)

        if self._video_stream_out is not None:
            self._video_stream_out.write(draw)


def homography_to_angle(homography):
    # Normalize H so H[2,2] = 1 for stability
    homography = homography / homography[2, 2]
    rads = np.arctan2(homography[1, 0], homography[0, 0])
    angle_degrees = np.degrees(rads)
    return angle_degrees


def homography_to_scale(homography):
    homography = homography / homography[2, 2]
    sx = np.sqrt(homography[0, 0] ** 2 + homography[1, 0] ** 2)
    sy = np.sqrt(homography[0, 1] ** 2 + homography[1, 1] ** 2)
    return np.array([sx, sy])  # scale in x and y


def homography_to_translation(homography):
    homography = homography / homography[2, 2]
    tx = homography[0, 2]
    ty = homography[1, 2]
    return np.array([tx, ty])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str)
    parser.add_argument('--video-output', type=str, default=None)
    args = parser.parse_args()
    map_path = args.map

    if not os.path.isfile(map_path):
        assert ValueError, "Map path is not valid."

    # Define a simple vector of mission points (u, v) and a mission planner
    # to track along those points.
    points = [(1000, 1000), (8000, 4000)]
    mission_planner = MissionPlan(
        map_path,
        points,
        max_velocity=4.0,
        template_size=DLK.net.template_size,
        input_size=DLK.net.input_size
    )

    # Initialize the UAV control system
    _, mp_center = mission_planner.next_crop()
    mission_planner.reset()
    uav = UAV(
        initial_position=np.array(mp_center, dtype=np.float32),
        initial_velocity=(0.0, 0.0),
        pid_gains=(1.0, 0.0, 0.0),
        map_image=mission_planner.get_map(),
        template_size=DLK.net.template_size,
        input_size=DLK.net.input_size
    )

    # Initialize DLK which owns networks and LK layers
    dlk = DeepLukasKanade(
        feature_map_type='special',
        initial_type='multi_net',
        template_size=DLK.net.template_size,
        input_size=DLK.net.input_size
    )
    dlk_offset = (DLK.net.input_size - DLK.net.template_size) // 2
    dlk_offset = np.array([dlk_offset, dlk_offset]).astype(np.float32)

    # For visualizing and recording
    draw = Draw(mission_planner.get_map(), mission_planner.get_path(),
                640, template_size=DLK.net.template_size, input_size=DLK.net.input_size,
                video_out=args.video_output)

    #############################################
    # Run through the mission plan.
    #############################################

    for mp_image_rgb, mp_center in mission_planner.iter_crops():

        # Convert the mission plan's map to a tensor
        input_tensor = (mp_image_rgb.astype(np.float32) / 320.0)[None, ...]  # (1,H,W,3) RGB [0,1]

        # Build a mocked distorted 128x128 template from UAV.
        # This is designed to mock an image from a downward facing camera on the UAV.
        # The channels are shifted to change the colors. In the ottawa map, this makes
        # the season change from summer to fall. Well. Kind of.
        template = uav.get_downward_facing_image()
        template = template.astype(np.float32) / 360.0
        template_tensor = template[None, ...]

        # Run inference. x, y are the x and y points of the homography between the UAV's template
        # and the mission plan's expected image.
        homography, x, y = dlk.run_inference(
            input_tensor,
            template_tensor,
            fk_loop=4,
        )
        homography = homography.numpy()[0]
        x = x.numpy().astype(np.float32)
        y = y.numpy().astype(np.float32)

        angle = homography_to_angle(homography)
        scale = homography_to_scale(homography)
        translation = homography_to_translation(homography) - dlk_offset

        # print(f"Angle :{angle}, "
        #       f"Scale : {scale}, "
        #       f"Translation: {translation}")

        # Run the PID loop. error is where we want to be minus where we think we are
        pid_error = np.array(mp_center).astype(np.float32) - uav.current_position - translation
        control = uav.pid_update(pid_error)
        desired_center = uav.step(control)

        # Finally kalman filter and update the UAV's position based on the
        # (noise cleaned-up) position estimate.
        # if the estimation is way off, skip it
        print(
            f"angle: {angle}, {abs(angle)}, translation: {translation}, {np.max(np.abs(translation))}")
        if abs(angle) < 10 and np.max(np.abs(scale)) < 2 and np.max(np.abs(translation)) < 20:
            uav.kalman_correct(desired_center)
            kalman_center = uav.kalman_predict()
            uav.current_position = kalman_center
        else:
            kalman_center = uav.kalman_predict()
            uav.kalman_correct(kalman_center)
            uav.current_position = kalman_center

        # Draw.
        draw.draw(
            template_tensor[0],
            input_tensor[0],
            x,
            y,
            uav.current_position,
            np.array(mp_center),
            desired_center,
        )

    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
