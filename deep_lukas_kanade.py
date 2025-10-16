"""
Utility classes for Deep Lucas-Kanade pipeline.
- DeepLukasKanade: encapsulates networks, LK layers, and inference pipeline
"""
from typing import Tuple
import numpy as np
import tensorflow as tf

from DLK.net import (
    ResNet_first_input,
    ResNet_first_template,
    ResNet_second_input,
    ResNet_second_template,
    ResNet_third_input,
    ResNet_third_template,
    Net_first,
    Net_second,
    Net_third,
    Lucas_Kanade_layer,
)


class DeepLukasKanade:
    """
    Encapsulates math utilities, networks, LK layers, and the full inference pipeline.
    Instance owns:
    - Feature-extraction networks (level one/two/three for input/template)
    - Regression networks (one/two/three)
    - Lucas–Kanade layers for templates at 1x, 1/2x, 1/4x and regression-scale
    Use run_inference(input_img, uav, i_size, t_size, fk_loop) to obtain
    predicted_matrix, x, y.
    """

    def __init__(
        self,
        feature_map_type: str,
        initial_type: str,
        template_size: int,
        input_size: int
    ) -> None:
        self.feature_map_type = feature_map_type
        self.initial_type = initial_type
        self.template_size = template_size
        self.input_size = input_size

        # LK layers
        self.LK_layer_one = Lucas_Kanade_layer(batch_size=1, height_template=template_size,
                                               width_template=template_size, num_channels=1)
        self.LK_layer_two = Lucas_Kanade_layer(batch_size=1, height_template=template_size // 2,
                                               width_template=template_size // 2, num_channels=1)
        self.LK_layer_three = Lucas_Kanade_layer(batch_size=1, height_template=template_size // 4,
                                                 width_template=template_size // 4, num_channels=1)
        self.LK_layer_regression = Lucas_Kanade_layer(batch_size=1, height_template=input_size,
                                                      width_template=input_size, num_channels=3)

        # Feature extraction networks (regular vs special)
        if self.feature_map_type == 'regular':
            self.level_one_input = ResNet_first_input(if_regular=True)
            self.level_one_template = ResNet_first_template(if_regular=True)
            self.level_two_input = ResNet_second_input(if_regular=True)
            self.level_two_template = ResNet_second_template(if_regular=True)
            self.level_three_input = ResNet_third_input(if_regular=True)
            self.level_three_template = ResNet_third_template(if_regular=True)
        else:
            self.level_one_input = ResNet_first_input()
            self.level_one_template = ResNet_first_template()
            self.level_two_input = ResNet_second_input()
            self.level_two_template = ResNet_second_template()
            self.level_three_input = ResNet_third_input()
            self.level_three_template = ResNet_third_template()

        # Regression networks (simple vs multi)
        if self.initial_type == 'simple_net':
            self.regression_network_one = Net_first()
            self.regression_network_two = None
            self.regression_network_three = None
        else:
            self.regression_network_one = Net_first()
            self.regression_network_two = Net_second()
            self.regression_network_three = Net_third()

        # Load weights for DLK networks and LK pipeline using hard-coded defaults
        # (as in original slam from gitlab)
        dataset = 'GoogleEarth'

        if self.feature_map_type == 'regular':
            lvl_one = f'./checkpoints/{dataset}/level_one_regular/'
            lvl_two = f'./checkpoints/{dataset}/level_two_regular/'
            lvl_three = f'./checkpoints/{dataset}/level_three_regular/'
        else:
            lvl_one = f'./checkpoints/{dataset}/level_one/'
            lvl_two = f'./checkpoints/{dataset}/level_two/'
            lvl_three = f'./checkpoints/{dataset}/level_three/'

        ep_one = 10
        ep_two = 10
        ep_three = 10

        ckpts = {
            'level_one_input': lvl_one + f'epoch_{ep_one}input_full',
            'level_one_template': lvl_one + f'epoch_{ep_one}template_full',
            'level_two_input': lvl_two + f'epoch_{ep_two}input_full',
            'level_two_template': lvl_two + f'epoch_{ep_two}template_full',
            'level_three_input': lvl_three + f'epoch_{ep_three}input_full',
            'level_three_template': lvl_three + f'epoch_{ep_three}template_full',
        }

        if self.initial_type == 'simple_net':
            ckpts['reg_one'] = f'./checkpoints/{dataset}/regression_stage_1/' + 'epoch_100'
        else:
            ckpts['reg_one'] = f'./checkpoints/{dataset}/regression_stage_1/' + 'epoch_100'
            ckpts['reg_two'] = f'./checkpoints/{dataset}/regression_stage_2/' + 'epoch_100'
            ckpts['reg_three'] = f'./checkpoints/{dataset}/regression_stage_3/' + 'epoch_80'

        self._maybe_load_weights(ckpts)

    def _maybe_load_weights(self, ckpts: dict) -> None:
        # Expect keys like: level_one_input, level_one_template, ..., reg_one, reg_two, reg_three
        try:
            self.level_one_input.load_weights(ckpts.get("level_one_input", ""))
            self.level_one_template.load_weights(ckpts.get("level_one_template", ""))
            self.level_two_input.load_weights(ckpts.get("level_two_input", ""))
            self.level_two_template.load_weights(ckpts.get("level_two_template", ""))
            self.level_three_input.load_weights(ckpts.get("level_three_input", ""))
            self.level_three_template.load_weights(ckpts.get("level_three_template", ""))
            self.regression_network_one.load_weights(ckpts.get("reg_one", ""))
            self.regression_network_two.load_weights(ckpts.get("reg_two", ""))
            self.regression_network_three.load_weights(ckpts.get("reg_three", ""))

        except (Exception, ) as e:
            print(f"Error loading weights: {e}")
            pass

    @staticmethod
    def construct_matrix_regression(batch_size: int, network_output, network_output_2=[0]):
        extra = tf.ones((batch_size, 1))
        predicted_matrix = tf.concat([network_output, extra], axis=-1)
        predicted_matrix = tf.reshape(predicted_matrix, [batch_size, 3, 3])
        if len(np.shape(network_output_2)) > 1:
            predicted_matrix_2 = tf.concat([network_output_2, extra], axis=-1)
            predicted_matrix_2 = tf.reshape(predicted_matrix_2, [batch_size, 3, 3])
        hh_matrix = []
        for i in range(batch_size):
            if len(np.shape(network_output_2)) > 1:
                hh_matrix.append(
                    np.linalg.inv(np.dot(predicted_matrix_2[i, :, :], predicted_matrix[i, :, :]))
                )
            else:
                hh_matrix.append(np.linalg.inv(predicted_matrix[i, :, :]))
        return np.asarray(hh_matrix)

    @staticmethod
    def construct_matrix(initial_matrix, scale_factor: float, batch_size: int):
        initial_matrix = tf.cast(initial_matrix, dtype=tf.float32)

        scale_matrix = np.eye(3) * scale_factor
        scale_matrix[2, 2] = 1.0
        scale_matrix = tf.cast(scale_matrix, dtype=tf.float32)
        scale_matrix_inverse = tf.linalg.inv(scale_matrix)

        scale_matrix = tf.expand_dims(scale_matrix, axis=0)
        scale_matrix = tf.tile(scale_matrix, [batch_size, 1, 1])

        scale_matrix_inverse = tf.expand_dims(scale_matrix_inverse, axis=0)
        scale_matrix_inverse = tf.tile(scale_matrix_inverse, [batch_size, 1, 1])

        final_matrix = tf.matmul(tf.matmul(scale_matrix, initial_matrix), scale_matrix_inverse)
        return final_matrix

    def corners(
        self,
        batch_size: int,
        predicted_matrix,
     ) -> Tuple[tf.Tensor, tf.Tensor]:

        top_left_u = 0
        top_left_v = 0
        bottom_right_u = self.template_size - 1
        bottom_right_v = self.template_size - 1

        four_corners = [[top_left_u, top_left_v, 1], [bottom_right_u, top_left_v, 1],
                        [bottom_right_u, bottom_right_v, 1], [top_left_u, bottom_right_v, 1]]
        four_corners = np.asarray(four_corners)
        four_corners = np.transpose(four_corners)
        four_corners = np.expand_dims(four_corners, axis=0)
        four_corners = np.tile(four_corners, [batch_size, 1, 1]).astype(np.float32)

        new_four_points = tf.matmul(predicted_matrix, four_corners)
        new_four_points_scale = new_four_points[:, 2:, :]
        new_four_points = new_four_points / new_four_points_scale

        u_predict = new_four_points[:, 0, :]
        v_predict = new_four_points[:, 1, :]
        return u_predict, v_predict

    @staticmethod
    def calculate_feature_map(input_tensor):
        bs, height, width, channel = tf.shape(input_tensor)
        path_extracted = tf.image.extract_patches(
            input_tensor,
            sizes=(1, 3, 3, 1),
            strides=(1, 1, 1, 1),
            rates=(1, 1, 1, 1),
            padding='SAME',
        )
        path_extracted = tf.reshape(path_extracted, (bs, height, width, channel, 9))
        path_extracted_mean = tf.math.reduce_mean(path_extracted, axis=3, keepdims=True)

        path_extracted = path_extracted - path_extracted_mean
        path_extracted_transpose = tf.transpose(path_extracted, (0, 1, 2, 4, 3))
        variance_matrix = tf.matmul(path_extracted_transpose, path_extracted)

        trace_value = tf.linalg.trace(variance_matrix)
        row_sum = tf.reduce_sum(variance_matrix, axis=-1)
        max_row_sum = tf.math.reduce_max(row_sum, axis=-1)
        min_row_sum = tf.math.reduce_min(row_sum, axis=-1)
        mimic_ratio = (max_row_sum + min_row_sum) / 2.0 / trace_value
        return tf.expand_dims(mimic_ratio, axis=-1)

    def run_inference(
        self,
        input_img,
        template_img,
        fk_loop: int = 2,
    ):
        """
        Full inference using internally owned networks and LK layers.
        Expects input_image and template_image as RGB in [0,1].
        Returns (homography, x, y) where x,y are projected corners.
        """

        # Initialize homography via regression nets
        if self.initial_type == 'simple_net':
            input_img_grey = tf.image.rgb_to_grayscale(input_img)
            template_img_new = tf.image.pad_to_bounding_box(template_img, 32, 32, self.input_size, self.input_size)
            template_img_grey = tf.image.rgb_to_grayscale(template_img_new)
            network_input = tf.concat([template_img_grey, input_img_grey], axis=-1)
            homography_vector = self.regression_network_one.call(network_input, training=False)
            extra = tf.ones((1, 1))
            initial_matrix = tf.concat([homography_vector, extra], axis=-1)
            initial_matrix = tf.reshape(initial_matrix, [1, 3, 3])
            initial_matrix = DeepLukasKanade.construct_matrix(initial_matrix, scale_factor=0.25, batch_size=1)
        else:
            input_img_grey = tf.image.rgb_to_grayscale(input_img)
            template_img_new = tf.image.pad_to_bounding_box(template_img, 32, 32, self.input_size, self.input_size)
            template_img_grey = tf.image.rgb_to_grayscale(template_img_new)
            network_input = tf.concat([template_img_grey, input_img_grey], axis=-1)
            homography_vector_one = self.regression_network_one.call(network_input, training=False)
            matrix_one = DeepLukasKanade.construct_matrix_regression(1, homography_vector_one)
            template_img_new = self.LK_layer_regression.projective_inverse_warp(
                tf.dtypes.cast(template_img, tf.float32), matrix_one
            )
            template_img_grey = tf.image.rgb_to_grayscale(template_img_new)
            network_input = tf.concat([template_img_grey, input_img_grey], axis=-1)
            # Guard: require multi-net networks
            if self.regression_network_two is None or self.regression_network_three is None:
                raise RuntimeError("initial_type is 'multi_net' but regression networks are not initialized.")
            homography_vector_two = self.regression_network_two.call(network_input, training=False)
            matrix_two = DeepLukasKanade.construct_matrix_regression(1, homography_vector_one, homography_vector_two)
            template_img_new = self.LK_layer_regression.projective_inverse_warp(
                tf.dtypes.cast(template_img, tf.float32), matrix_two
            )
            template_img_grey = tf.image.rgb_to_grayscale(template_img_new)
            network_input = tf.concat([template_img_grey, input_img_grey], axis=-1)
            homography_vector_three = self.regression_network_three.call(network_input, training=False)

            extra = tf.ones((1, 1))
            initial_matrix = tf.concat([homography_vector_three, extra], axis=-1)
            initial_matrix = tf.reshape(initial_matrix, [1, 3, 3])
            initial_matrix = np.dot(initial_matrix[0, :, :], np.linalg.inv(matrix_two[0, :, :]))
            initial_matrix = np.expand_dims(initial_matrix, axis=0)
            initial_matrix = DeepLukasKanade.construct_matrix(initial_matrix, scale_factor=0.25, batch_size=1)

        # Feature extraction
        input_feature_one = self.level_one_input.call(input_img, training=False)
        template_feature_one = self.level_one_template.call(template_img, training=False)

        input_feature_two = self.level_two_input.call(input_feature_one, training=False)
        template_feature_two = self.level_two_template.call(template_feature_one, training=False)

        input_feature_three = self.level_three_input.call(input_feature_two, training=False)
        template_feature_three = self.level_three_template.call(template_feature_two, training=False)

        # Feature map selection
        if self.feature_map_type == 'regular':
            input_feature_map_one = input_feature_one
            template_feature_map_one = template_feature_one

            input_feature_map_two = input_feature_two
            template_feature_map_two = template_feature_two

            input_feature_map_three = input_feature_three
            template_feature_map_three = template_feature_three
        else:
            input_feature_map_one = DeepLukasKanade.calculate_feature_map(input_feature_one)
            template_feature_map_one = DeepLukasKanade.calculate_feature_map(template_feature_one)

            input_feature_map_two = DeepLukasKanade.calculate_feature_map(input_feature_two)
            template_feature_map_two = DeepLukasKanade.calculate_feature_map(template_feature_two)

            input_feature_map_three = DeepLukasKanade.calculate_feature_map(input_feature_three)
            template_feature_map_three = DeepLukasKanade.calculate_feature_map(template_feature_three)

        # Multi-scale LK updates
        updated_matrix = initial_matrix
        for _ in range(fk_loop):
            try:
                updated_matrix = self.LK_layer_three.update_matrix(
                    template_feature_map_three, input_feature_map_three, updated_matrix
                )
            except (Exception, ):
                pass

        updated_matrix = DeepLukasKanade.construct_matrix(updated_matrix, scale_factor=2.0, batch_size=1)
        for _ in range(fk_loop):
            try:
                updated_matrix = self.LK_layer_two.update_matrix(
                    template_feature_map_two, input_feature_map_two, updated_matrix
                )
            except (Exception, ):
                pass

        updated_matrix = DeepLukasKanade.construct_matrix(updated_matrix, scale_factor=2.0, batch_size=1)
        for _ in range(fk_loop):
            try:
                updated_matrix = self.LK_layer_one.update_matrix(
                    template_feature_map_one, input_feature_map_one, updated_matrix
                )
            except (Exception, ):
                pass

        predicted_matrix = updated_matrix
        x, y = self.corners(1, predicted_matrix)

        return predicted_matrix, x, y

