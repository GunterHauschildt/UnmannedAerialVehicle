# UAV Guidance using Deep Lucas Kanade homography

https://github.com/user-attachments/assets/537bf125-0441-441f-b2c9-c5476f0b7925

Yiming Zhao, Xinming Huang and Ziming Zhang did all the real work in
"Deep Lucas-Kanade Homography for Multimodal Image Alignment"

I used their code and their pretrained model inside a PID / Kalman filter control loop
to see if I could track a simulated UAV across Ottawa.

It works, and everything you need should all be in this repo.

From it's map, the mission planner creates a list of images that the UAV should track to get from A to B.
To mock a bottom camera on that UAV, images from the same map are used but warp-distorted and color distorted (with an attempt to make summer look like fall).
The UAV then tracks its image on the mission planner image. With a little PID'ing and Kalman filter'ing, the UAV self guides.

The more textured the template image, the better the tracking. When we fly over Mooney's Bay, its only the Kalman filter that keeps us straight.

The simulated ground speed is very fast. We fly from Westboro to south Bronson in a couple of minutes on my Nvidia 3060 GPU.

The original work:
https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography
@inproceedings{zhao2021deep,
  title={Deep Lucas-Kanade Homography for Multimodal Image Alignment},
  author={Zhao, Yiming and Huang, Xinming and Zhang, Ziming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15950--15959},
  year={2021}
}
