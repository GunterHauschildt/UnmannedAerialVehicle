# UAV Guidance using Deep Lucas Kanade homography
Yiming Zhao, Xinming Huang and Ziming Zhang did all the real work in
"Deep Lucas-Kanade Homography for Multimodal Image Alignment"

I used their code and their pretrained "GoggleEarth" model inside a PID / Kalman filter control loop
to see if I could track a simulated UAV across Ottawa.

It works, and everything you need should all be in this repo.

From it's map, the mission planner creates a list of images that the UAV should track to get from A to B.
To mock a bottom camera on that UAV, images from the same map are used but warp-distorted and color distorted (with an attempt to make summer look like fall).
The UAV then tracks its image on the mission planner image. With a little PID'ing and Kalman Filter'ing, the UAV self guides.

The more textured the template image, the better the tracking. When we fly over Mooney's bay, its only the Kalman filter that keeps us straight.

The simulated ground speed is very fast. We fly from Westboro to south Bronson in a couple of minutes on my Nvidia 3060 GPU.
(And the mp4 even faster!)

I'm not sure what to do next but there's lots of work to do before, say, putting this on a drone.
