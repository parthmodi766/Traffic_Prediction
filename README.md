Goal:
To create an algorithm that can detect traffic congestion on the roads it is deployed on.

Working:
YOLOv3 object classification model trained to classify only road vehicles was applied to videos of highways collected from the internet. YOLOv3 classified all the road vehicles and created blobs surrounding them. The blobs also possessed the coordinates of the vehicle on the screen. By calculating the time it took for the vehicle to cross pre-determined points on the y-axis and the x-axis we were able to get a rough idea of the speed of the vehicles. By setting a threshold of speed we were able to classify if there was traffic congestion or not.

My Role:
My focus was on training YOLOv3 for the 5 classes of concern. I also came up with idea of using speed of the vehicle to classfiy traffic congestion.

Challenges:
At first we planned to use number of vehicles detected to determine traffic confgestion. But, we noticed that, in this approach the threshold needed to be changed for all the roads on which this algorithm will be used. To solve this problem we applied the approach of measuring speed to detect traffic congestion.

