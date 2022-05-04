#!/usr/bin/env python3

from multiprocessing import Lock

import rospy
from cv_bridge import CvBridge
from dt_computer_vision.anti_instagram import AntiInstagram
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import AntiInstagramThresholds
from sensor_msgs.msg import CompressedImage


class AntiInstagramNode(DTROS):
    """

    Subscriber:
        ~uncorrected_image/compressed: The uncompressed image coming from the camera


    Publisher:
        ~

    """

    def __init__(self, node_name):

        super(AntiInstagramNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Read parameters
        self._interval = rospy.get_param("~interval")
        self._color_balance_percentage = rospy.get_param("~color_balance_scale")
        self._output_scale = rospy.get_param("~output_scale")

        # Construct publisher
        self.pub = rospy.Publisher(
            "~thresholds", AntiInstagramThresholds, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )

        # Construct subscriber
        self.uncorrected_image_subscriber = rospy.Subscriber(
            "~uncorrected_image/compressed",
            CompressedImage,
            self.store_image_msg,
            buff_size=10000000,
            queue_size=1,
        )

        # Initialize Timer
        rospy.Timer(rospy.Duration(self._interval), self.calculate_new_parameters)

        # Initialize objects and data
        self.ai = AntiInstagram(image_scale=self._output_scale, color_balance_scale=self._color_balance_percentage)
        self.bridge = CvBridge()
        self.image_msg = None
        self.mutex = Lock()

        # ---
        self.log("Initialized.")

    def store_image_msg(self, image_msg):
        with self.mutex:
            self.image_msg = image_msg

    def decode_image_msg(self):
        with self.mutex:
            try:
                image = self.bridge.compressed_imgmsg_to_cv2(self.image_msg, "bgr8")
            except ValueError as e:
                self.log(f"Anti_instagram cannot decode image: {e}")
        return image

    def calculate_new_parameters(self, event):
        if self.image_msg is None:
            self.log("Waiting for first image!")
            return
        # Update thresholds
        self.ai.update(self.decode_image_msg())
        # Publish parameters
        msg = AntiInstagramThresholds()
        msg.low = self.ai.lower_threshold
        msg.high = self.ai.higher_threshold
        self.pub.publish(msg)


if __name__ == "__main__":
    node = AntiInstagramNode(node_name="anti_instagram_node")
    rospy.spin()
