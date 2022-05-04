#!/usr/bin/env python3

import os
from typing import Optional

import cv2
import numpy as np
import yaml

import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Segment, SegmentList
from geometry_msgs.msg import Point as PointMsg
# from image_processing.ground_projection_geometry import GroundProjectionGeometry, Point
# from image_processing.rectification import Rectify
from sensor_msgs.msg import CameraInfo, CompressedImage

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.camera.types import Rectifier, Pixel, NormalizedImagePoint, Point
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.ground_projection.types import GroundPoint


class GroundProjectionNode(DTROS):
    """
    This node projects the line segments detected in the image to the ground plane and in the robot's
    reference frame.
    In this way it enables lane localization in the 2D ground plane. This projection is performed using the
    homography
    matrix obtained from the extrinsic calibration procedure.

    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Subscribers:
        ~camera_info (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera. Needed for
        rectifying the segments.
        ~lineseglist_in (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
        unrectified images

    Publishers:
        ~lineseglist_out (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in the ground plane
        relative to the robot origin
        ~debug/ground_projection_image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Debug image
        that shows the robot relative to the projected segments. Useful to check if the extrinsic
        calibration is accurate.
    """

    bridge: CvBridge
    ground_projector: Optional[GroundProjector]
    # rectifier: Optional[Rectifier]
    camera: Optional[CameraModel]

    def __init__(self, node_name: str):
        # Initialize the DTROS parent class
        super(GroundProjectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.bridge = CvBridge()
        self.ground_projector = None
        # self.rectifier = None
        self.camera = None
        self.homography = np.reshape(self.load_extrinsics(), (3, 3))
        self.first_processing_done = False
        self.camera_info_received = False

        # subscribers
        self.sub_camera_info = rospy.Subscriber("~camera_info", CameraInfo, self.cb_camera_info, queue_size=1)
        self.sub_lineseglist_ = rospy.Subscriber(
            "~lineseglist_in", SegmentList, self.lineseglist_cb, queue_size=1
        )

        # publishers
        self.pub_lineseglist = rospy.Publisher(
            "~lineseglist_out", SegmentList, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )
        self.pub_debug_img = rospy.Publisher(
            "~debug/ground_projection_image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        self.bridge = CvBridge()

        self.debug_img_bg = None


    def cb_camera_info(self, msg: CameraInfo):
        """
        Initializes a :py:class:`image_processing.GroundProjectionGeometry` object and a
        :py:class:`image_processing.Rectify` object for image rectification

        Args:
            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera.

        """
        if not self.camera_info_received:
            _K = np.reshape(msg.K, (3,3)).tolist()
            # _K[0][2] = _K[0][2] - x
            # _K[1][2] = _K[1][2] - y
            _P = np.reshape(msg.P, (3, 4)).tolist()
            # _P[0][2] = _P[0][2] - x
            # _P[1][2] = _P[1][2] - y  # TODO: cropped x, y

            self.camera = CameraModel(
                width=msg.width,  # TODO: cropped size
                height=msg.height,
                K=_K,
                D=msg.D,
                P=_P,
                H=self.homography,
            )
            # self.rectifier = camera.rectifier
            self.ground_projector = GroundProjector(camera=self.camera)
        self.camera_info_received = True

    def lineseglist_cb(self, seglist_msg):
        """
        Projects a list of line segments on the ground reference frame point by point by
        calling :py:meth:`pixel_msg_to_ground_msg`. Then publishes the projected list of segments.

        Args:
            seglist_msg (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
            unrectified images

        """
        def _fill_point_msg(pt: Point) -> PointMsg:
            p_msg = PointMsg()
            p_msg.x = pt.x
            p_msg.y = pt.y
            p_msg.z = 0
            return p_msg

        if self.camera_info_received:
            seglist_out = SegmentList()
            seglist_out.header = seglist_msg.header
            for received_segment in seglist_msg.segments:
                new_segment = Segment()
                # distorted pixels
                l0, l1 = received_segment.pixels_normalized
                p0: Pixel = Pixel(l0[0], l0[1])
                p1: Pixel = Pixel(l1[2], l1[3])
                # distorted pixels to rectified pixels
                p0_rect: Pixel = self.camera.rectifier.rectify_pixel(p0)
                p1_rect: Pixel = self.camera.rectifier.rectify_pixel(p1)
                # rectified pixel to normalized coordinates
                p0_norm: NormalizedImagePoint = self.camera.pixel2vector(p0_rect)
                p1_norm: NormalizedImagePoint = self.camera.pixel2vector(p1_rect)
                # project image point onto the ground plane
                grounded_p0: GroundPoint = self.ground_projector.vector2ground(p0_norm)
                grounded_p1: GroundPoint = self.ground_projector.vector2ground(p1_norm)

                new_segment.points[0] = _fill_point_msg(grounded_p0)
                new_segment.points[1] = _fill_point_msg(grounded_p1)

                new_segment.color = received_segment.color
                # TODO what about normal and points <= here before ente-new-deal
                seglist_out.segments.append(new_segment)
            self.pub_lineseglist.publish(seglist_out)

            if not self.first_processing_done:
                self.log("First projected segments published.")
                self.first_processing_done = True

            if self.pub_debug_img.get_num_connections() > 0:
                debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(self.debug_image(seglist_out))
                debug_image_msg.header = seglist_out.header
                self.pub_debug_img.publish(debug_image_msg)
        else:
            self.log("Waiting for a CameraInfo message", "warn")

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.

        Returns:
            :obj:`numpy array`: the loaded homography matrix

        """
        # load intrinsic calibration
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(
                f"Can't find calibration file: {cali_file}.\n Using default calibration instead.", "warn"
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = "Found no calibration file ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file, "r") as stream:
                calib_data = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError:
            msg = f"Error in parsing calibration file {cali_file} ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        return calib_data["homography"]

    def debug_image(self, seg_list):
        """
        Generates a debug image with all the projected segments plotted with respect to the robot's origin.

        Args:
            seg_list (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in the ground plane relative
            to the robot origin

        Returns:
            :obj:`numpy array`: an OpenCV image

        """
        # dimensions of the image are 1m x 1m so, 1px = 2.5mm
        # the origin is at x=200 and y=300

        # if that's the first call, generate the background
        if self.debug_img_bg is None:

            # initialize gray image
            self.debug_img_bg = np.ones((400, 400, 3), np.uint8) * 128

            # draw vertical lines of the grid
            for vline in np.arange(40, 361, 40):
                cv2.line(
                    self.debug_img_bg, pt1=(vline, 20), pt2=(vline, 300), color=(255, 255, 0), thickness=1
                )

            # draw the coordinates
            cv2.putText(
                self.debug_img_bg,
                "-20cm",
                (120 - 25, 300 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                self.debug_img_bg,
                "  0cm",
                (200 - 25, 300 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                self.debug_img_bg,
                "+20cm",
                (280 - 25, 300 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (255, 255, 0),
                1,
            )

            # draw horizontal lines of the grid
            for hline in np.arange(20, 301, 40):
                cv2.line(
                    self.debug_img_bg, pt1=(40, hline), pt2=(360, hline), color=(255, 255, 0), thickness=1
                )

            # draw the coordinates
            cv2.putText(
                self.debug_img_bg, "20cm", (2, 220 + 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1
            )
            cv2.putText(
                self.debug_img_bg, " 0cm", (2, 300 + 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1
            )

            # draw robot marker at the center
            cv2.line(
                self.debug_img_bg,
                pt1=(200 + 0, 300 - 20),
                pt2=(200 + 0, 300 + 0),
                color=(255, 0, 0),
                thickness=1,
            )

            cv2.line(
                self.debug_img_bg,
                pt1=(200 + 20, 300 - 20),
                pt2=(200 + 0, 300 + 0),
                color=(255, 0, 0),
                thickness=1,
            )

            cv2.line(
                self.debug_img_bg,
                pt1=(200 - 20, 300 - 20),
                pt2=(200 + 0, 300 + 0),
                color=(255, 0, 0),
                thickness=1,
            )

        # map segment color variables to BGR colors
        color_map = {Segment.WHITE: (255, 255, 255), Segment.RED: (0, 0, 255), Segment.YELLOW: (0, 255, 255)}

        image = self.debug_img_bg.copy()

        # plot every segment if both ends are in the scope of the image (within 50cm from the origin)
        for segment in seg_list.segments:
            if not np.any(
                np.abs([segment.points[0].x, segment.points[0].y, segment.points[1].x, segment.points[1].y])
                > 0.50
            ):
                cv2.line(
                    image,
                    pt1=(int(segment.points[0].y * -400) + 200, int(segment.points[0].x * -400) + 300),
                    pt2=(int(segment.points[1].y * -400) + 200, int(segment.points[1].x * -400) + 300),
                    color=color_map.get(segment.color, (0, 0, 0)),
                    thickness=1,
                )

        return image


if __name__ == "__main__":
    ground_projection_node = GroundProjectionNode(node_name="ground_projection")
    rospy.spin()
