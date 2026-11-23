"""Stand-in for the vision in end-to-end runs: publishes a zero offset on /target_TF at 30 Hz.

With a zero offset the task sees "aligned" right away in both stages, so its motion targets
depend only on the program and its config (the golden comparison relies on that). This is
what airbot_play_pick_place.py's __pick_place_test was meant for.
"""

import rospy
from geometry_msgs.msg import TransformStamped

if __name__ == "__main__":
    rospy.init_node("zero_target", anonymous=True)
    pub = rospy.Publisher("/target_TF", TransformStamped, queue_size=1)
    msg = TransformStamped()
    msg.transform.rotation.w = 1.0
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()
