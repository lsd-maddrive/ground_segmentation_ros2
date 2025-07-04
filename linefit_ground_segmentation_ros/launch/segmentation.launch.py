
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node



def generate_launch_description():
        # Getting directories and launch-files
        bringup_dir = get_package_share_directory('linefit_ground_segmentation_ros')
        params_file = os.path.join(bringup_dir, 'launch', 'segmentation_params.yaml')

        # Nodes launching commands
        node_start_cmd = Node(
                package='linefit_ground_segmentation_ros',
                executable='ground_segmentation_node',
                output='screen',
                parameters=[params_file])

        Node(
        package="topic_tools", executable="relay",
        name="ground_relay",
        arguments=[
                "/segmentation/ground", "/segmentation/ground_reliable",
                "--qos-profile", "rmw_qos_profile_reliable"
        ],
        output="screen",
        )


        ld = LaunchDescription()

        # Declare the launch options
        ld.add_action(node_start_cmd)


        return ld