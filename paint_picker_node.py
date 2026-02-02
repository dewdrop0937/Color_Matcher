# ROS2 Paint Picker Node (3-Layer Architecture)
# - paint_command 토픽을 구독하여 목표 색상 명령을 수신
# - Vision Layer:
#   · YOLO 기반 세그멘테이션으로 대상 페인트 인식
#   · 중심 좌표 및 각도 추정 (RGB-D + RealSense)
# - Robot Control Layer:
#   · 카메라 좌표 → 로봇 베이스 좌표 변환
#   · 안전 Z 유지 이동, XY 정렬, 손목 각도 정렬
#   · 그리퍼로 페인트 집기 후 홈/검출 위치로 복귀
# - ROS Node Layer:
#   · 비전/로봇 제어 모듈을 통합하여 명령 기반 픽앤플레이스 수행

import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ultralytics import YOLO
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# RealSense image node (user-provided class)
from corobot2_project.realsense import ImgNode  # ros2로 실행할 때 사용
from corobot2_project.onrobot import RG
from corobot2_project.visionmodule import VisionModule
from corobot2_project.robotmodule import RobotController
# from visionmodule import VisionModule
# from robotmodule import RobotController
# from realsense import ImgNode  
# from onrobot import RG

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "/home/gjlee/ros2_ws/src/corobot2_project/corobot2_project/best_v8.pt"
T_gripper2cam_PATH = "/home/gjlee/ros2_ws/src/corobot2_project/corobot2_project/T_gripper2camera.npy"

ROBOT_ID = "dsr01"

# =========================
# ROS Interface Node
# =========================

class PaintPickNode(Node):
    def __init__(self, robot_id: str, model_path: str):
        super().__init__("paint_picker_node", namespace=robot_id)

        # --- Params ---
        self.declare_parameter("model_path", model_path)
        self.declare_parameter("safe_z", 300.0)
        self.declare_parameter("vel", 30.0)
        self.declare_parameter("acc", 30.0)
        self.declare_parameter("debug_save", True) ## True / False 설정으로 이미지 저장

        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        safe_z = float(self.get_parameter("safe_z").get_parameter_value().double_value)
        vel = float(self.get_parameter("vel").get_parameter_value().double_value)
        acc = float(self.get_parameter("acc").get_parameter_value().double_value)
        debug_save = bool(self.get_parameter("debug_save").get_parameter_value().bool_value)

        # --- Camera node ---
        self.img_node = ImgNode()
        self.bridge = CvBridge()

        time.sleep(0.5)
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        self.get_logger().info(f"Camera intrinsics: {self.intrinsics}")

        # --- TF: gripper -> camera ---
        self.T_gripper2cam = np.load(T_gripper2cam_PATH)

        
        # --- Modules ---
        self.vision = VisionModule(self.model_path, self.intrinsics, debug_save=debug_save)
        self.robot = RobotController(safe_z=safe_z, vel=vel, acc=acc)

        # --- Runtime state ---
        self.is_busy = False

        # --- Subscriptions ---
        self.sub_cmd = self.create_subscription(String, "paint_command", self.on_paint_command, 10)

        # --- Ready pose ---
        self.robot.set_home()
        self.get_logger().info("PaintPickNode is ready.")

    def _wait_for_valid_data(self, getter, description):
        """getter 함수가 유효한 데이터를 반환할 때까지 spin 하며 재시도합니다."""
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data
    

    # -------------- Helpers --------------
    # @staticmethod
    def base_pose_matrix(self):
        px = self.robot.current_posx()
        return self.robot.pose_to_matrix(*px)

    def camera_to_base(self, cam_xyz: np.ndarray):
        coord = np.append(cam_xyz, 1.0)
        T_base2gripper = self.base_pose_matrix()
        T_base2cam = T_base2gripper @ self.T_gripper2cam
        # td = T_base2cam @ coord           # 강사님이 주신 코드와 차이
        td = np.dot(T_base2cam, coord)      # 강사님이 주신 코드
        return td[:3]

    def fetch_frames(self):
        time.sleep(0.2)
        color = self.img_node.get_color_frame()
        depth = self.img_node.get_depth_frame()
        return color, depth

    def parse_color_tokens(self, s: str):
        paint_name_map = {
            "black_paint": "Black",
            "white_paint": "White",
            "yellow_paint": "Yellow",
            "red_paint": "Red",
            "blue_paint": "Blue",
        }
        tokens = s.split()
        mapped = [paint_name_map.get(t, t) for t in tokens]
        # Only allow known target colors
        allowed = {"Black", "Blue", "Red", "White", "Yellow"}
        print("tokens_str =", tokens)
        print("mapped =", mapped)
        return [c for c in mapped if c in allowed]

    def debug_frame(self, target_color, stri):
        print("debug_frame")
        time.sleep(1)

        color, depth = self.fetch_frames()
        if color is None or depth is None:
            self.get_logger().warn("Frames not available.")
            return None, None, None, None
        
        det = self.vision.find_target(color, depth, target_color)
        if det is None:
            self.get_logger().info(f"No detection for {target_color}.")
            return None, None, None, None
        
        cam_xyz = det["cam_xyz"]
        angle_deg = det["angle_deg"]
        print(angle_deg)

        base_xyz = self.camera_to_base(cam_xyz)
        if base_xyz is None or len(base_xyz) < 3:
            self.get_logger().error(f"camera_to_base returned invalid value: {base_xyz}")
            return None, None, None, None
        
        bx, by, bz = base_xyz[:3]
        self.get_logger().info(
            f"Target {target_color}: base XYZ=({bx:.1f}, {by:.1f}, {bz:.1f}), angle={angle_deg:.1f}"
        )
        self.vision.save_debug(color, det, target_color, stage=stri)
        return bx, by, bz, det

    # -------------- Main pipeline --------------
    def process_target_color(self, target_color: str):
        self.get_logger().info(f"Processing target: {target_color}")
        self.robot.set_home()
        # Step 1: Read frames & detect target
        time.sleep(1)
        bx, by, bz, det = self.debug_frame(target_color, 'first')
        print(bx, by, bz, det)
        if not det:
            print("det is not")
            return False
        
        # Step 2: Safe motion sequence (Z-up → XY → wrist align)
        # self.robot.go_safe_z()
        self.robot.move_xy_keep_z(bx, by)

        # Optional: refine angle after move (re-acquire)
        bx2, by2, bz2, det2 = self.debug_frame(target_color, 'second')
        if not det2:
            print('error')
            return False
        self.robot.rotate_wrist_deg(det2['angle_deg'])
        time.sleep(1)
            
        bx3, by3, bz3, det3 = self.debug_frame(target_color, 'third')
        if not det3:
            return False

        # Step 3: (Optional) Pick sequence
        grip_width = self.robot.color_to_grip.get(target_color, 220)
        print(f"grip_width : {grip_width}")
        # Move down close to object height if desired
        # Here we keep Z and perform a standard approach
        self.robot.pick_sequence(grip_width=grip_width, base_xyz = self.camera_to_base(det3['cam_xyz']), approach_dz=-25.0, retreat_dz=40.0)

        return True

    # -------------- ROS Callbacks --------------
    def on_paint_command(self, msg: String):
        if self.is_busy:
            self.get_logger().warn("Busy. Ignoring new command.")
            return
        self.is_busy = True

        try:
            raw = msg.data.strip()
            self.get_logger().info(f"Received command: {raw}")

            # "/" 로 나눠서 각각 색상 블록을 처리
            blocks = raw.split("/")
            targets = []

            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                # 색상 토큰 파싱 함수 활용
                parsed = self.parse_color_tokens(block)
                print('parsed : ', parsed)
                if parsed:
                    targets.extend(parsed)
            print(f"targets: {targets}")
            print(f"targets: {type(targets)}")

            if not targets:
                self.get_logger().info("No valid target colors in command.")
                return

            # 순서대로 색상 pick
            for color_name in targets:
                print('color name', color_name)
                ok = self.process_target_color(color_name)
                print('ok')
                if not ok:
                    self.get_logger().info(f"Skip/failed: {color_name}")

            # Return to detect pose
            self.robot.set_detect()

        except Exception as e:
            self.get_logger().error(f"Error handling command: {e}")
        finally:
            self.is_busy = False


# =========================
# Main
# =========================
from rclpy.executors import MultiThreadedExecutor

def main():
    # Create an auxiliary ROS node for Doosan SDK if required by your stack
    node = PaintPickNode(robot_id=ROBOT_ID, model_path=MODEL_PATH)
    executor = MultiThreadedExecutor()
    # 노드 추가
    executor.add_node(node)
    executor.add_node(node.img_node)
    try:
        executor.spin()  # 멀티스레드 실행
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        node.img_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
