# Doosan M0609 Robot Controller Module
# - Doosan 로봇 SDK를 사용한 로봇 팔 및 OnRobot RG2 그리퍼 제어 모듈
# - 로봇의 기본 자세(Home / Detect) 이동 및 좌표 기반 모션 제어
# - 비전 모듈로부터 전달받은 목표 좌표를 기반으로
#   · XY 이동 (Z 유지)
#   · 손목 각도 정렬
#   · Z축 접근 후 그리퍼로 물체 픽업
#   · Force Control을 이용한 안정적인 안착 및 배치 수행
# - 색상별 그리퍼 폭 매핑을 통해 물감 크기에 따른 픽 동작 지원

import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ultralytics import YOLO
from scipy.spatial.transform import Rotation

from corobot2_project.onrobot import RG

# from onrobot import RG

rclpy.init()
# Configure robot identity
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
node = rclpy.create_node("dsr_seg_grip_move_py", namespace=ROBOT_ID)

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# Doosan SDK init handles
import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = node

# Doosan APIs
try:
    from DSR_ROBOT2 import (
        get_current_posx,
        get_current_posj, 
        wait, 
        DR_MV_MOD_REL,
        release_compliance_ctrl,
        release_force,
        check_force_condition,
        task_compliance_ctrl,
        set_desired_force,
        movej,
        movel,
        moveb,
        DR_FC_MOD_REL,
        DR_AXIS_Z,
        DR_BASE,
        DR_LINE,
    )

    from DR_common2 import posx, posj, posb

except Exception as e:
    raise RuntimeError(f"Failed to import Doosan SDK: {e}")

# =========================
# Robot Controller Module
# =========================

class RobotController:
    def __init__(self, safe_z: float, vel: float = 20.0, acc: float = 20.0):
        self.safe_z = safe_z
        self.vel = vel
        self.acc = acc
        # Example: color-dependent gripper widths (user may tune)
        self.color_to_grip = {
            "Black": 360,
            "White": 360,
            "Yellow": 200,
            "Red": 200,
            "Blue": 200,
        }
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        self.JReady = posj([0, 0, 90, 0, 90, 0])
        self.paint_put_pos = [587.0, -180.0, 35.0, 19.0, -179.0, 19.0]
        self.detect_pos = [727.86, -238.58, 75.62, 161.48, -131.09, -111.12]

    # --- Pose helpers ---
    @staticmethod
    def pose_to_matrix(x, y, z, rx, ry, rz):
        Rm = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def current_posx():
        return get_current_posx()[0]

    # --- Motion primitives ---

    def force_ctrl(self):
        print("Starting task_compliance_ctrl")
        task_compliance_ctrl(stx=[1500, 1500, 1500, 200, 200, 200]) #힘 제어 시작
        time.sleep(0.5)
        fd = [0, 0, -15, 0, 0, 0]
        fctrl_dir= [0, 0, 1, 0, 0, 0]
        print("Starting set_desired_force")
        set_desired_force(fd=fd, dir=fctrl_dir, mod=DR_FC_MOD_REL) 

        # 외력이 0 이상 5 이하이면 0
        # 외력이 5 초과이면 -1
        while not check_force_condition(DR_AXIS_Z, max=5):
            print("Waiting for an external force greater than 5 ")
            time.sleep(0.5)
            pass

        print("Starting release_force")
        release_force()
        time.sleep(0.5)

        print("Starting release_compliance_ctrl")      
        release_compliance_ctrl()

        return True
    
    def trans(self, pos: list[float], delta: list[float], ref: int = DR_BASE, ref_out: int = DR_BASE) -> list[float]:

        if len(pos) != 6 or len(delta) != 6:
            raise ValueError("pos와 delta는 각각 6개의 float 값을 포함하는 리스트여야 합니다.")

        # 결과를 저장할 새로운 리스트 생성
        result_pos = []

        # 각 인덱스에 맞춰 값 더하기
        for i in range(6):
            result_pos.append(pos[i] + delta[i])

        return result_pos
    
    def set_home(self):
        movej(self.JReady, vel=self.vel, acc=self.acc)
    def set_detect(self):
        movel(self.detect_pos, vel=self.vel, acc=self.acc)

    def move_xy_keep_z(self, x, y):
        p = self.current_posx()
        target = posx([x, y, p[2], p[3], p[4], p[5]])
        movel(target, vel=self.vel, acc=self.acc)

    def move_xyz_keep_rpy(self, x, y, z):
        p = self.current_posx()
        target = posx([x, y, z, p[3], p[4], p[5]])
        movel(target, vel=self.vel, acc=self.acc)

    def rotate_wrist_deg(self, delta_deg):
        print(delta_deg, "type : ",type(delta_deg))
        p = get_current_posj()
        if -90 < delta_deg <= 0 :
            p[5] = p[5] + 90 + float(delta_deg)
        elif 0 < delta_deg < 90:
            p[5] = p[5] - 90 + float(delta_deg)
        print(f"p[5] : {p[5]}")
        target = posj(p)
        movej(target, vel=min(self.vel, 20.0), acc=min(self.acc, 20.0))

    def descend_relative(self, dz):
        movel([0, 0, dz, 0, 0, 0], vel=self.vel, acc=self.acc, mod=DR_MV_MOD_REL)

    def pick_sequence(self, grip_width: int, base_xyz, approach_dz: float = -25.0, retreat_dz: float = 40.0):

        """
        주어진 색상의 물감을 집는 동작을 수행합니다.
        Args:
            paint_color (str): 집을 물감의 색상 ('white', 'black', 'yellow', 'red', 'blue').
            approach_dz (float): 접근 시 Z축 상대 이동 거리 (기본값: -25.0).
            retreat_dz (float): 후퇴 시 Z축 상대 이동 거리 (기본값: 40.0).
        """
        
        print("print black", self.color_to_grip['Black'])
        if grip_width == self.color_to_grip['Red']:
            self.gripper.move_gripper(250)
        else:
            self.gripper.move_gripper(410)
        # # Approach (접근)
        # self.descend_relative(approach_dz)
        wait(0.5)
        bx, by, bz = base_xyz.tolist()
        if bz < 9.05:
            bz = 9.05
        elif grip_width == self.color_to_grip['Black']:
            bz -= 20.0
        p = get_current_posx()[0]
        pos = posx([bx, by, bz-0.5, p[3], p[4], p[5]])
        print('bz = ', bz)
        movel(pos, vel=20, acc=20)

        # Gripper (그리퍼) 동작
        # 주석에 따라 물감 색상에 맞는 그리퍼 너비 설정
        print(f"Closing gripper width {grip_width}")
        self.gripper.move_gripper(grip_width)

        wait(0.5)

        # Retreat (후퇴)
        # self.descend_relative(retreat_dz)           # b
        # delta_z = [0, 0, retreat_dz, 0, 0, 0] # 축이동 trans변수
        # cur_pos = get_current_posx()   # base 좌표계에서 현재 TCP pose (posx 형태)
        # pos1 = posb(DR_LINE, self.trans(cur_pos, delta_z), radius=10)

        # movel(p, vel=30, acc=30)                    # b
        pos2 = posb(DR_LINE, p, radius=10)
        # movel(posx([587.0, -180.0, 35.0, 19.0, -179.0, 19.0]), vel=30, acc=30)  # b
        pos3 = posb(DR_LINE, self.paint_put_pos, radius=10)
        b_list = [pos2, pos3]
        moveb(b_list, vel=30, acc=30)
        print('force ctrl')
        if self.force_ctrl():
            self.gripper.move_gripper(500)
            wait(0.2)
        
        # movej(posj([0, 0, 90, 0, 90, 0]), vel=30, acc=30)
        wait(0.2)
        
