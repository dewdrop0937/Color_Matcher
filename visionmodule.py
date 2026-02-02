# Vision Node for Paint Pick-and-Place System
# - paint_command 토픽에서 전달된 목표 물체 정보를 기반으로
#   카메라 영상에서 해당 물감을 인식하고 위치 및 각도를 추정하는 비전 노드
# - YOLO Segmentation 모델을 사용하여 물감 객체를 검출
# - 마스크 기반 minAreaRect를 이용해 물감의 회전 각도 계산
# - Depth 영상과 카메라 내부 파라미터를 이용해 3D 카메라 좌표 추정
# - 추정된 (x, y, z) 위치와 각도 정보를 로봇 제어 노드로 전달하기 위한 입력 생성
#
# 본 노드는 전체 시스템의 3-레이어 구조 중 Vision 레이어에 해당하며,
# 로봇 제어 및 ROS 통신과 분리된 순수 인식 기능을 담당한다.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import cv2

from ultralytics import YOLO

# =========================
# Utility: angle from mask
# =========================



def calc_angle_from_mask_rotated(mask: np.ndarray):
    """Return (angle_deg, (cx, cy), box_points) using minAreaRect.
    OpenCV returns angle in [-90, 0) for rectangles; we normalize for long-axis alignment.
    """
    ys, xs = np.where(mask > 0.5)
    if xs.size < 2:
        return None, None, None

    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect).astype(np.int32)

    # Normalize so that angle describes the object's long axis orientation.
    if w < h:
        angle = angle + 90.0
    # Map to [-90, 90) range for stability.
    if angle > 90:
        angle -= 180
    if angle <= -90:
        angle += 180
 
    return float(angle), (int(cx), int(cy)), box


# =========================
# Vision Module
# =========================


class VisionModule:
    def __init__(self, model_path: str, intrinsics: dict, debug_save: bool = True):
        self.model = YOLO(model_path)
        self.intrinsics = intrinsics  # dict with fx, fy, ppx, ppy
        self.debug_save = debug_save
        self.class_names = self.model.names  # id -> name
        self.debug_dir = "/home/gjlee/ros2_ws/src/corobot2_project/corobot2_project/debug_img_folder"

    def infer(self, color_frame):
        return self.model.predict(color_frame, imgsz=640, conf=0.5, verbose=False)

    def pixel_to_camera(self, u: int, v: int, z: float):
        fx = self.intrinsics["fx"]; fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]; ppy = self.intrinsics["ppy"]
        X = (u - ppx) * z / fx
        Y = (v - ppy) * z / fy
        Z = z
        return np.array([X, Y, Z], dtype=float)

    def depth_median_in_mask(self, depth_frame: np.ndarray, mask: np.ndarray):
        m = mask > 0.5
        if not np.any(m):
            return None
        vals = depth_frame[m]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def find_target(self, color_frame, depth_frame, target_class_name: str):
        """Return dict with keys: angle_deg, cx, cy, cam_xyz, mask, box, score.
        None if not found.
        """
        res = self.infer(color_frame)[0]
        if res.masks is None or res.boxes is None:
            return None

        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        masks = res.masks.data.cpu().numpy()
        names = self.class_names
        print(names)
        # pick highest conf among target_class
        best = None
        for i, cid in enumerate(class_ids):
            det_name = names[int(cid)]
            if det_name != target_class_name:
                continue
            mask = masks[i]
            angle, (cx, cy), box = calc_angle_from_mask_rotated(mask)
            if angle is None:
                continue
            z_med = self.depth_median_in_mask(depth_frame, mask)
            if z_med is None or np.isnan(z_med) or z_med <= 0:
                continue
            cam_xyz = self.pixel_to_camera(cx, cy, z_med)
            cand = {
                "angle_deg": angle,
                "cx": int(cx),
                "cy": int(cy),
                "cam_xyz": cam_xyz,
                "mask": mask,
                "box": box,
                "score": float(scores[i])
            }
            if (best is None) or (cand["score"] > best["score"]):
                best = cand

        if best and self.debug_save:
            print('save img best')
            dbg = res.plot()
            cv2.circle(dbg, (best["cx"], best["cy"]), 6, (0, 0, 255), 2)
            cv2.drawContours(dbg, [best["box"]], -1, (0, 255, 0), 2)
            cv2.putText(dbg, f"{target_class_name} angle={best['angle_deg']:.1f}",
                        (best["cx"], max(0, best["cy"]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            os.makedirs("/tmp/paint_picker_dbg", exist_ok=True)
            cv2.imwrite("/tmp/paint_picker_dbg/detect.jpg", dbg)
            print('save img done')

        return best


    def save_debug(self, frame, det, name, stage="before"):
        print(f'save debug_img{name} {stage}')
        dbg = frame.copy()
        cv2.drawContours(dbg, [det["box"]], -1, (0, 255, 0), 2)
        cv2.putText(
            dbg,
            f"{name} {stage} angle={det['angle_deg']:.1f}",
            (det["cx"] - 10, max(0, det["cy"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        save_path = f"{self.debug_dir}/{name}_{stage}.jpg"
        success = cv2.imwrite(save_path, dbg)

        if success:
            print(f"[INFO] Debug image saved: {save_path}")
        else:
            print(f"[ERROR] Failed to save debug image: {save_path}")
