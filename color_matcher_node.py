# ROS2 Wakeup Word Node
# - 마이크 입력을 기반으로 wakeup word를 실시간 감지
# - wakeup word 인식 시 STT 및 색상 파싱 노드를 서비스로 트리거

import cv2
import numpy as np
import math
import time
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import queue
import json

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# -------------------------------------------------------------------
# Helper Functions and Classes (from original script)
# -------------------------------------------------------------------

class KalmanFilter1D:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        self.x = 0.0
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, measurement):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x

def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0,0,0,1
    r_,g_,b_ = r/255.0, g/255.0, b/255.0
    k = 1 - max(r_,g_,b_)
    c = (1 - r_ - k)/(1-k+1e-10)
    m = (1 - g_ - k)/(1-k+1e-10)
    y = (1 - b_ - k)/(1-k+1e-10)
    return c,m,y,k

def rgb_to_lightness(r,g,b):
    r_,g_,b_ = r/255.0, g/255.0, b/255.0
    return (max(r_,g_,b_)+min(r_,g_,b_))/2

def color_distance(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def find_missing_paint_with_ratio(current_rgb, target_rgb):
    dist = color_distance(current_rgb, target_rgb)
    if dist > 300:
        return [("Color creation difficult", None)], dist
    cur_cmyk = rgb_to_cmyk(*current_rgb)
    tgt_cmyk = rgb_to_cmyk(*target_rgb)
    c_diff = tgt_cmyk[0] - cur_cmyk[0]
    m_diff = tgt_cmyk[1] - cur_cmyk[1]
    y_diff = tgt_cmyk[2] - cur_cmyk[2]
    res = []
    if c_diff > 0.08 or m_diff < -0.5:
        res.append(("Blue", abs(c_diff)*100))
    if m_diff > 0.08 or c_diff < -0.5:
        res.append(("Red", abs(m_diff)*100))
    if y_diff > 0.08:
        res.append(("Yellow", abs(y_diff)*100))
    cur_light = rgb_to_lightness(*current_rgb)
    tgt_light = rgb_to_lightness(*target_rgb)
    ld = tgt_light - cur_light
    if ld > 0.08:
        res.append(("White", abs(ld)*100))
    elif ld < -0.08:
        res.append(("Black", abs(ld)*100))
    return res, dist

def draw_paint_need_graph(missing_paints):
    graph_h, graph_w = 200, 400
    graph = np.zeros((graph_h, graph_w, 3), dtype=np.uint8) + 30
    paint_colors = {"Blue":(255,0,0),"Red":(0,0,255),"Yellow":(0,255,255),"White":(255,255,255),"Black":(0,0,0)}
    bar_x = 120; bar_h = 25; gap = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i,(paint,ratio) in enumerate(missing_paints):
        y = i*(bar_h+gap)+20
        cv2.putText(graph, paint, (10,y+bar_h-5), font, 0.6, (255,255,255),2)
        if ratio is None:
            cv2.putText(graph,"Fail",(bar_x,y+bar_h-5),font,0.6,(0,0,255),2)
            continue
        bar_len = int(min((ratio/100.0)*250, 250))
        color = paint_colors.get(paint,(255,255,255))
        cv2.rectangle(graph,(bar_x,y),(bar_x+bar_len,y+bar_h),color,-1)
        cv2.putText(graph,f"{ratio:.1f}%",(bar_x+bar_len+10,y+bar_h-5),font,0.5,(255,255,255),1)
    return graph


from dotenv import load_dotenv
import os
from openai import OpenAI

# .env 파일에서 OPENAI_API_KEY 불러오기
load_dotenv(dotenv_path='/home/gjlee/ros2_ws/src/corobot2_project/corobot2_project/.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

class ColorUpdater:
    def __init__(self):
        self.client = OpenAI(api_key=openai_api_key)  # 환경변수로 불러오는게 안전함
        self.color_map = {
            "빨간색": (255, 0, 0),
            "노란색": (255, 255, 0),
            "파란색": (0, 0, 255),
            "연두색": (57, 128, 71),
            "보라색": (122, 82, 122),
            "청보라색": (155, 155, 210)
        }

    def get_rgb_from_openai(self, color_name: str):
        """
        OpenAI API를 호출해서 한국어 색상명을 RGB 값으로 변환
        """
        prompt = f"""
        너는 색상 전문가야. 
        한국어 색상 이름 "{color_name}"에 가장 적합한 RGB 값을 정수 튜플 (R, G, B)로만 출력해.
        예시: (255, 0, 0)
        {color_name} 값이 조색x 또는 색상이 아닌경우 (0, 0, 0)
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # 응답에서 RGB 값만 파싱
        content = response.choices[0].message.content.strip()
        rgb_tuple = eval(content)  # ex: "(255, 0, 0)" → (255,0,0)
        return rgb_tuple

    def update_color_map(self, target_color: str, prev_value=(0,0,0)):
        """
        딕셔너리에 색상이 없으면 OpenAI로 RGB 생성 후 추가
        단, RGB 값이 (0, 0, 0)이면 업데이트하지 않고 이전 값 반환
        """
        if target_color not in self.color_map:
            rgb_value = self.get_rgb_from_openai(target_color)

            if rgb_value == (0, 0, 0):
                # 업데이트하지 않고 이전 값 반환
                print(f"[무시됨] {target_color}: RGB 값이 (0, 0, 0)이라 업데이트하지 않음")
                return prev_value, False
            else:
                self.color_map[target_color] = rgb_value
                print(f"[업데이트됨] {target_color}: {rgb_value}")
                return rgb_value, True
        else:
            rgb_value = self.color_map[target_color]
            print(f"[이미 존재] {target_color}: {rgb_value}")
            return rgb_value, False
# -------------------------------------------------------------------
# ROS2 Node Class
# -------------------------------------------------------------------

class ColorMatcherNode(Node):
    def __init__(self):
        super().__init__('color_matcher_node')

        # --- cv_bridge 초기화 ---
        self.bridge = CvBridge()
        self.latest_frame = None  # 최신 이미지를 저장할 변수
        self.color_update = ColorUpdater()
        # --- ROS2 이미지 구독 ---
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # --- ROS2 Communication ---
        self.subscription = self.create_subscription(
            String,
            '/dsr01/target_paint_color',
            self.target_color_callback,
            10)
        
        # --- GUI Setup (in a separate thread) ---
        self.gui_queue = queue.Queue() # Queue to send data to the GUI thread
        self.gui_thread = threading.Thread(target=self.run_gui, daemon=True)
        self.gui_thread.start()

        self.use_kalman = None
        self.use_white_balance = None
        self.use_gaussian = None
        self.use_hist_eq = None
        self.use_sharpen = None
        self.use_median = None
        self.use_bilateral = None
        self.use_hsv_analysis = None

        # ret, frame = self.cap.read()
        # h,w = frame.shape[:2]
        # self.patch_w, self.patch_h = 50,50
        # self.white_patch_coords = (w-self.patch_w, 0, self.patch_w, self.patch_h)

        # --- Filters & Settings ---
        self.kf_r, self.kf_g, self.kf_b = KalmanFilter1D(), KalmanFilter1D(), KalmanFilter1D()
        # self.fixed_kb, self.fixed_kg, self.fixed_kr = 2.129, 2.176, 2.182
        self.fixed_kb, self.fixed_kg, self.fixed_kr = 2.08, 2.08, 2.08
        self.white_balance_fixed_k = 1.06
        # self.target_color = (196, 147, 245) # Initial target color
        self.target_color = (255, 255, 255) # Initial target color
        self.color_map = {
            "빨간색": (255, 0, 0), "노란색": (255, 255, 0), "파란색": (0, 0, 255),
            "연두색": (57, 128, 71), "보라색": (122, 82, 122), "청보라색": (155, 155, 210)
        }
        
        # --- State Variables ---
        self.last_log_time = time.time() - 2.0
        self.log_interval = 2.0
        self.match_percentage = None

        self.get_logger().info("Color Matcher Node has started.")

    # image callback
    def image_callback(self, msg):
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 변환 실패: {e}")
            return
        
        self.latest_frame = frame
        self.process_frame(frame)

    def target_color_callback(self, msg):
        self.get_logger().info(f'Received message on final_colors_topic: "{msg.data}"')
        
        # 문자열 처리
        color_name = msg.data.strip().lower()  # 앞뒤 공백 제거, 소문자 변환
        self.log_to_gui(f"Received color: '{color_name}'")

        # 이전 target_color 값을 prev_value로 전달
        prev_value = getattr(self, 'target_color', (0, 0, 0))
        
        # ColorUpdater 활용
        rgb_value, is_new = self.color_update.update_color_map(color_name, prev_value)
        self.target_color = rgb_value

        if is_new:
            self.log_to_gui(f"New color added: {color_name} -> {self.target_color}")
        else:
            self.log_to_gui(f"Target color changed to: {color_name} -> {self.target_color}")

    def process_frame(self, frame):
        processed_frame = frame.copy()

        # Apply Median Filter if enabled
        # (미디안 필터 적용)
        if self.use_median and self.use_median.get():
            processed_frame = self.median_filter(processed_frame)

        # Apply Gaussian Blur if enabled
        # (가우시안 블러 적용)
        if self.use_gaussian and self.use_gaussian.get():
            processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)

        # Apply Bilateral Filter if enabled
        # (양방향 필터 적용)
        if self.use_bilateral and self.use_bilateral.get():
            processed_frame = self.bilateral_filter(processed_frame)

        # Apply Histogram Equalization if enabled
        # (히스토그램 평준화 적용)
        if self.use_hist_eq and self.use_hist_eq.get():
            processed_frame = self.equalize_histogram_color(processed_frame)

        # Apply Sharpening if enabled
        # (샤프닝 적용)
        if self.use_sharpen and self.use_sharpen.get():
            processed_frame = self.sharpen_image(processed_frame)

        # Apply White Balance if enabled
        # (화이트 밸런스 적용)
        if self.use_white_balance and self.use_white_balance.get():
            processed_frame = self.white_balance_fixed(processed_frame, self.fixed_kb, self.fixed_kg, self.fixed_kr, self.white_balance_fixed_k)

        # --- Color Analysis (색상 분석) ---
        h, w = processed_frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Get BGR color from the center pixel
        # (중심 픽셀에서 BGR 색상 값 가져오기)
        b, g, r = processed_frame[cy, cx]
        
        # Apply Kalman Filter if enabled
        # (칼만 필터 적용)
        if self.use_kalman and self.use_kalman.get():
            r_f = self.kf_r.update(float(r))
            g_f = self.kf_g.update(float(g))
            b_f = self.kf_b.update(float(b))
            current_rgb_int = (int(r_f), int(g_f), int(b_f))
        else:
            current_rgb_int = (int(r), int(g), int(b))

        # --- Logging and Graphing (로깅 및 그래프 생성) ---
        graph_img = None
        if time.time() - self.last_log_time >= self.log_interval:
            # Log HSV Analysis if enabled
            # (HSV 분석 로그)
            if self.use_hsv_analysis and self.use_hsv_analysis.get():
                hsv_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
                h_val, s_val, v_val = hsv_frame[cy, cx]
                self.log_to_gui(f"HSV Value: ({h_val}, {s_val}, {v_val})")

            missing, dist = find_missing_paint_with_ratio(current_rgb_int, self.target_color)
            self.match_percentage = 100 - min(100, (color_distance(current_rgb_int, self.target_color) / math.sqrt(3 * 255 * 255)) * 100)
            
            self.log_to_gui(f"RGB: {current_rgb_int} | Match: {self.match_percentage:.2f}% | dist: {dist:.1f}")
            for paint, ratio in missing:
                log_msg = f"{paint}: Measurement Failed" if ratio is None else f"{paint}: {ratio:.1f}%"
                self.log_to_gui(log_msg)
            
            graph_img = draw_paint_need_graph(missing)
            self.last_log_time = time.time()

        # --- Prepare frame for GUI (GUI용 프레임 준비) ---
        cv2.circle(processed_frame, (cx, cy), 20, (0, 255, 0), 2)
        # cv2.rectangle(processed_frame, (w - self.patch_w, 0), (w, self.patch_h), (255, 0, 0), 2)
        if self.match_percentage is not None:
            text = f"Match: {self.match_percentage:.2f}%"
            cv2.putText(processed_frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # --- Send data to GUI thread (GUI 스레드로 데이터 전송) ---
        gui_data = {
            'video_frame': processed_frame,
            'graph_frame': graph_img,
            'current_color': current_rgb_int,
            'target_color': self.target_color
        }
        self.gui_queue.put(gui_data)

    # --- Filter Methods (필터 메서드) ---

    def sharpen_image(self, img):
        # Sharpening Filter (샤프닝 필터)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)

    def median_filter(self, img):
        # Median Filter (미디안 필터)
        return cv2.medianBlur(img, 5)

    def bilateral_filter(self, img):
        # Bilateral Filter (양방향 필터)
        return cv2.bilateralFilter(img, 9, 75, 75)

    def equalize_histogram_color(self, img):
        # Histogram Equalization on Y channel of YCrCb space
        # (YCrCb 색상 공간의 Y 채널에 대한 히스토그램 평준화)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
        img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
        return img_equalized

    def white_balance_fixed(self, img, kb, kg, kr, brightness_scale=1.0):
        # White Balance (화이트 밸런스)
        balanced = img.copy().astype(np.float32)
        balanced[:,:,0] = np.clip(balanced[:,:,0] * kb * brightness_scale, 0, 255)
        balanced[:,:,1] = np.clip(balanced[:,:,1] * kg * brightness_scale, 0, 255)
        balanced[:,:,2] = np.clip(balanced[:,:,2] * kr * brightness_scale, 0, 255)
        return balanced.astype(np.uint8)

    def log_to_gui(self, text):
        # This can be called from any thread
        self.gui_queue.put({'log': text})

    def on_close(self):
        self.get_logger().info("Shutting down...")
        self.cap.release()
        # The rclpy.shutdown() will be handled in the main block
        
    # -------------------------------------------------------------------
    # GUI Thread Function
    # -------------------------------------------------------------------
    def run_gui(self):
        root = tk.Tk()
        root.title("Color Matching (ROS2)")
        root.geometry("1090x620") # Increased width for checkboxes
        root.resizable(False, False)

        self.use_kalman = tk.BooleanVar(value=True)
        self.use_white_balance = tk.BooleanVar(value=True)
        self.use_gaussian = tk.BooleanVar(value=False)
        self.use_hist_eq = tk.BooleanVar(value=False)
        self.use_sharpen = tk.BooleanVar(value=False)
        self.use_median = tk.BooleanVar(value=False)
        self.use_bilateral = tk.BooleanVar(value=False)
        self.use_hsv_analysis = tk.BooleanVar(value=False)

        # --- Widget Creation ---
        video_frame = tk.Frame(root, width=600, height=350, bg="black")
        video_frame.place(x=10, y=10)
        video_label = tk.Label(video_frame, bg="black")
        video_label.place(relwidth=1, relheight=1)

        tk.Label(root, text="Target Color").place(x=620, y=10)
        target_canvas = tk.Canvas(root, width=280, height=170, bg="#c495f5", bd=2, relief="ridge")
        target_canvas.place(x=620, y=40)

        tk.Label(root, text="Current Color").place(x=620, y=220)
        current_canvas = tk.Canvas(root, width=280, height=170, bg="#777777", bd=2, relief="ridge")
        current_canvas.place(x=620, y=240)

        tk.Label(root, text="Terminal Log").place(x=10, y=375)
        terminal = tk.Text(root, bg="black", fg="white", width=85, height=10)
        terminal.place(x=10, y=405)

        tk.Label(root, text="Paint Need").place(x=620, y=420)
        graph_label = tk.Label(root, bg="#202020")
        graph_label.place(x=620, y=445, width=280, height=150)
        
        # --- Checkbox Controls ---
        controls_frame = tk.LabelFrame(root, text="Controls", padx=10, pady=10)
        controls_frame.place(x=910, y=10)

        kalman_check = tk.Checkbutton(controls_frame, text="Kalman Filter", var=self.use_kalman)
        kalman_check.pack(anchor='w')

        wb_check = tk.Checkbutton(controls_frame, text="White Balance", var=self.use_white_balance)
        wb_check.pack(anchor='w')

        gaussian_check = tk.Checkbutton(controls_frame, text="Gaussian Filter", var=self.use_gaussian)
        gaussian_check.pack(anchor='w')

        hist_eq_check = tk.Checkbutton(controls_frame, text="Hist. Equalization", var=self.use_hist_eq)
        hist_eq_check.pack(anchor='w')

        sharpen_check = tk.Checkbutton(controls_frame, text="Sharpen", var=self.use_sharpen)
        sharpen_check.pack(anchor='w')

        median_check = tk.Checkbutton(controls_frame, text="Median Filter", var=self.use_median)
        median_check.pack(anchor='w')

        bilateral_check = tk.Checkbutton(controls_frame, text="Bilateral Filter", var=self.use_bilateral)
        bilateral_check.pack(anchor='w')

        hsv_check = tk.Checkbutton(controls_frame, text="HSV Analysis", var=self.use_hsv_analysis)
        hsv_check.pack(anchor='w')

        # --- White Balance Gain Sliders ---
        wb_frame = tk.LabelFrame(root, text="White Balance Gain", padx=5, pady=5)
        wb_frame.place(x=910, y=250)

        self.kb_var = tk.DoubleVar(value=self.fixed_kb)
        self.kg_var = tk.DoubleVar(value=self.fixed_kg)
        self.kr_var = tk.DoubleVar(value=self.fixed_kr)
        self.wb_k_var = tk.DoubleVar(value=self.white_balance_fixed_k)

        # Blue
        tk.Label(wb_frame, text="Blue (kb)").pack(anchor="w")
        blue_frame = tk.Frame(wb_frame)
        blue_frame.pack(fill="x")
        kb_slider = tk.Scale(
            blue_frame, from_=0.5, to=3.0, resolution=0.01,
            orient="horizontal", variable=self.kb_var,
            command=lambda val: setattr(self, "fixed_kb", float(val))
        )
        kb_slider.pack(side="left", fill="x", expand=True)
        kb_entry = tk.Entry(blue_frame, textvariable=self.kb_var, width=5)
        kb_entry.pack(side="right", padx=5)

        # Green
        tk.Label(wb_frame, text="Green (kg)").pack(anchor="w")
        green_frame = tk.Frame(wb_frame)
        green_frame.pack(fill="x")
        kg_slider = tk.Scale(
            green_frame, from_=0.5, to=3.0, resolution=0.01,
            orient="horizontal", variable=self.kg_var,
            command=lambda val: setattr(self, "fixed_kg", float(val))
        )
        kg_slider.pack(side="left", fill="x", expand=True)
        kg_entry = tk.Entry(green_frame, textvariable=self.kg_var, width=5)
        kg_entry.pack(side="right", padx=5)

        # Red
        tk.Label(wb_frame, text="Red (kr)").pack(anchor="w")
        red_frame = tk.Frame(wb_frame)
        red_frame.pack(fill="x")
        kr_slider = tk.Scale(
            red_frame, from_=0.5, to=3.0, resolution=0.01,
            orient="horizontal", variable=self.kr_var,
            command=lambda val: setattr(self, "fixed_kr", float(val))
        )
        kr_slider.pack(side="left", fill="x", expand=True)
        kr_entry = tk.Entry(red_frame, textvariable=self.kr_var, width=5)
        kr_entry.pack(side="right", padx=5)

        # White Balance K
        tk.Label(wb_frame, text="White Balance (wb_k)").pack(anchor="w")
        wbk_frame = tk.Frame(wb_frame)
        wbk_frame.pack(fill="x")
        wbk_slider = tk.Scale(
            wbk_frame, from_=0.5, to=3.0, resolution=0.01,
            orient="horizontal", variable=self.wb_k_var,
            command=lambda val: setattr(self, "white_balance_fixed_k", float(val))
        )
        wbk_slider.pack(side="left", fill="x", expand=True)
        wbk_entry = tk.Entry(wbk_frame, textvariable=self.wb_k_var, width=5)
        wbk_entry.pack(side="right", padx=5)

        def update_gui():
            try:
                while not self.gui_queue.empty():
                    data = self.gui_queue.get_nowait()
                    
                    if 'log' in data:
                        now = time.strftime("%H:%M:%S")
                        terminal.insert("end", f"[{now}] {data['log']}\n")
                        terminal.see("end")

                    if 'video_frame' in data:
                        # Video frame
                        frame_rgb = cv2.cvtColor(data['video_frame'], cv2.COLOR_BGR2RGB)
                        im_pil = PILImage.fromarray(frame_rgb)
                        im_pil.thumbnail((560, 320), PILImage.LANCZOS)
                        padded_img = PILImage.new("RGB", (600, 350), (0, 0, 0))
                        padded_img.paste(im_pil, ((600 - im_pil.width) // 2, (350 - im_pil.height) // 2))
                        im_tk = ImageTk.PhotoImage(padded_img)
                        video_label.configure(image=im_tk)
                        video_label.image = im_tk

                        # Color boxes
                        r_i, g_i, b_i = data['current_color']
                        current_canvas.configure(bg=f"#{r_i:02x}{g_i:02x}{b_i:02x}")
                        tr, tg, tb = data['target_color']
                        target_canvas.configure(bg=f"#{tr:02x}{tg:02x}{tb:02x}")

                    if 'graph_frame' in data and data['graph_frame'] is not None:
                        graph_rgb = cv2.cvtColor(data['graph_frame'], cv2.COLOR_BGR2RGB)
                        img_pil = PILImage.fromarray(graph_rgb)
                        img_tk = ImageTk.PhotoImage(img_pil.resize((250, 150)))
                        graph_label.configure(image=img_tk)
                        graph_label.image = img_tk

            except queue.Empty:
                pass
            finally:
                root.after(30, update_gui)

        def on_closing_wrapper():
            self.on_close()
            root.destroy()
            # Request ROS shutdown from the main thread
            rclpy.shutdown()

        root.protocol("WM_DELETE_WINDOW", on_closing_wrapper)
        update_gui()
        root.mainloop()

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    try:
        color_matcher_node = ColorMatcherNode()
        rclpy.spin(color_matcher_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Node is destroyed automatically after spin exits
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

