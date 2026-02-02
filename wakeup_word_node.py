# Wakeup Word Detection Node
# - 마이크 스트림에서 Wakeup Keyword(hello_rokey)를 실시간으로 인식
# - Wakeup 감지 시 start_paint_command 서비스 호출
# - 음성 명령 처리(STT + 색깔 파싱)를 시작하는 트리거 역할
# - 3-레이어 구조(입력/인식 / 명령 처리 / 로봇 제어) 중 입력·인식 단계

import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
from ament_index_python.packages import get_package_share_directory
from corobot2_project import MicController

import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger  # 서비스 예시: Trigger 사용

MODEL_NAME = "hello_rokey_8332_32.tflite"


class WakeupWord(Node):
    def __init__(self, buffer_size):
        super().__init__("wakeup_word_node")
        openwakeword.utils.download_models()
        self.model = None
        self.model_name = MODEL_NAME.split(".", maxsplit=1)[0]
        self.stream = None
        self.buffer_size = buffer_size

        # 서비스 클라이언트 생성
        self.cli = self.create_client(Trigger, "start_paint_command")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("서비스 대기 중: start_paint_command ...")

        # wakeword 중복 방지용
        self.last_trigger_time = 0.0
        self.cooldown = 2.0  # 2초간 재인식 방지

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        self.get_logger().info(f"confidence: {confidence:.3f}")

        # 현재 시간 확인
        now = time.time()

        if confidence > 0.3 and (now - self.last_trigger_time) > self.cooldown:
            self.get_logger().info("Wakeword detected! 서비스 호출 시도")
            self.last_trigger_time = now
            self.call_service()
            return True
        return False

    def call_service(self):
        req = Trigger.Request()
        future = self.cli.call_async(req)
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"서비스 응답: {res.message}")
            else:
                self.get_logger().warn(f"서비스 실패: {res.message}")
        except Exception as e:
            self.get_logger().error(f"서비스 호출 에러: {e}")

    def set_stream(self, stream):
        self.model = Model(
            wakeword_models=[
                "/home/gjlee/ros2_ws/src/corobot2_project/corobot2_project/hello_rokey_8332_32.tflite"
            ]
        )
        self.stream = stream


def main(args=None):
    rclpy.init(args=args)

    Mic = MicController.MicController()
    Mic.open_stream()

    wakeup_node = WakeupWord(Mic.config.buffer_size)
    wakeup_node.set_stream(Mic.stream)

    try:
        while rclpy.ok():
            wakeup_node.is_wakeup()
    except KeyboardInterrupt:
        pass
    finally:
        wakeup_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()