# ROS2 STT & Paint Command Service Server
# - Trigger 서비스(start_paint_command)를 통해 음성 명령 처리 시작
# - Whisper 기반 STT로 음성을 텍스트로 변환
# - LLM(GPT)을 이용해 색상 요청 및 조색 명령을 파싱
# - 파싱 결과를 기반으로
#   · 목표 색상(target_paint_color) 토픽 퍼블리시
#   · 조색 명령(paint_command) 토픽 퍼블리시
# - Wakeup Word 노드에 의해 트리거되는 음성 기반 페인팅 명령 처리 노드

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wav
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import re

# .env 파일에서 OPENAI_API_KEY 불러오기
load_dotenv(dotenv_path='/home/gjlee/ros2_ws/src/corobot2_project/corobot2_project/.env')
openai_api_key = os.getenv("OPENAI_API_KEY")


class PaintCommandParser:
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            openai_api_key=openai_api_key
        )

    def parse(self, user_input: str) -> str:
        prompt_content = f"""
        당신은 사용자의 문장에서 특정 물감 색상을 추출하고, 
        그 색을 만들기 위해 빨강(red_paint), 노랑(yellow_paint), 하양(white_paint), 검정(black_paint), 파랑(blue_paint) 
        다섯 가지 기본 물감을 사용하여 조색 방법을 알려주는 전문가입니다.

        <목표>
        1. 문장에서 원하는 최종 색상을 최대한 정확히 추출하세요.
        2. 만약 사용자가 "만들어줘", "만들래", "섞어줘" 등 **조색 명령**을 하면 → 조색 비율을 제시하세요.
        3. 만약 단순히 "가져와", "줘", "필요해" 등 **단순 요청**을 하면 → "조색x/ [기본색]" 형태로 출력하세요.

        <물감 리스트>
        - white_paint, black_paint, red_paint, blue_paint, yellow_paint

        <출력 형식>
        - 조색 명령일 때: 색상명/ 기본물감 비율
        예: "연두색 만들어줘" → 연두색/ yellow_paint 70% / blue_paint 30%
        - 단순 요청일 때: 조색x/ 기본색
        예: "노란색 가져와" → 조색x/ yellow_paint
        - 사용자가 말한 색상이 기본 5색 중 하나면 그대로 출력하고 비율은 100%로 설정하세요.
        - 비율은 대략적인 값으로도 괜찮습니다.

        <특수 규칙>
        - 사용자가 "파란색 물감"이라고 하면 blue_paint, "하얀색"이면 white_paint처럼 기본 색과 매칭하세요.
        - 기본 5색으로 만들 수 없는 색이라면 "조색 불가능"이라고 표시하세요.
        - 여러 색을 요청하면 각각의 결과를 줄바꿈으로 구분해 순서대로 출력하세요.

        <예시>
        - 입력: "연두색 물감 만들어줘"  
        출력: 연두색/ yellow_paint 70% / blue_paint 30%

        - 입력: "보라색이랑 하얀색 필요해"  
        출력: 보라색/ red_paint 75% / blue_paint 25%
                조색x/ white_paint

        - 입력: "노란색 가져와"  
        출력: 조색x/ yellow_paint

        <사용자 입력>
        "{user_input}"
        """
        response = self.llm.predict(prompt_content)
        return response.strip()


    def parse_to_dict(self, user_input: str) -> dict:
        """
        user_input -> {색상명: {물감: 비율, ...}, ...} 형태로 반환
        """
        raw_text = self.parse(user_input)
        result = {}

        lines = raw_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Case 1. "보라색: red_paint 50% / blue_paint 50%"
            if ':' in line:
                color_name, paints_str = line.split(':', 1)
                color_name = color_name.strip()
                paints_dict = {}
                
                # 슬래시 구분
                paints = paints_str.split('/')
                for p in paints:
                    p = p.strip()
                    match = re.match(r'(\w+)\s+(\d+)%', p)
                    if match:
                        paint, ratio = match.groups()
                        paints_dict[paint] = int(ratio)
                result[color_name] = paints_dict

            else:
                # Case 2. "조색x/ yellow_paint"
                if '/' in line:
                    parts = [p.strip() for p in line.split('/')]
                    color_name = parts[0]   # 예: "조색x"
                    paints_dict = {}

                    for p in parts[1:]:
                        match = re.match(r'(\w+)', p)
                        if match:
                            paint = match.group(1)
                            paints_dict[paint] = 100  # 단일 색상은 100%로 처리
                    result[color_name] = paints_dict

                # Case 3. "조색 불가능" 처럼 아예 단독 문구
                else:
                    result[line] = {}

        return result

class STT:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.duration = 3
        self.samplerate = 16000

    def speech2text(self):
        print("🎤 3초간 음성 입력을 시작합니다...")
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("🛜 Whisper 모델에 전송 중...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)

            with open(temp_wav.name, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )

        print("🗣 인식된 음성:", transcript.text)
        return transcript.text


class PaintCommandNode(Node):
    def __init__(self, openai_api_key):
        super().__init__("paint_command_server")
        self.publisher_command = self.create_publisher(String, "/dsr01/paint_command", 10)
        self.publisher_target = self.create_publisher(String, "/dsr01/target_paint_color", 10)
        self.stt = STT(openai_api_key)
        self.parser = PaintCommandParser(openai_api_key)

        # Trigger 서비스 서버 생성
        self.srv = self.create_service(Trigger, "start_paint_command", self.handle_request)

        self.get_logger().info("🎯 PaintCommandNode 서비스 서버가 시작되었습니다. (대기 중)")

    def handle_request(self, request, response):
        """
        서비스 요청이 들어오면:
        1. 3초 음성 입력
        2. 파싱
        3. 토픽 퍼블리시
        4. 응답 반환
        이후 다시 대기 상태로 전환
        """
        self.get_logger().info("🎤 서비스 요청 수신 → 음성 입력 시작")
        
        # 1. 음성 → 텍스트
        text = self.stt.speech2text()
        if not text:
            response.success = False
            response.message = "음성 입력 실패"
            return response

        # 2. LLM으로 파싱 → 딕셔너리 형태로
        parsed_dict = self.parser.parse_to_dict(text)
        self.get_logger().info(f"🔍 파싱 결과: {parsed_dict}")

        # 3-1. target color 퍼블리시
        if parsed_dict:
            first_color_name = list(parsed_dict.keys())[0]
            target_msg = String()
            target_msg.data = first_color_name
            self.publisher_target.publish(target_msg)
            self.get_logger().info(f"🎨 /target_paint_color 퍼블리시: {first_color_name}")
        
        # 3-2. paint command 퍼블리시 (조색 정보)
        command_msg = String()

        # parsed_dict의 첫 번째 key의 value만 가져오기
        first_key = list(parsed_dict.keys())[0]
        paint_command_value = parsed_dict[first_key]  # {"red_paint": 50, "blue_paint": 50}

        # 문자열 형태로 변환
        parts = [f"{k} {v}" for k, v in paint_command_value.items()]
        command_str = " / ".join(parts)

        command_msg.data = command_str
        self.publisher_command.publish(command_msg)
        self.get_logger().info(f"🖌️ /paint_command 퍼블리시: {command_msg.data}")

        # 4. 서비스 응답
        response.success = True
        response.message = f"명령 파싱 및 퍼블리시 완료: {command_msg.data}"
        self.get_logger().info("✅ 처리 완료 → 서비스 대기 상태로 복귀")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PaintCommandNode(openai_api_key)
    try:
        rclpy.spin(node)  # 서비스 대기 상태 유지
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
