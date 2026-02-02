# ROS2 기반 음성 명령 처리 노드
# - 마이크 입력을 받아 Whisper(STT)를 통해 음성을 텍스트로 변환
# - 변환된 텍스트를 LLM(ChatGPT)을 이용해 물감 색상과 목적지로 파싱
# - 파싱 결과를 "paint_command" 토픽으로 퍼블리시
# - 음성 → 자연어 → 구조화된 명령 → 로봇 제어 노드로 전달하는 역할 수행

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wav
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# .env 파일에서 OPENAI_API_KEY 불러오기
load_dotenv(dotenv_path='/home/hihigiig101/doosan_arm_ai/src/DoosanBootcamp3rd/dsr_rokey/rokey/resource/.env')
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
            당신은 사용자의 문장에서 특정 물감과 목적지를 추출해야 합니다.

            <목표>
            - 문장에서 다음 리스트에 포함된 물감 색상을 최대한 정확히 추출하세요.
            - 문장에 등장하는 물감의 목적지(어디로 옮기라고 했는지)도 함께 추출하세요.

            <물감 리스트>
            - white_paint, black_paint, red_paint, blue_paint, yellow_paint, 1번 위치, green_paint, 2번 위치, 3번 위치

            <출력 형식>
            - 다음 형식을 반드시 따르세요: [물감1 물감2 ... / 1번 위치 2번 위치 ...]
            - 물감과 위치는 각각 공백으로 구분
            - 물감이 없으면 앞쪽은 공백 없이 비우고, 목적지가 없으면 '/' 뒤는 공백 없이 비웁니다.
            - 물감과 목적지의 순서는 등장 순서를 따릅니다.

            <특수 규칙>
            - 명확한 물감 명칭이 없지만 문맥상 유추 가능한 경우(예: "파란색 물감" → blue_paint)는 리스트 내 항목으로 최대한 추론해 반환하세요.
            - 다수의 물감과 목적지가 동시에 등장할 경우 각각에 대해 정확히 매칭하여 순서대로 출력하세요.

            <예시>
            - 입력: "red_paint를 1번 위치에 가져다 놔"  
            출력: red_paint / 1번 위치

            - 입력: "파란색 물감과 green_paint를 2번 위치에 넣어줘"  
            출력: blue_paint green_paint / 2번 위치

            - 입력: "노란색 물감 줘"  
            출력: yellow_paint /

            - 입력: "파란 물감은 1번 위치 두고 초록 물감은 3번 위치에 둬"  
            출력: blue_paint green_paint / 1번 위치 3번 위치

            <사용자 입력>
            "{user_input}"                
            """
        response = self.llm.predict(prompt_content)
        return response.strip()


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
        super().__init__("paint_command_node")
        self.publisher_ = self.create_publisher(String, "paint_command", 10)
        self.stt = STT(openai_api_key)
        self.parser = PaintCommandParser(openai_api_key)

        # 5초마다 음성 명령을 받음
        self.timer = self.create_timer(5.0, self.timer_callback)
        self.get_logger().info("🎯 PaintCommandNode가 시작되었습니다.")

    def timer_callback(self):
        # 1. 음성 → 텍스트
        text = self.stt.speech2text()
        if not text:
            return

        # 2. LLM으로 파싱
        parsed = self.parser.parse(text)
        self.get_logger().info(f"🔍 파싱 결과: {parsed}")

        # 3. ROS 퍼블리시
        msg = String()
        msg.data = parsed
        self.publisher_.publish(msg)
        self.get_logger().info(f"📢 퍼블리시 완료: {parsed}")


def main(args=None):
    rclpy.init(args=args)

    node = PaintCommandNode(openai_api_key)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
