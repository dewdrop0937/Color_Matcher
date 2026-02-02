# Color_Matcher(ROS 2)

## 📌 프로젝트 개요

본 프로젝트는 **음성 명령을 통해 특정 색상을 인식하고 조색에 필요한 물감을 인식하고, 로봇이 이를 집어 지정된 위치(Home)로 이동시키는 ROS 2 기반 로봇 시스템**입니다.

Wake-word 기반 음성 인터페이스, YOLO 기반 비전 인식, 깊이 카메라를 이용한 3D 위치 추정, 그리고 로봇 제어까지  
**실제 서비스 로봇 적용을 목표로 한 전체 시스템 구조를 팀 단위로 설계·구현했습니다.**

---

## 🧠 시스템 핵심 특징

* **Wake-word 기반 음성 인터페이스**
  → "Hello Rokey" 인식 시 자동으로 음성 명령 파이프라인 활성화

* **3-Layer 아키텍처 설계**

  * Vision Layer (인식)
  * Robot Control Layer (동작 계획 및 제어)
  * ROS Node Layer (통신 및 상태 관리)

* **YOLO Segmentation + Mask 기반 각도 추정**
  → minAreaRect를 활용한 물체 장축 기준 정렬 각도 계산

* **Depth 기반 3D 좌표 복원**
  → 픽셀 → 카메라 좌표 변환

* **GUI 기반 실시간 영상 디버깅 도구**
  → 필터링, White Balance, Kalman Filter 적용 가능

---

## 🏗️ 전체 시스템 구조

```
[Wake Word]
     ↓
[STT + Color Parsing]
     ↓ (Service)
[Vision Module (YOLO + Depth)]
     ↓
[Robot Control Node]
     ↓
[Pick & Place Execution]
```

---

## 📂 코드 구성

```
corobot2_project/
├── voice/
│   └── wakeup_word_node.py       # Wake-word 인식 및 서비스 호출
│   └── mic_controller.py          # 마이크 스트림 열기/닫기 기능
│   └── paint_command_server.py
│   └── stt_parser_topic_pub.py
│
├── vision/
│   └── vision_module.py          # YOLO + Depth 기반 타겟 인식 (각도/위치 추정)
│   └── color_matcher_node.py     # 카메라 영상 기반 색상 매칭 및 조색 비율 추정 노드 (GUI 포함)
│
├── robot_control/                # 로봇 이동, 정렬, 집기, 홈 복귀
│   └── paint_picker_node.py    
│   └── onrobot.py
│   └── robotmodule.py
│
└── README.md
```

---

## 🔍 주요 모듈 설명

### 1️⃣ Wakeup Word Node

* **openWakeWord + TFLite 모델** 사용
* 실시간 오디오 스트림에서 Wake-word 감지
* 중복 인식 방지를 위한 **Cooldown 로직** 적용
* 인식 시 `start_paint_command` 서비스 호출

---

### 2️⃣ Vision Module (3-Layer 중 Vision)

* **Ultralytics YOLO Segmentation 모델** 사용
* Mask 기반 물체 영역 추출
* `minAreaRect`를 통한 **물체 정렬 각도 계산**
* Depth 이미지에서 **Mask 내부 Median Depth 추출**
* Camera Intrinsic 기반 3D 좌표 복원

```python
(angle_deg, (cx, cy), box) = calc_angle_from_mask_rotated(mask)
```

---

### 3️⃣ Image Viewer GUI

* ROS Image Topic 자동 탐색
* 멀티 윈도우 지원
* 실시간 영상 필터링:

  * Gaussian / Median / Bilateral
  * Histogram Equalization
  * White Balance Gain
  * Kalman Filter 시각화

→ **실제 로봇 디버깅을 고려한 실용 도구**

---

## 🤖 AI API 활용 구조

본 프로젝트에서는 음성 명령의 유연한 처리를 위해 **AI 기반 API를 활용한 하이브리드 구조**를 적용했습니다.

- Wake-word 인식 이후 사용자의 음성 명령을 **STT(Speech-To-Text) API**를 통해 텍스트로 변환
- 변환된 텍스트에서 **색상 및 조색 관련 키워드 파싱**
- 파싱 결과를 ROS 2 서비스로 전달하여 Vision 및 Robot Control 모듈과 연동

이를 통해 단순 키워드 매칭이 아닌,  
**확장 가능한 음성 인터페이스 구조**를 설계했습니다.

---

## 🛠️ 사용 기술 스택

* **ROS 2 (rclpy)**
* **Python 3**
* **YOLOv8 (Ultralytics)**
* **OpenCV / NumPy**
* **Depth Camera (RGB-D)**
* **openWakeWord / TFLite**
* **AI API (Speech-To-Text, Color Parsing)**
* **Tkinter GUI**

---

## 👤 역할 및 기여도 (Role & Contribution)

본 프로젝트는 **팀 프로젝트**로 진행되었습니다.

- **본인**  
  - 카메라 기반 **조색(Color Mixing) 시스템 설계 및 구현**
  - 목표 색상 인식 및 물감 색상 매칭 로직 개발
  - 조색에 필요한 **물감 비율 추정 알고리즘 구현**
  - Vision 결과를 기반으로 한 **물감 각도 계산 로직 개발**
  - Vision 모듈과 ROS 2 노드·서비스 간 연동 구조 설계

- **팀원 1**  
  - Tkinter 기반 **GUI 인터페이스 개발**
  - 실시간 영상 시각화 및 디버깅 도구 구현
  - 영상 필터링 (Gaussian, Median, White Balance 등) 기능 개발

- **팀원 2**  
  - 로봇 매니퓰레이터 **이동 및 제어 로직 구현**
  - Pick & Place 동작 시퀀스 설계
  - Vision 결과를 활용한 로봇 동작 연계
  - Vision 모듈과 ROS 2 노드·서비스 간 연동 구조 설계

- **팀원 3**  
  - **STT(Speech-To-Text) 기반 음성 명령 처리 모듈 구현**
  - 음성 입력 파싱 및 ROS 2 통신 연동
  - Wake-word 이후 명령 흐름 처리

---

