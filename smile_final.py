
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import time
import sys
import webbrowser
import mediapipe as mp
import itertools

# 기울기 함수
@st.cache_data
def slopee(x1,y1,x2,y2):
    x = (y2 - y1) / (x2 - x1)
    return x

# 기본 화면 표시
main_placeholder = st.empty()

with main_placeholder.container():
    st.markdown("# :red[웃지마]")
    st.markdown('<p style="font-size: 40px;">웃음참기 게임에 오신 것을 환영합니다.</p>', unsafe_allow_html=True)
    st.markdown("# How to Play")
    st.write('* 참여하려는 인원 수를 설정해주세요')
    st.write('* 게임 난이도를 설정해주세요')
    st.write('* 하 -> 상 으로 갈 수록 작은 미소에도 반응합니다')
    st.write('* 원하는 챌린지 영상이 있다면 동영상 설정 - manual 선택 후 URL 주소를 입력해주세요')
    st.markdown('<p style="font-size: 20px;">모든 설정이 끝났다면 즐겨주세요!</p>', unsafe_allow_html=True)


# 기본 화면_sidebar 표시
main_side_title = st.sidebar.title("Game Start!")
select_options = ["★About★","★게임시작★"]
main_select_button = st.sidebar.selectbox("MODE", select_options)
main_side_txt_2 = st.sidebar.write(':arrow_up_small: 시작하기')
num = st.sidebar.slider('인원 수', min_value = 1, max_value = 5, step = 1) # 참여 인원 수 설정
# diff = st.sidebar.slider('난이도', min_value = 1, max_value = 10, step = 1) # 난이도 설정

diff = st.sidebar.select_slider(
    '난이도를 선택해주세요',
    options=['하', '중', '상'])
st.sidebar.write('선택한 난이도는', diff, '입니다.')

select_video = st.sidebar.radio('동영상 설정', ['auto', 'manual']) # 원하는 동영상 재생 여부
if select_video == 'manual': # 원하는 동영상 주소 넣을 곳
    url_input = st.sidebar.text_input('url')
    
# 게임시작 화면
def game():
    #  게임시작 화면 설정
    game_title = st.title('웃지마!')
    game_button = st.button('시작')
    
    if game_button and select_video == 'auto': # 시작 버튼 누르면 영상 출력/모델실행
        video = st.video('https://www.youtube.com/watch?v=uShSDKIpyTE&t=2s', start_time = 0)
        model()
        game_title.empty()
        video.empty()
        st.title('그게나야~~~~~')
        st.image(image1)
        st.balloons()
        audio_html = """
        <audio src='http://localhost:8501/media/9eb21960efdb64218f2f6bc5a83afd7337dfa6aaa17b17d4ffb98c3d.wav' autoplay>
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        t= st.audio("FINAL.mp3")
        t.empty()
            
        
    elif game_button and select_video == 'manual': # 시작 버튼 누르면 입력한 url영상 출력/모델실행
        video = st.video(url_input, start_time = 0)
        model()
        game_title.empty()
        video.empty()
        st.title('그게나야~~~~~')
        st.image(image1)
        st.balloons()
        audio_html = """
        <audio src='http://localhost:8501/media/9eb21960efdb64218f2f6bc5a83afd7337dfa6aaa17b17d4ffb98c3d.wav' autoplay>
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        t= st.audio("FINAL.mp3")
        t.empty()
            
        
def model():
    global image1
    
    # 실패한 사람 이미지 캡쳐 경로 설정
    save_dir = 'output_capture'
    os.makedirs(save_dir, exist_ok = True)
    file_path = os.path.join(save_dir, f'cap.jpg')
    
    # Mediapipe 실행
    mp_face_mesh = mp.solutions.face_mesh
    
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces = num,
        refine_landmarks = True,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.7) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            image.flags.writeable = False # 이미지에 마스크팩 표시 안함
            
            results = face_mesh.process(image) # facemesh 처리한 결과 받아두기

            # 기울기 값 계산
            image = cv2.flip(image, 1) # 좌우 반전
            image = cv2.resize(image, dsize = (0, 0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA) # 이미지 사이즈 축소
            height, width, _ = image.shape # 웹캠 사이즈 확인
            try:
                for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
                    # 왼쪽 입꼬리 - 입술 중앙 기울기 계산
                    left = slopee(face_landmarks.landmark[61].x,
                         face_landmarks.landmark[61].y,
                         face_landmarks.landmark[14].x,
                        face_landmarks.landmark[14].y)
                    # 오른쪽 입꼬리 - 입술 중앙 기울기 계산
                    right = slopee(face_landmarks.landmark[14].x,
                         face_landmarks.landmark[291].y,
                         face_landmarks.landmark[291].x,
                        face_landmarks.landmark[14].y)

                    difficulty = {'하' : 0.4, '중' : 0.3, '상' : 0.2}

                    # 얼굴 전체 좌표 확인
                    top = int(face_landmarks.landmark[10].y * height)
                    bot = int(face_landmarks.landmark[152].y * height)
                        # 좌우 반전된 x값을 원래 값으로 만들어주기
                    rig = abs(int(face_landmarks.landmark[234].x * width) - width)
                    lef = abs(int(face_landmarks.landmark[454].x * width) - width)

                    # 얼굴 따라서 사각형 박스 표시
                        # 좌표 변수 값 뒤에 상수 값으로 사각형 크기 조절 가능(0일 경우 눈썹부터 턱까지)
                    p1 = (rig+10, top-10)
                    p2 = (lef-10, bot+10)
                    p3 = (rig-int((rig-left)/2), top) # text 표시할 좌표
                    cv2.rectangle(image, pt1 = p1, pt2 = p2, color = [255, 0, 0], thickness = 2, lineType = cv2.LINE_AA)
                    cv2.putText(image, text = f'person {face_no + 1}', org = p3, fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1,
                                color = [255, 255, 255])
                    if left >= difficulty[diff] -0.1 or right >= difficulty[diff] - 0.1: # 임계값 근처에서 박스 색깔 변환
                        cv2.rectangle(image, pt1 = p1, pt2 = p2, color = [0, 0, 255], thickness = 2, lineType = cv2.LINE_AA)
                    cv2.imshow('smile!', image)
                    cv2.moveWindow('smile!', -10, -10) # 이미지 창 위치 고정
                    cv2.setWindowProperty('smile!', cv2.WND_PROP_TOPMOST, 1) # 새 창을 제일 위로 올림

                    # 기울기 값 확인해서 웃은 사람 판별
                    if (left or right) > difficulty[diff] : # 0.5 기울기값(임계값) 난이도 별 설정 필요
                        # 실패한 사람 얼굴 캡쳐
                        cv2.imwrite(file_path, image)
                        # 캡쳐한 사진 자르기
                            # 좌표 변수 값 뒤에 상수 값으로 캡쳐 크기 조절 가능(0일 경우 눈썹부터 턱까지)
                        crop_img = image[top-10:bot+10, lef-10:rig+10]
                        # 자른 사진 확대
                        image = cv2.resize(crop_img, (480, 640))
                        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cv2.waitKey(1000) # 3000ms 이후 자동 종료
                        cap.release()
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            except:
                pass
#                 it_is.empty()
            
    cap.release()
    cv2.destroyAllWindows()
    
# 함수가 전부 로딩 된 다음에 이동이 가능해서 마지막으로 설정
# 선택 옵션에 따라서 기본화면 내용 지우고 함수 호출
if main_select_button == select_options[1]:
    main_placeholder.empty()
    game()
