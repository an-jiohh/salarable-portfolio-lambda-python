FROM python:3.12.5-slim

# 작업 디렉토리 설정
WORKDIR /app

# OpenCV와 dlib을 설치하기 위한 추가 패키지 설치
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    build-essential cmake \
    libx11-dev libatlas-base-dev \
    libgtk-3-dev && \
    apt-get clean

# 패키지 설치
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt



COPY . .
CMD ["python","portfolio_preprocessing.py"]