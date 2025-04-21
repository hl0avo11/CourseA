# I. 빈 VM에 직접 환경 설정

## 1. VM 초기 생성

### 체크리스트
- 머신 구성
    - 이름: **ai-vm-{본인녹스ID, 콤마없이}**
    - GPU 유형: NVIDIA V100
    - GPU 수: 1
    - 머신 유형: 사전 설정 > n1-standard-8
- 운영체제 및 스토리지: 변경 클릭    
    - 운영체제: Ubuntu    
    - 버전: Ubuntu 22.04 LTS x86/64 jammy (minimal 안됨)
    - 부팅 디스크 유형:  SSD 영구 디스크
    - 크기: 100
- 네트워킹
    - HTTP 트래픽 허용
    - HTTPS 트래픽 허용
    - 네트워크 태그: jupyter, tensorboard \
      ※ 사전 방화벽이 설정되어 있음
- 보안
    - 서비스 계정: 서비스 계정 없음
 
## 2. VM SSH 설치 실행

### 초기 설정
- sudo apt-get update
- sudo apt-get install python3-pip -y

### GPU driver 설치
- sudo apt-get install ubuntu-drivers-common -y
    - ubuntu-drivers devices > driver 리스트 확인 후 recommend 버전 설치
    - sudo apt-get install nvidia-driver-550 -y
- reboot now > 이후 SSH 창 닫고 1분쯤 후 다시 연결
- nvidia-smi
    - 명령어 실행이 잘 되면 기본 드라이버는 설치가 된 것
 
### CUDA, cuDNN 설치
- mkdir tmp; cd tmp; wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
- sudo dpkg -i cuda-keyring_1.1-1_all.deb
- sudo apt-get update
- sudo apt-get install cuda -y
- sudo vi ~/.bashrc
    - 마지막에 아래 두줄 추가
    - export PATH=/usr/local/cuda/bin:/home/\$USER/.local/bin:\$PATH
    - export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
- source ~/.bashrc
- sudo apt-get install nvidia-cuda-toolkit -y
- nvcc -V
    - 명령어 실행이 잘 되면 cuDNN 설치가 된 것
 
### 기본 세팅: torch, jupyter 설치
- sudo apt-get install jupyter jupyter-core -y
- pip install torch==2.5.1+cu124 jupyterlab ipykernel
- jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token="" --NotebookApp.password="" --no-browser --NotebookApp.allow_origin="*"
- 이후 본인의 외부IP로 jupyterlab 연결 확인: http://{YOUR_IP}:8888/lab
※ 주의: 사용 종료 후에는 반드시 ctrl+c 두 번으로 종료해야 함

# II. 빈 VM에 GPU Docker 환경 설정

## 1. VM 설정

### 체크리스트
- I-1번 단락의 체크리스트를 그대로 하되, 머신 구성의 이름 설정만 아래와 같이 설정
    - 머신 구성-이름: **ai-docker-vm-{본인녹스ID, 콤마없이}**

## 2. VM SSH 설치 실행
- 초기 설정, GPU driver 설치, CUDA & cuDNN 설치까지 완료

### Docker 설치
- sudo apt update
- sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
- curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
- sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
- sudo apt update
- sudo apt install docker-ce docker-ce-cli containerd.io -y
- sudo apt-mark hold docker-ce docker-ce-cli

### Docker 설정
- sudo mkdir /etc/systemd/system/docker.service.d
- sudo vi /etc/systemd/system/docker.service.d/proxy.conf
- 내용은 아래와 같음
```bash
[Service]
Environment="NO_PROXY=localhost,127.0.0.1"
```

- sudo usermod -aG docker $USER
- newgrp docker
    - `docker version` 입력 후 잘 나오는지 확인
- sudo apt-get install docker-compose -y

### NVIDIA GPU와 Docker 연결
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
- sudo apt update
- sudo apt-get install nvidia-container-toolkit -y
- sudo nvidia-ctk runtime configure --runtime=docker
- sudo systemctl daemon-reload
- sudo systemctl restart docker

### pytorch용 docker image 다운로드 및 container 실행
- sudo docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
- mkdir ~/workspace; cd ~/workspace
- docker run -itd --name pytorch -v $HOME/workspace:/workspace -p 8888:8888 -p 8089:8089 -p 8006-8010:8006-8010/tcp --gpus all pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
- docker ps -a로 확인

### docker container 세팅
- docker exec -it pytorch /bin/bash
    - pip install jupyterlab ipykernel
    - mkdir ~/.jupyter; touch ~/.jupyter/jupyter_lab_config.py
    - echo "c.NotebookApp.terminado_settings = { 'shell_command': ['bash'] }" >> ~/.jupyter/jupyter_lab_config.py
    - jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token="" --NotebookApp.password="" --no-browser --NotebookApp.allow_origin="*"
    - 이후 본인의 외부IP로 jupyterlab 연결 확인: http://{YOUR_IP}:8888/lab