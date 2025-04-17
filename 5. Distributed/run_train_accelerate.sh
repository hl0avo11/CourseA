#!/bin/bash

# 환경 변수 설정
export OMP_NUM_THREADS=4

# accelerate 설정 초기화 (처음 실행 시에만 필요)
# accelerate config --config_file accelerate_config.yaml

# FSDP로 학습 실행
accelerate launch --config_file accelerate_config.yaml train.py 