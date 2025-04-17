#!/bin/bash

# 환경 변수 설정
export OMP_NUM_THREADS=4

# DDP로 학습 실행
torchrun --nproc_per_node=2 train.py 