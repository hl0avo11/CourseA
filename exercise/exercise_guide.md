# 4주차 개인과제 가이드

## 사전 설정
서빙 시간에 강의했던 것처럼, "9. Serving" 폴더에서 터미널을 열고 Qwen2.5-1.5B vllm을 실행해주세요.

실행 코드)
> `vllm serve models/Qwen2.5-1.5B-Instruct/ --port 8006 --dtype half --gpu-memory-utilization 0.6 --enforce-eager --no-enable-prefix-caching | grep metrics`

※ 실행 명령어 뒤의 `| grep metric`은 GET, POST에 대한 정보들은 빼고 아래 형태의 메트릭에 집중하기 위함 \
`INFO 04-22 01:33:20 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.`

## 문제.
두 개의 파일이 주어집니다.
- fairy_tale.txt: 여러 개의 텍스트가 들어있는 파일
- locust_test.py: locust의 실행 로직이 들어있는 파일

여러분들은 이 폴더에서 locust_test.py를 이용해 locust 부하 테스트를 진행합니다. 

실행 코드)
> `locust -f locust_test.py --host http://localhost:8006`

여러분이 locust의 설정을 아래와 같이 실행했을 때, 종료시점의 RPS가 약 **9**가 나오는 것을 발견했습니다. (host는 그대로 둡니다)

locust 수행 조건)
- Number of Users: 1000
- Ramp up: 1000
- Advanced options
    - Run time: 60
 
여러분의 목표는 ***vllm 서빙 실행 시 넣을 수 있는 옵션을 변경***해 종료시점의 RPS가 **10**이 넘도록 하는 것입니다. 

다시, "9. Serving" 폴더에서 vllm serve의 옵션들을 변경해서 종료시점의 RPS가 10이 넘는 조건을 찾아 제출해주세요.

종료시점의 RPS 10을 달성하지 못했어도, 실험했던 결과 중 가장 RPS가 높았던 실험결과를 제출해 주세요.

### 제약 조건
- 사용 모델: Qwen2.5-1.5B-Instruct (양자화 모델 사용은 안됩니다)
- 모델 로드시 자료형: half 고정
- --no-enable-prefix-caching 인수 고정
- locust_test.py 파일과 fairy_tale.txt 파일은 수정 불가
- locust 실행 시 옵션은 위의 locust 수행 조건과 일치해야 합니다.

## 제출 양식
아래의 내용을 txt파일에 작성해주세요. (이창희_ckd_lee.txt를 참고해주세요)
- 달성한 RPS 수치
- 높은 RPS가 나오도록 vllm serve를 실행한 명령어
- 왜 RPS가 높아졌는지에 대한 분석 1~2줄
- 매 RPS 측정 시 locust는 터미널에서 종료 후 재시작해주세요. (seed 초기화 목적)

파일명은 이름_KnoxID.txt로 해주시고, KnoxID의 마침표(.)는 밑줄(_)로 바꿔주세요. \
(예: 이창희_ckd_lee.txt)

작성한 txt 파일을 아래 구글 드라이브 폴더에 업로드해주세요.
https://drive.google.com/drive/folders/1zpSkf29gagzMw8THawzAylZmmBWw7_SC?usp=sharing

## 참고
- [vLLM serve arguments](https://docs.vllm.ai/en/latest/serving/engine_args.html)
    - --gpu-memory-utilization
    - --max-num-seqs
    - --max-model-len