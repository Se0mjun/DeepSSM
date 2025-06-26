# 🔁 Recurrent Neural Network (RNN) 정리

## 📌 정의
- 시퀀스(Sequence) 데이터를 처리하는 **순환 구조의 신경망**
- 이전 시점의 hidden state를 사용하여 **시간적으로 연속된 데이터의 패턴을 학습**

## 🧠 기본 수식

$$
\begin{aligned}
h_t &= \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h) \\
o_t &= W_{ho} h_t + b_o
\end{aligned}
$$

| 기호 | 의미 |
|------|------|
| $x_t$ | 입력 (at time t) |
| $h_t$ | hidden state |
| $o_t$ | 출력 |
| $W_{xh}$ | 입력 → hidden 가중치 |
| $W_{hh}$ | hidden → hidden 가중치 |
| $W_{ho}$ | hidden → output 가중치 |
| $b_h$, $b_o$ | 편향 (bias) |
| $	anh$ | 활성화 함수 (비선형성 부여) |

## 🧮 훈련 가능한 파라미터

| 파라미터 | 크기 | 설명 |
|----------|------|------|
| $W_{xh}$ | $H \times D$ | 입력 → hidden 변환 |
| $W_{hh}$ | $H \times H$ | 순환 연결 (이전 hidden → 현재 hidden) |
| $W_{ho}$ | $O \times H$ | hidden → 출력 (optional) |
| $b_h$ | $H$ | hidden bias |
| $b_o$ | $O$ | output bias (optional) |

- $D$: 입력 차원  
- $H$: hidden state 차원  
- $O$: 출력 차원  

✅ 이 모든 파라미터는 **trainable = True**로 설정되어 학습됨

## ❌ 훈련 불가능한 파라미터 (Non-trainable)

| 항목 | 설명 |
|------|------|
| 하이퍼파라미터 | hidden size, learning rate, layer 수 등 |
| 초기 상태 $h_0$ | 보통 0으로 고정되며 학습되지 않음 |
| 입력 데이터 $x_t$ | 학습의 대상이 아니라 조건 |
| dropout 비율 등 | 모델 외부 설정값 (학습되지 않음) |

## 🔍 활성화 함수

- 기본 RNN은 **$\tanh$** 함수를 사용
- 이유:  
  - 출력 범위가 $[-1, 1]$ → **zero-centered**
  - 학습 안정성 ↑
- ReLU는 잘 사용되지 않음 (gradient vanish 문제 적지만 memory 유지에 부적합)

## ✅ 특징 요약

| 항목 | 설명 |
|------|------|
| 장점 | 시퀀스 기반 입력 처리, 시간적 맥락 반영 |
| 단점 | 긴 시퀀스에서 gradient vanishing/exploding 문제 |
| 응용 | 자연어 처리, 음성 인식, 시계열 예측 등 |
