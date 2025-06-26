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

- 각 이산 시간 단계 k 에서 표준 RNN은 이전 단계의 hidden state $h_{k-1}$ 와 함께 벡터 $x_k$ 를 처리하여 출력 벡터 $o_k$ 를 생성하고 hidden state를 $h_k$로 업데이트
- hidden state는 네트워크의 메모리 역할을 하며 과거 입력 에 대한 정보를 유지.


| 기호 | 의미 |
|------|------|
| $x_t$ | 입력 (at time t) |
| $h_t$ | hidden state |
| $o_t$ | 출력 |
| $W_{xh}$ | 입력 → hidden 가중치 | 모델 입력을 hidden state로 처리하는 가중치 행렬. 
| $W_{hh}$ | hidden → hidden 가중치 | hidden state 간의 반복 연결
| $W_{oh}$ | hidden → output 가중치 | hidens state에서 파생된 출력을 생성하는데 사용되는 가중치
| $b_h$, $b_o$ | 편향 (bias) |
| $tanh$ | 활성화 함수 (비선형성 부여) |

## RNN의 한계점 
1.  long range dynamics 를 추출하는데 제한적 (가중치를 반복적으로 곱하면서 dilution or loss
2.  순차적인 데이터를 점진적으로 처리하기에 각 시간 단계가 이전 시간 단계에 의존 ( 계산 효율성 저하 )

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

## 🧮 연산 복잡도 및 병렬성

- RNN의 연산 복잡도: $O(L D^2)$  
  - $L$: 시퀀스 길이, $D$: hidden 차원 수
- Transformer ($O(L^2 D)$)보다 효율적일 수 있으나
- **순차적인 구조**로 인해 병렬화가 어려워 **학습 속도 및 효율성은 낮음**

## ✅ 특징 요약

| 항목 | 설명 |
|------|------|
| 장점 | 시퀀스 기반 입력 처리, 시간적 맥락 반영 |
| 단점 | 긴 시퀀스에서 gradient vanishing/exploding 문제 |
| 응용 | 자연어 처리, 음성 인식, 시계열 예측 등 |

# 🧠 Transformer 정리 (from A Survey of Mamba)

## 📌 정의
- Transformer는 **Self-Attention 메커니즘**을 도입하여 RNN의 순차적 처리 한계를 극복한 딥러닝 모델 아키텍처
- 입력 시퀀스를 **병렬 처리**할 수 있어 학습 속도와 효율성이 뛰어남
- 자연어 처리, 비전, 멀티모달 학습 등 다양한 분야에서 사용됨

## 🔍 핵심 구조

### 1. 입력 → 쿼리/키/값 변환

$$
Q = xW_Q, \quad K = xW_K, \quad V = xW_V
$$

- $W_Q$, $W_K$, $W_V$: 학습 가능한 가중치 행렬

### 2. Self-Attention 계산

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_K}}\right) V
$$

- $d_K$: key 벡터의 차원
- 입력 시퀀스 내 위치 간의 관계를 반영하여 중요도 가중치 부여

### 3. Multi-Head Attention

$$
\text{MultiHead}(Q,K,V) = (S_1 \oplus S_2 \oplus \cdots \oplus S_m)W_O
$$

- 여러 개의 attention head 병렬 수행
- 다양한 관계 및 특징을 학습 가능

## ⚙️ 연산 특징

| 항목 | 설명 |
|------|------|
| 연산 복잡도 | $O(L^2 D)$ (시퀀스 길이 $L$, 차원 $D$) |
| 병렬 처리 | 가능 (GPU에 적합) |
| 추론 속도 | 느림 (Auto-regressive generation 시 반복 계산 필요) |

## ✅ 장점

- 긴 시퀀스 내 위치 간의 **전역적 관계 파악 가능**
- 병렬 처리 덕분에 **학습 속도 빠름**
- 다양한 태스크에 높은 성능

## ❌ 단점

- Attention 연산의 **연산량이 $O(L^2)$로 큼** → 긴 입력 시 비효율적
- 추론(inference) 시 auto-regressive 특성으로 **token-by-token 생성 → 느림**
- 긴 문서 처리에는 비효율적일 수 있음

## 📌 응용 분야

- 자연어 처리 (BERT, GPT, T5 등)
- 비전 (ViT, DeiT 등)
- 시계열 예측, 멀티모달 학습, 음성 인식 등

---

**참고:** 본 내용은 “A Survey of Mamba (2024)”에서 Transformer 관련 서술을 요약한 것입니다.
