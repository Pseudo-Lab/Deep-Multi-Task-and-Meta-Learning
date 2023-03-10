# Lecture 15. Lifelong Learning

>Organization: 가짜연구소 (Pseudo Lab)  
>Editor: [김누리 (Nuri Kim)](https://github.com/knurii)  
>강의 자료: [CS330 2020 Fall](http://cs330.stanford.edu/fall2020/slides/cs330_lifelonglearning_karol.pdf)  
>강의 영상: [Youtube](https://www.youtube.com/watch?v=bHmOYLAtE6I)

## 1. The Problem Statement

1. brief review of problem statements
    - Multi-task Learning
        - Learn to solve a set of tasks
    - Meta-Learning
        - Given i.i.d. task distribution, learn a new task efficiently
        - 즉 새로운 과제를 빠르게 학습
    - Real World Settings
        - 실제 환경에서는 사전학습 data / task 배치가 없음
        - 대량의 데이터가 확보되지 않으므로, 순차적으로 새로운 task 학습
        - e.x.
            - a student learning concepts in school : 대수1 학습 → 대수2 학습
            - a deployed image classification system learning from a stream of images from users
            - a robot acquiring an increasingly large set of skills in different environments
            - a virtual assistant learning to help different users with different tasks at different points in time
            - a doctor’s assistant aiding in medical decision-making
2. Some Terminology
    - Sequential learning settings
        - online learning, lifelong learning, continual learning, incremental learning, streaming data
        - sequence data 혹은 sequential decision-making과는 구분된다.
3. Problem statement
    - 토의
        
        ![image](https://user-images.githubusercontent.com/101931446/221550308-2196fb52-ce50-411c-bee2-a82f2abc941b.png)
        
        - MNIST
        - 실내 로봇 네비게이션 문제
        - 학교에서의 학습 개념
        - 방에서 방으로 이동하며 보이는 물체를 인식하는 로봇
        
        ⇒ **점점 새로운 데이터가 추가되면서, 새로운 task에 대해 더 잘 학습할 수 있는 상황**
        
    - problem variations
        - task/data order : i.i.d. vs. predictable vs. curriculum vs. adversarial
        - discrete task boundaries vs. continuous shifts
        - known task boundaries / shifts vs. unknown
    - some considerations
        - model performance
        - data efficiency
        - computational resources
        - memory
        - others : privacy, interpretability, fairness, test time compute & memory
    - General [supervised] online learning problem
        - 시간이 지남에 따라 데이터 포인트 주어짐
        - observe $x_t$, predict $\hat{y}_t$ and observe label $y_t$
        - 근데 i.i.d setting이라면 시간에 의존하지 않음
            - x 분포 어디서 샘플링하던 상관 없다.
            - 분포가 시간에 따라 달라지게 되면 샘플 추출 시점에 따라 다르게 나타남
        - streaming setting
            - 현재 샘플 저장할 수 없다는 가정하에
            - 여러 이유 있음(메모리나 리소스 부족, 개인정보보호, 신경망 연구)
        - recall
            - 강화학습에서는 거의 모든 off policy 알고리즘에 replay buffers 사용 → 데이터 저장 쉬움
4. **What do you want** from your lifelong learning algorithm?
    - minimal regret
        
        $$
        Regret_T := \sum_1^T \mathcal{L}_t(\theta_t) - \min_\theta\sum_1^T \mathcal{L}_t(\theta)
        $$
        
        - learner의 누적 loss에서 best learner의 누적 loss를 뺀 값
            - best learner : 최고의 누적 로스를 구하기 때문에 뒤늦게 발견할 수도 있다(hindsight)
        - 모든 task를 미리 알고 있다면, 모든 task에 대해 최적의 세타를 찾았을 때 이 regret이 어떻게 커질지
            - oracle과 같다 = 궁극적 목표?
        - 한가지 주목할 것은 regret은 선형적으로 증가
            - 각 작업을 처음부터 학습하는 알고리즘이 있다고 가정할 때, 매번 새로운 태스크 받으면 처음부터 훈련한다고 가정
            - 이 기간은 t에 따라 선형적으로 증가함(t는 설정에 따라 달라지지만 현재 데이터 포인트라고 가정)
            - 따라서 모든 태스크에서 동일한 난이도라고 가정
            - regeret은 같은 양만큼 점점 커지게 됨
            - 따라서 시간이나 태스크에 따라 선형적으로 증가
            - 각 태스크 수행할 때마다 학습한다면, 선형적 regret 있는 것
        - pro : 알고리즘 분석 유용
        - con : 매번 실행하고 실제 평가하는 것 어려움
    - positive & negative transfer
        - positive forward transfer
            - 이전 작업으로 더 잘 할수 있는 부분
            - 즉, 더 많은 작업 수행할수록 향후 작업 잘 수행
            - 미래작업을 처음부터 학습하는 것과 비교됨
        - negative backward transfer
            - 과거 작업을 처음부터 다시 학습하는 것보다, 현재 작업은 이전 작업을 더 잘 수행하지 못하게 함
            - forgetting?

## 2. Basic approaches to Lifelong Learning

1. Follow the reader algorithm
    - 지금까지 본 모든 데이터를 저장하고, 이를 기반으로 학습
    - 가장 잘 작동한다는 장점
    - 계산 집약적 → 이때, continuous fine-tuning이 도움이 될 수 있다.
    - 전체 데이터셋을 불러와야 하기 때문에 memroy 집약적이 될 수 있음
2. stochastic gradient descent
    - 계산적으로 저렴하다
    - 0메모리가 필요함
    - negative backward transfer = “forgetting”에 취약
    - 느린 학습
3. Very simple continual RL algorithm
    
    [](https://arxiv.org/pdf/2004.10190.pdf)
    
    ![image](https://user-images.githubusercontent.com/101931446/221550407-0e9e3e41-4529-4345-b337-e71195d43420.png)
    
    - 매우 간단한 알고리즘
    
    ![image](https://user-images.githubusercontent.com/101931446/221550520-d7828a6d-beb1-4865-86fa-ec6210f78843.png)
    
    - 로봇이 작동하지 않는  여러개의 시나리오
    
    ![image](https://user-images.githubusercontent.com/101931446/221550604-877aab61-abdc-47a8-ad1f-ae40c13b7d9f.png)
    
    - lifelong learning에 적용
    - 기본적으로 follow the reader algorithm
        - 작은 조각들을 수집
        - 모든 작업에 대한 학습이 아니라 fine-tuning 수행
    - 여기선 강화학습으로 이 작업 수행

## 3. Better than the Basics

 ✏ **Case Study : Can we modify vanilla SGD to avoid negative backward transfer?**

- vanilla SGD : 확률적 gradient 하강
- 여기선 minus backward transfer에 더 초점 → 망각의 개념

[](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)

1. Idea
    - negative backward transfer를 피하는 것에 초점 → forgetting
    - 각 task 당 소량의 데이터를 메모리에 저장 (전체 데이터셋 X)
    - 새로운 task에 대한 update 이루어질 때, 이전 task를 잊어버리지 않도록
    - 하지만 이 문제를 어떻게 달성? → **optimization** **관점**에서 해결하고자 함
        - learning predictor $y_t = f_\theta(x_t, z_t)$
        - memory :  $\mathcal{M}_k$ for task $z_k$
        - minimize : $\mathcal{L}(f_\theta(\cdot, z_t), (x_t, y_t))$
        - subject to :  $\mathcal{L}(f_{\theta}, \mathcal{M}_k) \leq  \mathcal{L}(f_\theta^{t-1}, \mathcal{M}_k)$ for all $z_k < z_t$
        - assume local linearity : $\large <g_t, g_k> = <{{\partial\mathcal{L}(f_\theta, (x_t, y_t))} \over {\partial\theta}}, {{\mathcal{L}(f_\theta, \mathcal{M}_k)} \over {\partial\theta}} >$  $\geqq$ 0 for all $z_k < z_t$
2. Experiments
    
    ![image](https://user-images.githubusercontent.com/101931446/221550705-41071466-ff5a-4d64-b80e-7a65d290e8a9.png)
    
    - GEM : 가장 높은 정확도
    - 성능 저하는 있지만, 제약조건 때문에 일정 정확도 유지
3. Can we meta-learn how to avoid negative backward transfer?
    
    [](https://proceedings.neurips.cc/paper/2019/file/f4dd765c12f2ef67f98f3558c282a9cd-Paper.pdf)
    
    - negative backward transfer 방지하는 방법 중 또 다른 하나
    - 메타러닝 이용

## 4. Revisiting the problem statement (from the meta-learning perspective)

1. Online Learning
    
    ![image](https://user-images.githubusercontent.com/101931446/221550773-641e538a-8b1b-43e6-8469-361154e40c32.png)
    
    - static regret을 최소화하면서 일련의 task 수행
    - 작업이 순서대로 주어지고, 새로운 task 주어지는 즉시 성과 측정
    - 따라서 적응할 시간이 필요 없다 → **zero shot performance**
    
    ![image](https://user-images.githubusercontent.com/101931446/221550850-d4e24a80-fd6c-49c4-823f-ce3283ca5c30.png)
    
    ![image](https://user-images.githubusercontent.com/101931446/221550903-9fba7173-3e9d-4e16-af58-8035713360fd.png)
    - 하지만 현실적으로 학습 시간이 조금 필요
    - 작업이 무엇인지 파악하고 평가하면서 점점 더 빠르게 학습 가능
2. Online Meta-Learning
    
    [](https://arxiv.org/pdf/1902.08438.pdf)
    
    - 논문
        - 이 논문에서 online meta-learning 공식 제안
        - non-stationary 분포에서 일련의 task를 효율적으로 학습
        - 이 경우 성능평가는 약간의 학습 후에 이뤄짐
        - 데이터보다 평가에 차이 있음을 유의 (순차적으로 작업 주어지는 것은 동일하지만 평가 방식이 달라지는 것, 약간의 적응 위한 학습 기간 주어짐)
    - Setting
        - 시간 간격으로 주어진 작업에 대한 학습 데이터셋인 $D^{tr}_t$ 을 관찰
        - 그 다음 파라미터 $\Phi(\theta_t, D_t^{tr})$ 을 만들기 위해 업데이터 과정 거침
        - $x_t$ 를 관찰하고, 파라미터 사용하여 label  $\hat{y}_t =  f_{\Phi_t}(x_t)$ 예측
        - label $y_t$ 관찰
        - 하지만, 학습 기간 두어서 더 나은 업데이트 학습한다는 것이 차이점
    - Goal
        - learning algorithm with sub-linear $Regret_t := \sum_{t=1}^{T} \mathcal{l}_t(\Phi_t(\theta_t)) - \min_{\theta\in\Theta}\sum_{t=1}^T \mathcal{l}_t(\Phi_t(\theta))$
3. Can we apply meta-learning in lifelong learning settings?
    - Follow The Leader 알고리즘
        - 지금까지 본 모든 데이터 저장
        - 그리고 train
        - 끝나면 현재 task에서 배포
    - Follow the Meta-Leader 알고리즘
        - 지금까지 확인된 모든 데이터를 저장하되, 이를 단순 학습시키는 것이 아니라 meta-training으로 더 나은 update
        - 그런 다음 현재 task에서 update 실행
    - 어떤 메타러닝 알고리즘에 FTML에 잘 맞을까?
        
        ![image](https://user-images.githubusercontent.com/101931446/221551017-7351a612-f175-4b61-ac45-a30852278e8d.png)
        
        - MAML 알고리즘 사용하여 Meta training
4. Experiments
    - tasks
        - colored, rotated, scaled MNIST
        - 3D object pose prediction
        - CIFAR-100 classification
    - Comparisons
        
        ![image](https://user-images.githubusercontent.com/101931446/221551088-55ac61f3-1b46-489c-acf9-3d7496328e5d.png)
        
        - TOE : 모든 데이터로 학습
        - FTML : 학습 기간 데이터로 fine-tune
        - From Scratch : 각 task에 대해 처음부터 다시 train
        
        ⇒ FTML 사용하면 새로운 작업을 더 빨리, 더 능숙하게 학습할 수 있다.