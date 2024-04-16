"# LeNeT5" 
# 1. 파라미터 개수
## LeNet5

### LeNet-5의 파라미터 수 계산
LeNet-5 모델은 일반적으로 다음과 같은 구조를 가집니다:
1. 첫 번째 합성곱 계층: 6개의 (5 *5) 필터 => 6 *(5 *5 *1 + 1) = 156 파라미터
2. 두 번째 합성곱 계층: 16개의 5 * 5 필터, 입력 채널 6 => 16 * (5 * 5 * 6 + 1) = 2416 파라미터
3. 세 번째 완전 연결 계층: 120개의 뉴런, 입력은 16 * 5 * 5 = 400 => 120 * (400 + 1) = 48120 파라미터
4. 네 번째 완전 연결 계층: 84개의 뉴런, 입력은 120 => 84 *(120 + 1) = 10164 파라미터
5. 출력 계층: 10개의 출력, 입력은 84 => 10 *(84 + 1) = 850 파라미터

이를 모두 더하면, LeNet-5의 총 파라미터 수는:
156 + 2416 + 48120 + 10164 + 850 = 61706 

###  CustomMLP 모델의 파라미터 계산
MLP 모델이 LeNet-5와 비슷한 파라미터 수를 갖기 위해, 레이어의 크기를 조정해야 할 수 있습니다. 예를 들어, MLP 모델의 기존 구성이 다음과 같다고 가정합니다:
1. 첫 번째 완전 연결 계층: 입력 784, 출력 100
2. 두 번째 완전 연결 계층: 입력 100, 출력 50
3. 출력 계층: 입력 50, 출력 10


이 구성의 파라미터 수를 계산해보면:
1. 784 *75 + 75 = 58875
2. 75*30+30 = 2280
3. 30*10+10=310

총합: 58875+2280+310=61465

---
# 2. 모델 손실 밑 정확도 곡선
LeNet-5 모델과 CustomMLP모델 모두 epoch=10
## LeNet-5 모델 성능(test_LeNet5_CustomMLP.txt에서 확인 가능)

Train_avg_Loss:0.0552,Train_avg_Accuracy:98.23,Test_avg_Loss:0.0424,Test_avg_Accuracy:98.62%

## CustomMLP 모델 성능(test_LeNet5_CustomMLP.txt에서 확인 가능)

Train_avg_Loss:0.0772,Train_avg_Accuracy:97.65,Test_avg_Loss:0.0865,Test_avg_Accuracy:97.30%
---
# 3. 알려진 LeNet-5 정확도
LeNet-5 논문 링크
<http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>

논문에 제시된 minst 데이터에 대한 error rate 그래프와 내가 짠 LeNet-5의 loss 그래프와 유사함.
또한, 실험 결과에서 테스트 데이터에 대한 오류율은 1.9%, 훈련 데이터에 대한 오류율을 1.1%에 수렴한다고 했으므로 내가 만든 LeNet-5와 유사함.(논문에서는 모델의 구성이 조금 다름)
---

# 4.  LeNet-5 모델 개선

모델 개선은 Dropout,BatchNormalize, optimizer를 Adam으로 변경 + weight_decay 추가(L2 regularization)을 하여 모델의 성능을 개선함.
epoch=20으로 설정, Dropout 비율과 batch_size를 변경해 가면서 실험
