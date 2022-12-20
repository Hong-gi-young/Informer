# Informer
 Informer 모델은 AAAI 학회에서 Outstanding paper award를 수상한 논문으로써  Transformer를 기반으로 long sequence의 시계열 예측을 수행하는 모델을 제안했습니다.

# explain
본 프로젝트에서는 회귀/범주형으로 각기 수행했으며 기존 소스코드의 구조 및 코드를 변경하여 수행했습니다.
회귀와 범주형이 혼합된 프로젝트였으므로 data_loader.py를 각기 다르게 수정해야 했으므로 추후 수월한 관리 차원에서 각기 다른 파일로 구현했습니다.
예측기간은 1일로 설정하여 진행했습니다.

# Usage
*회귀/ 범주
각 reg_exp_main.py, cls_exp_main.py 파일 실행 

#result
회귀	cg_curtain
MSE	0.041
MAE	0.057
분류	flow_fan
Acc	0.921
F1	0.928
Recall	0.917
![image](https://user-images.githubusercontent.com/69567516/208602183-cf2ccf20-011d-4c5c-bafb-675df48ceb96.png)
