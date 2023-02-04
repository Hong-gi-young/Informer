# Informer
#### Informer 모델은 AAAI 학회에서 Outstanding paper award를 수상한 논문이며
Transformer를 기반으로 long sequence의 시계열 예측을 수행하는 모델입니다.
Informer를 다변량의 단기간 예측하는 모델로 수정하여 진행했습니다

# explain
본 프로젝트에서는 회귀/범주형으로 각기 수행했으며 기존 소스코드의 구조 및 코드를 변경하여 수행했습니다.
회귀와 범주형이 혼합된 프로젝트였으므로 data_loader.py를 각기 다르게 수정해야 했으므로 추후 수월한 관리 차원에서 각기 다른 파일로 구현했습니다.
예측기간은 1일로 설정하여 진행했습니다.

# Usage
#### Target data 형식인 범주형, 연속형에 따라 이분화하여 수정하였습니다. 
* reg_exp_main.py ▶ 연속형
* cls_exp_main.py 파일 실행 ▶ 범주형 

# result


