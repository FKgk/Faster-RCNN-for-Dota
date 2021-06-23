# Enhanced Small object detection in aerial image
Using Faster-RCNN-for-Dota

## Copy and paste augmentation

32x32 이하의 작은 객체에 대해 오버샘플링을 수행하여 작은 객체 수를 증가시키는 오그멘테이션 작업
객체 크기는 랜덤하게 +-20%, 회전도 +-15%로 랜덤하게 수행

1. demo.py 파일에서 데이터셋 디렉토리를 설정한 후, 실행

https://github.com/gmayday1997/SmallObjectAugmentation 코드를 기반으로 구현함
