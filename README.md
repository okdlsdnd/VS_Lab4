<!--21700150 김인웅-->

Lab4
======

목표 : YOLOv5를 이용해 주차장 영상에서 남은 자리를 출력  

기본 영상 상태

![default](https://user-images.githubusercontent.com/80805040/169969331-65976937-202e-4613-afd0-d61acb9f022d.png)


위 영상에서 주차장 부분에 있는 차만 YOLOv5를 이용하여 검출해 내어야 한다.

#### Import Libraries

이번 Lab은 미리 훈련된 모델을 사용할 것이기 때문에 많은 라이브러리가 필요하지 않다.

    import cv2
    import torch

영상처리를 위한 cv2와 모델을 불러오기 위한 torch를 불러온다.

#### YOLOv5 모델

이번 Lab에서 사용할 모델은 YOLOv5 모델이며 그 중에서도 컴퓨터 사양과 속도, 그리고 정확성을 고려하여 yolov5l 모델을 사용하였다.

또한 더욱 정확한 검출을 위하여 Confidence와 IoU의 Threshold 값을 각각 지정해준다.

이번 Lab의 목표는 자동차 만을 검출해 내는 것이기 때문에 coco class에서 'car' 만을 사용할 것이다. 따라서 검출할 class 또한 지정해준다.

    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    
    model.conf = 0.45
    model.iou = 0.4
    model.classes = [2]

#### 영상캡쳐 및 txt 파일 저장을 위한 변수 지정

Lab을 위해 지정된 avi 파일을 불러온다.

각 프레임에 탐지된 자동차의 대수를 counting_result.txt 파일로 저장해야 한다. 따라서 우선 frame이라는 변수를 만들어 0이라는 값을 주고 txt 파일을 새로 생성한다.

	cap = cv2.VideoCapture('DLIP_parking_test_video.avi')
	
	frame = 0
	
	f = open('./counting_result.txt', 'w')

#### 영상 전처리

영상 처리는 while문 안에서 각 프레임을 이미지 파일로 만들어서 시행할 것이다.

    rect, img = cap.read()
    
    roi = img[250:400, 0:1280]
    
    img[250:400, 0:1280] = roi

이 때 주차장 안에 있는 차만을 검출해야 하기 때문에 roi를 지정해준다.

지정된 roi는 다음과 같다.


![roi](https://user-images.githubusercontent.com/80805040/169969357-ffc220bd-7a8c-444a-857d-1e5eae57f216.png)



#### 객체 탐지 

    results = model(roi)
    
    results.save()

훈련된 모델로 roi 내의 자동차를 탐지한다.  그리고 그 결과를 이미지로 만들어 저장한다. 그 이미지는 다음과 같다.
![detection](https://user-images.githubusercontent.com/80805040/169969371-932acd80-427a-449c-8b0a-e9bfff6ba2f9.jpg)

#### 탐지된 자동차 개수 계산 및 txt 파일 저장

    count = 0
    
    count = len(results.pandas().xyxy[0])
    
    available = 13 - count
    
    text = str('Parking Lots Available : %s' % (available))
    
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    
    f_results = '%s %s \n' % (frame, count)
    f.write(f_results)
    
    frame += 1

탐지된 결과는 dataframe의 형태로 저장된다. 이 때 그 길이만을 추출하여 탐지된 객체의 개수를 알아낼 수 있다. 이를 이용하여 탐지된 자동차의 대수를 찾고 화면에 그 결과를 출력한다.

또한 txt파일에 각 프레임별 탐지된 자동차 대수를 입력하고 다음 프레임으로 넘어가기 이전 frame에 1을 더해준다.

#### 종료

	cv2.imshow('result', img)
	if cv2.waitKey(1) == ord('q'):
	    break

while 문 내에서 실행 중 q를 눌렀을 때 종료되게 한다. 위 코드를 마지막으로 while문은 끝난다.

	f.close()
	cap.release()
	cv2.destroyAllWindows()

모든 영상이 끝났을 때 txt 파일을 저장하고 프로그램이 종료되게 한다.

최종적으로 얻어낸 결과는 다음과 같다.

![result](https://user-images.githubusercontent.com/80805040/169969381-d3f18b29-5876-428d-8203-062915718cf1.png)

주차장 내의 자동차만 탐지된 것을 알 수 있으며 남은 주차장 자리의 수가 영상에 출력된다.

### Appendix

##### 코드

    # 21700150 김인웅 LAB4
    
    # 모듈 Import
    import cv2
    import torch
    
    # 모델 불러오기(yolov5l) 및 treshold 지정
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    
    model.conf = 0.45
    model.iou = 0.4
    model.classes = [2]
    
    # 영상캡쳐
    cap = cv2.VideoCapture('DLIP_parking_test_video.avi')
    
    frame = 0
    
    f = open('./counting_result.txt', 'w')
    
    while True:
        rect, img = cap.read()
    
        # 주차장 roi
        roi = img[250:400, 0:1280]
    
        img[250:400, 0:1280] = roi
    
        # 객체 탐지
        results = model(roi)
    
        results.save()
    
        # 주차장 빈자리 계산 및 출력
        count = 0
    
        count = len(results.pandas().xyxy[0])
    
        available = 13 - count
    
        text = str('Parking Lots Available : %s' % (available))
    
        cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    
        # txt 파일 저장
        f_results = '%s %s \n' % (frame, count)
        f.write(f_results)
    
        frame += 1
    
        # q로 영상 종료
        cv2.imshow('result', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 종료
    f.close()
    cap.release()
    cv2.destroyAllWindows()


##### Flow Chart

![flow_chart](https://user-images.githubusercontent.com/80805040/169978907-234347f3-1c36-440f-8740-0f15d1e66f20.png)
