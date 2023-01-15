import cv2
import numpy as np
import cv2
import mediapipe as mp
import csv
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def getHand(image):
    
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.3,
        max_num_hands=5,
        min_tracking_confidence=0.3) as hands:
# Draw the hand annotations on the image.

        image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = hands.process(image)
        if results.multi_hand_landmarks and len(results.multi_handedness) == 2:
            from google.protobuf.json_format import MessageToDict
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                print(handedness_dict)
                if handedness_dict['classification'][0]['label'] == 'Left':
                    i = 0
                    j = 1
                elif handedness_dict['classification'][0]['label'] == 'Right':
                    i = 1
                    j = 0
                #print('left')
                data[0] = results.multi_hand_landmarks[i].landmark[0].x
                data[1] = results.multi_hand_landmarks[i].landmark[0].y
                data[2] = results.multi_hand_landmarks[i].landmark[0].z
                mp_drawing.draw_landmarks(
                        image,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                #print('right')
                data[3] = results.multi_hand_landmarks[j].landmark[0].x
                data[4] = results.multi_hand_landmarks[j].landmark[0].y
                data[5] = results.multi_hand_landmarks[j].landmark[0].z
                mp_drawing.draw_landmarks(
                        image,
                        results.multi_hand_landmarks[1],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                break

        if results.multi_hand_landmarks and len(results.multi_handedness) == 1:
            from google.protobuf.json_format import MessageToDict
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                print(handedness_dict)
                if handedness_dict['classification'][0]['label'] == 'Left':
                    data[0] = results.multi_hand_landmarks[0].landmark[0].x
                    data[1] = results.multi_hand_landmarks[0].landmark[0].y
                    data[2] = results.multi_hand_landmarks[0].landmark[0].z
                    mp_drawing.draw_landmarks(
                        image,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                elif handedness_dict['classification'][0]['label'] == 'Right':
                    data[3] = results.multi_hand_landmarks[0].landmark[0].x
                    data[4] = results.multi_hand_landmarks[0].landmark[0].y
                    data[5] = results.multi_hand_landmarks[0].landmark[0].z
                    mp_drawing.draw_landmarks(
                        image,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
    return image, data

def getPose(image):

    with mp_pose.Pose(
        static_image_mode = True,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as pose:
# Draw the hand annotations on the image.
        image_height, image_width, _ = image.shape
        image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = pose.process(image)
        if results.pose_landmarks != None:
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]:
                data[7] = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x #* image_width # supposedly these are distances from from midpoiint of hip, being origin,
                data[8] = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y #* image_width
                data[9] = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z #* image_width
                
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]:
                data[10] = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x #* image_width
                data[11] = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y #* image_width
                data[12] = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z #* image_width
               
        else:
            pass
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return image, data

with open(os.path.join(os.getcwd(), "cat.csv"), 'w') as f:  
    writer = csv.writer(f)
    writer.writerow(['t', 'L_x', 'L_y', 'L_z', 'R_x', 'R_y', 'R_z', ' ', 'S_x', 'S_y', 'S_z', 'E_x', 'E_y', 'E_z'])

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load video
cap = cv2.VideoCapture("Intramuscular.mp4")
#cap = cv2.VideoCapture("HoonGinaInjection.avi")
#cap = cv2.VideoCapture("COLORFUL Fengze & Shou.mp4")
#cap = cv2.VideoCapture("delt inject.mp4")
#cap = cv2.VideoCapture(0)

t = 0
store = []
patient = []
nurse = []

# Process video frames
while True:
    # Read frame
    _, img = cap.read()
    height, width, channels = img.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Show results
    class_ids = []
    confidences = []
    boxes = []
    size = 14
    data = [None for i in range(size)]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes and classes[class_ids[i]] == "person":
            x, y, w, h = boxes[i]

            if x < 0 or y < 0:
                continue
            else:
                if len(store) == 0:
                    store.append(1)
                    patient.append(boxes[i])
                elif len(store) == 1:
                    store.append(1)
                    nurse.append(boxes[i])
                    if x < patient[0][0]: #switch this sign if nurse is on left
                        patient, nurse = nurse, patient
                else:
                    if abs(x - patient[-1][0]) < abs(x - nurse[-1][0]):
                        patient.append(boxes[i])
                        person = img[y:y+h, x:x+w]  # Extract person object using array slicing
                        print('person')
                        label = 'Patient'
                        color = colors[class_ids[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

                        image, data = getPose(person)
                    
                        # image.flags.writeable = False
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cv2.imshow('MediaPipe Pose', image) #cv2.flip(image, 1)
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
                    else:
                        nurse.append(boxes[i])
                        person = img[y:y+h, x:x+w]  # Extract person object using array slicing
                        print('nurse')
                        label = 'Nurse'
                        color = colors[class_ids[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
                
                        image, data = getHand(person)
                    
                        # image.flags.writeable = False
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cv2.imshow('MediaPipe Hands', image) #cv2.flip(image, 1)
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
    data = [t,] + data
    t+=1
    with open(os.path.join(os.getcwd(), "cat.csv"), 'a',  newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
    # Show frame
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

