# How it's works
1. First import the library that we need
````
import cv2
import mediapipe as mp
````
2. Make the program to connect to the webcam
```
import cv2
import numpy as numpy

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.imshow('Face Mesh', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
`````
3. Load the module of face_mesh and drawing_utils
````
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
````
4. Determine the minimum percentage
````
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
````
5. Chanfe BGR to RGB
````
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
````
6. To optimize the program change writeable to False
````
image.flags.writeable = False
````
7. processing
````
results = hands.process(image)
````
8. Change RGB to BGR 
````
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
````
9. Make a loop and draw the landmark 
````
drawing_spec = mp_drawing.DrawingSpec(color=(0,251,251),thickness=1, circle_radius=1)

if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image,face_landmarks,mp_face_mesh.FACE_CONNECTIONS,drawing_spec,drawing_spec)
````                                        