import face_recognition
import cv2
import numpy as np 
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

sudhir_image = face_recognition.load_image_file("Photos/sudhir.jpg")
sudhir_image_encoding = face_recognition.face_encodings(sudhir_image)[0]

prem_image = face_recognition.load_image_file("Photos/prem.jpg")
prem_image_encoding = face_recognition.face_encodings(prem_image)[0]

ritesh_image = face_recognition.load_image_file("Photos/ritesh.jpg")
ritesh_image_encoding = face_recognition.face_encodings(ritesh_image)[0]

rajeev_image = face_recognition.load_image_file("Photos/rajeev.jpg")
rajeev_image_encoding = face_recognition.face_encodings(rajeev_image)[0]


known_faces_encoding = [
    sudhir_image_encoding,
    prem_image_encoding,
    ritesh_image_encoding,
    rajeev_image_encoding
]
known_faces_names = [
    "Sudhir Kumar",
    "Prem Prakash",
    "Ritesh Kumar",
    "Rajeev Kumar"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
S = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

file = open(current_date + '.csv', 'w+', newline='')
csv_writer = csv.writer(file)

while True:
    _, frame = video_capture.read()

    if frame is None:
        break

    crop_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_crop_frame = crop_frame[:, :, ::-1] 

    if S:
        face_locations = face_recognition.face_locations(rgb_crop_frame)
        face_encodings = face_recognition.face_encodings(rgb_crop_frame, face_locations)

        face_names = []

        for face_encodings in face_encodings:
            matches = face_recognition.compare_faces(known_faces_encoding, face_encodings)
            name = ""

            face_distances = face_recognition.face_distance(known_faces_encoding, face_encodings)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
            
            face_names.append(name)

            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = datetime.now().strftime("%H-%M-%S")
                    csv_writer.writerow([name, current_time])
            
    cv2.imshow("Capturing Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
file.close()
