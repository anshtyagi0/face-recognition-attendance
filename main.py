import cv2
import face_recognition
from pymongo import MongoClient
from datetime import datetime
import pandas as pd

# Connect to MongoDB
client = MongoClient('Enter mongodb url.')
db = client['face_recognition']
collection = db['students']

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    known_enrollment_numbers = []

    # Fetch student data from MongoDB
    for student in collection.find():
        # Make sure 'image', 'name', and 'enrollment_number' keys are present in the document
        if all(key in student for key in ['image', 'name', 'enrollment_number']):
            image_path = student['image']
            print(f"Loading image: {image_path}")

            try:
                face_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(face_image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(student['name'])
                    known_enrollment_numbers.append(student['enrollment_number'])
                else:
                    print(f"No face detected in the image for student: {student['name']}")
            except Exception as e:
                print(f"Error loading image for student {student['name']}: {e}")

    return known_face_encodings, known_face_names, known_enrollment_numbers

def recognize_faces(frame, known_face_encodings, known_face_names, known_enrollment_numbers):
    if frame is None:
        print("Error: Unable to read frame from the camera.")
        return None

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame)

    # Loop through each face found in the current frame
    for (top, right, bottom, left) in face_locations:
        # Extract the face encoding for the current face
        face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]

        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        name = "Person not found"
        enrollment_number = "Unknown"

        # If a match was found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            enrollment_number = known_enrollment_numbers[first_match_index]

            # Check for liveness
            face_landmarks = face_recognition.face_landmarks(frame, [(top, right, bottom, left)])[0]
        
            # Draw a box around the face and display the name and enrolment number
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} - {enrollment_number}", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Update the Excel file with attendance
            update_excel(sno=1, name=name, enrollment_number=enrollment_number, course="Computer Science", present="Yes")
        else:
            # Draw a box around the face but don't display the name and enrolment number for unrecognized faces
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    return frame

def update_excel(sno, name, enrollment_number, course, present):
    now = datetime.now()
    excel_file_path = f"attendance_{now.strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"

    # Check if the Excel file exists, create a new one if not
    try:
        attendance_df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        attendance_df = pd.DataFrame(columns=['S.NO', 'Name', 'Enrolment Number', 'Course', 'PRESENT'])

    # Check if the student is already present in the Excel file
    existing_student = attendance_df[attendance_df['Enrolment Number'] == enrollment_number]

    if existing_student.empty:
        # Add a new row for the student
        new_row = {'S.NO': sno, 'Name': name, 'Enrolment Number': enrollment_number, 'Course': course, 'PRESENT': present}
        new_df = pd.DataFrame([new_row])  # Create a DataFrame from the new row
        attendance_df = pd.concat([attendance_df, new_df], ignore_index=True)

        # Save the updated DataFrame to the Excel file
        attendance_df.to_excel(excel_file_path, index=False)
        print(f"Attendance recorded for {name} ({enrollment_number})")
    else:
        print(f"{name} ({enrollment_number}) already marked as present")

def main():
    known_face_encodings, known_face_names, known_enrollment_numbers = load_known_faces()

    # Set up the camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust if needed

    while True:
        ret, frame = cap.read()

        # Recognize faces in the current frame
        frame_with_recognition = recognize_faces(frame, known_face_encodings, known_face_names, known_enrollment_numbers)

        # Display the frame with face recognition results
        cv2.imshow("Face Recognition", frame_with_recognition)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
