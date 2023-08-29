from flask import Flask, jsonify, request
import pickle
import face_recognition
import os
from io import BytesIO
import base64
from io import BytesIO
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    known_faces = []
    known_names = []

    known_faces_folder = "C:/Users/Kuldeep Singh/OneDrive/Desktop/synclovisApi/known_faces"  # Replace with the folder path containing known face images

    for image_filename in os.listdir(known_faces_folder):
        if allowed_file(image_filename):
            image_path = os.path.join(known_faces_folder, image_filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]  # Assuming there's only one face in each image
            known_faces.append(encoding)
            known_names.append(os.path.splitext(image_filename)[0])  # Use the image filename as the known name

    knn_clf = KNeighborsClassifier(n_neighbors=1)
    knn_clf.fit(known_faces, known_names)

    with open("trained_knn_model.clf", 'wb') as f:
        pickle.dump(knn_clf, f)

    @app.route('/api/facedetect', methods=['GET','POST'])
    def recognize_face():
        if request.method == 'GET':
            resp = jsonify({"message": "Error", "data": "Method not allowed"})
            resp.status_code = 405
            return resp
        elif request.method == 'POST':
            if 'face' not in request.form:
                resp = jsonify({"message": "Error", "data": "No face found"})
                resp.status_code = 400
                return resp

            decoded_image_data = base64.b64decode(request.form['face'])

            # Load the decoded image data using face_recognition
            image = face_recognition.load_image_file(BytesIO(decoded_image_data))

            face_encodings = face_recognition.face_encodings(image)
               # Make predictions only if a confident match is found
        results = []
        for face_encoding in face_encodings:
            closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=1)
            is_match = closest_distances[0][0][0] <= 0.6  # Adjust the threshold as needed
            if is_match:
                prediction = knn_clf.predict([face_encoding])[0]
                results.append({"name": prediction})
            else:
                results.append({"name": "Unknown"})

        resp = jsonify({"message": "success", "data": results})
        resp.status_code = 200
        return resp

    app.run(host='0.0.0.0', port=5000, debug=True)
