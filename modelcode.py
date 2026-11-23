import os
import numpy as np
import cv2
import pickle
from google.colab import files
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

print("Libraries successfully imported.")

 Load pre-trained MobileNetV2 on ImageNet without the classifier top
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_features(img_path):
    # Load and resize image
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # Extract features
    features = base_model.predict(img_data)
    return features.flatten()

    # Creating dummy dataset directories
os.makedirs('dataset/person_A', exist_ok=True)
os.makedirs('dataset/person_B', exist_ok=True)

# Creating dummy placeholder images for demonstration
placeholder_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
cv2.imwrite('dataset/person_A/face1.jpg', placeholder_img)
cv2.imwrite('dataset/person_A/face2.jpg', placeholder_img)
cv2.imwrite('dataset/person_B/face1.jpg', placeholder_img)
cv2.imwrite('dataset/person_B/face2.jpg', placeholder_img)

# Step 1: Extract features and label them
X = []
y = []

dataset_path = 'dataset'
for label_name in os.listdir(dataset_path):
    label_dir = os.path.join(dataset_path, label_name)
    if os.path.isdir(label_dir):
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(label_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 2: Train Classifier (SVM)
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y_encoded)

# Step 3: Save Model and Label Encoder together
model_package = {
    'classifier': classifier,
    'label_encoder': le
}

model_filename = 'attendance_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_package, f)

print(f"Model training completed and saved to {model_filename}")