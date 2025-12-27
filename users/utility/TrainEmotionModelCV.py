import cv2
import os
import numpy as np
from django.conf import settings

def train_opencv_emotion_model():
    print("Starting OpenCV LBPH Face Recognizer Training...")
    
    # Define mapping to match the one in StartHumanEmotions
    # emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    # Folders in data/train: angry, disgusted, fearful, happy, neutral, sad, surprised
    
    label_map = {
        "angry": 0,
        "disgusted": 1,
        "fearful": 2,
        "happy": 3,
        "sad": 4,
        "surprised": 5,
        "neutral": 6
    }
    
    train_dir = os.path.join(settings.MEDIA_ROOT, 'data', 'train')
    
    faces = []
    labels = []
    
    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        return False

    classes = os.listdir(train_dir)
    print(f"Found classes: {classes}")

    count = 0
    for emotion_name in classes:
        emotion_dir = os.path.join(train_dir, emotion_name)
        if not os.path.isdir(emotion_dir):
            continue
            
        label_id = label_map.get(emotion_name)
        if label_id is None:
            print(f"Skipping unknown emotion folder: {emotion_name}")
            continue
            
        print(f"Processing {emotion_name} (Label: {label_id})...")
        
        # Limit number of images per class to avoid taking too long/memory usage if dataset is huge
        # But for best results, we should use all. 
        # FER2013 train set is ~28k images. LBPH might be slow to train with all.
        # Let's try reasonable limit, maybe 500 images per class for speed? 
        # Or better: all images. LBPH is relatively fast.
        
        image_files = os.listdir(emotion_dir)
        # Increase limit for better accuracy
        # 'disgusted' only has 436 images, others have thousands.
        # 1000 is a good balance between speed and accuracy for LBPH.
        image_files = image_files[:1000]
        
        for image_name in image_files:
            image_path = os.path.join(emotion_dir, image_name)
            try:
                # Read image in grayscale
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Resize to standard size (usually 48x48 in FER)
                img = cv2.resize(img, (48, 48))
                
                faces.append(img)
                labels.append(label_id)
                count += 1
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
                
    print(f"Total training images: {count}")
    
    if count == 0:
        print("No images found for training.")
        return False
        
    # Create LBPH Face Recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    print("Training model...")
    # Train
    face_recognizer.train(faces, np.array(labels))
    
    output_path = os.path.join(settings.MEDIA_ROOT, 'face_emotion_model.xml')
    face_recognizer.write(output_path)
    print(f"Model saved to {output_path}")
    return True

if __name__ == "__main__":
    # Setup Django settings manually if run as script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "EGGHumanEmotion.settings")
    import django
    django.setup()
    
    train_opencv_emotion_model()
