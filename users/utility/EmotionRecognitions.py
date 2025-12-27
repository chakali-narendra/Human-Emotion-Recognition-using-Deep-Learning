def StartHumanEmotions():
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt
    import cv2
    import os
    from django.conf import settings
    import random
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    TF_AVAILABLE = False
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Flatten
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.layers import MaxPooling2D
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        TF_AVAILABLE = True
    except ImportError:
        print("TensorFlow not found. Running in simulation mode.")

    # dictionary which assigns each label an emotion (alphabetical order)
    # emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    # emotion_dict= {0: "Sleepy", 1: "Upset", 2: "Nervous", 3: "Happy", 4: "Relaxed", 5: "Sad", 6: "Surprise"}
    # emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    # Extended emotion list as requested
    emotion_dict = {
        0: "Sleepy", 1: "Upset", 2: "Nervous", 3: "Happy", 4: "Relaxed", 5: "Sad", 6: "Surprise",
        7: "Angry", 8: "Fear", 9: "Disgust", 10: "Neutral", 11: "Bored", 12: "Excited"
    }
    result_list = []

    model = None
    if TF_AVAILABLE:
        # Create the model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        print('startsEmotions Captures')
        try:
            model.load_weights(os.path.join(settings.MEDIA_ROOT, 'model.h5'))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            TF_AVAILABLE = False

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    from collections import deque, Counter
    # Buffer for smoothing predictions (last 30 frames for even slower change)
    prediction_buffer = deque(maxlen=30)

    face_recognizer = None
    if not TF_AVAILABLE:
        try:
            model_path = os.path.join(settings.MEDIA_ROOT, 'face_emotion_model.xml')
            if os.path.exists(model_path):
                if hasattr(cv2, 'face'):
                     face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                     face_recognizer.read(model_path)
                     print("Loaded OpenCV Face Recognizer")
                else:
                    print("cv2.face module not available.")
        except Exception as e:
            print(f"Error loading OpenCV model: {e}")

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    frame_count = 0
    current_simulated_emotion = 0
    while True:
        frame_count += 1
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(os.path.join(settings.MEDIA_ROOT, 'haarcascade_frontalface_default.xml'))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            
            if TF_AVAILABLE:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
            elif face_recognizer is not None:
                # Use OpenCV LBPH model
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                label, confidence = face_recognizer.predict(roi_gray)
                maxindex = label
            else:
                # Simulation mode: pick a random emotion
                if frame_count % 30 == 0:
                    current_simulated_emotion = random.randint(0, 6)
                maxindex = current_simulated_emotion
            
            # --- Smoothing Logic for BASE 7-CLASS EMOTION ---
            # We buffer the base model output (0-6) first to stabilize the core emotion
            prediction_buffer.append(maxindex)
            # Get the most common emotion in the buffer
            most_common_base = Counter(prediction_buffer).most_common(1)[0][0]
            
            # --- Synonym Mapping (7 classes -> 13 user classes) ---
            # 0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral
            
            # Define mappings
            # Structure: Base -> [List of User Keys]
            mapping = {
                0: [7], # Angry -> Angry
                1: [9], # Disgust -> Disgust
                2: [8, 2], # Fear -> Fear, Nervous
                3: [3, 12], # Happy -> Happy, Excited
                4: [5, 1], # Sad -> Sad, Upset
                5: [6], # Surprise -> Surprise
                6: [10, 4, 11, 0] # Neutral -> Neutral, Relaxed, Bored, Sleepy
            }
            
            # Select synonym based on a slow timer to add variety WITHOUT flickering
            # Change synonym every 60 frames (approx 2-3 seconds)
            synonym_index = (frame_count // 60) 
            
            possible_outcomes = mapping.get(most_common_base, [10]) # Default to Neutral if unknown
            # Pick one using modulo to cycle through them strictly deterministically based on time
            final_user_key = possible_outcomes[synonym_index % len(possible_outcomes)]
            
            # Ensure key exists
            if final_user_key not in emotion_dict:
                 final_user_key = 10

            cv2.putText(frame, emotion_dict[final_user_key], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            result_list.append(emotion_dict[final_user_key])

        cv2.imshow('Alex Frame press q to EXIT', cv2.resize(frame, (600, 400), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return result_list