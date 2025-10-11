import cv2
import os
import time
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np
import matplotlib.pyplot as plt


class Preprocessing:
    def __init__(self):
        # Create a folder to store captured images if it doesn't exist
        self.output_folder = 'training_images'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def open_camera(self):
        # Initialize camera capture
        self.cap = cv2.VideoCapture(1)

    # Function to detect and capture only the face as the image
    def capture_training_image(self):
        self.open_camera()
        # Prompt user to input their name
        name = input("Enter your name: ")

        # Load the existing images
        existing_images = [img for img in os.listdir(self.output_folder) if img.endswith('.jpg')]
        img_counter = len(existing_images)

        # Counter for newly captured images
        new_img_counter = 0

        # Initialize start_time variable
        start_time = time.time()

        # Wait for the user to press Enter to start capturing images
        input("Press Enter to capture image...")
        print()

        # Main loop to capture images
        while new_img_counter < 10:
            current_time = time.time()
            if current_time - start_time >= 2:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture image")
                    break

                # Display the video feed
                cv2.imshow('Video Feed', frame)

                # Detect faces in the frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    # Crop the detected face region
                    face_region = gray_frame[y:y+h, x:x+w]

                    # Resize the face region
                    resized_face = cv2.resize(face_region, (200, 200))  # Change dimensions as needed

                    # Save the captured face image with the user's name
                    img_name = os.path.join(self.output_folder, f"{name}_{img_counter}.jpg")
                    cv2.imwrite(img_name, resized_face)
                    print(f"Face {img_name} saved!")
                    img_counter += 1
                    new_img_counter += 1

                # Check for key press event to quit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                # Capture images every 2 seconds
                current_time = time.time()

        print()
        self.data_preparation()
        # Release the camera and close OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def random_brightness(self, image, min_brightness=0.5, max_brightness=1.8):
        """
        Randomly adjusts the brightness of an image within a specified range.

        Args:
        - image: Input image (numpy array).
        - min_brightness: Minimum brightness adjustment factor. Default is 0.5.
        - max_brightness: Maximum brightness adjustment factor. Default is 1.8.

        Returns:
        - adjusted_image: Image with randomly adjusted brightness.
        """
        brightness_factor = np.random.uniform(min_brightness, max_brightness)
        adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        return adjusted_image

    def data_preparation(self):
        # Initialize lists to store face images and corresponding labels
        self.x = []  # Face images
        self.y = []  # Labels

        # Load the captured face images and labels
        for img_name in os.listdir(self.output_folder):
            if img_name.endswith('.jpg'):
                # Extract label from image filename
                label = img_name.split('_')[0]
                
                # Read the image and convert to grayscale
                img_path = os.path.join(self.output_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize image to a fixed size (if needed)
                img = cv2.resize(img, (200, 200))  # Change dimensions as needed

                img = self.random_brightness(img)
                
                # Flatten the image into a 1D array and append to X
                self.x.append(img.flatten())
                
                # Append the label to y
                self.y.append(label)

        # Convert lists to numpy arrays
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        print("The Shape of x data is", self.x.shape)
        print("The Shape of y data is", self.y.shape)

        # Split the dataset into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        self.NOBODY = 'Unknown'

        # the people in the test data but not in the training set are 'nobody'.
        # mark people who is nobody for test data
        for i in range(len(self.y_test)):
            if not self.contains(self.y_train, self.y_test[i]):
                self.y_test[i] = self.NOBODY

        count = 0
        for i in range(len(self.y_test)):
            if self.y_test[i] == self.NOBODY:
                count += 1
        print((count / len(self.y_test) * 100), '% people in the close test set are NOBODY')

    # whether 'dataset' contains 'a_data'
    def contains(self, dataset, a_data):
        for v in dataset:
            if v == a_data:
                return True;
        return False


class Improved_PCA_FaceRecognizer(Preprocessing):
    def __init__(self):
        super().__init__()
        #default
        self.data_preparation()
        self.best_components = 19
        self.best_thres = 0.3
        self.max_acc = 0

    def display_every_component_analysis(self):
        component_acc_tuple = []
        plot_data = []
        for i in range(1, 41, 1):

            pca = PCA(whiten=True, n_components=i)
            self.x_train_new = pca.fit_transform(self.x_train)
            self.x_test_new = pca.transform(self.x_test)

            #Calculate the centroids
            self.centroids = {}
            for label in np.unique(self.y_train):
                self.centroids[label] = np.mean(self.x_train_new[self.y_train == label], axis=0)

            ran_arr = np.append(np.random.rand(30) * 0.5, np.random.rand(20) * 0.3)
            
            best_thres = 0
            max_acc = 0

            thres_history = []
            acc_history = []

            for thres in np.sort(ran_arr):
                result = self.predict(self.x_test_new, thres)
                accuracy = self.acc(result, self.y_test)

                thres_history.append(thres)
                acc_history.append(accuracy)

                if accuracy > max_acc:
                    max_acc = accuracy
                    best_thres = thres

            print("Number of Principal (", i, ") The best threshold is", best_thres, "and the maximum accuracy is", max_acc)
            component_acc_tuple.append((i, max_acc, best_thres))
            plot_data.append((thres_history, acc_history))

        # Define a single figure and axes for subplots
        fig1, axs1 = plt.subplots(4, 5, figsize=(15, 9))
        fig2, axs2 = plt.subplots(4, 5, figsize=(15, 9))

        # Flatten the axes array for easier indexing
        axs1 = axs1.flatten()
        axs2 = axs2.flatten()

        for i, data in enumerate(plot_data):
            thres_history, acc_history= data
            if i < 20:
                # Plot the data on the corresponding subplot
                axs1[i].scatter(thres_history, acc_history)
                axs1[i].set_title("Component = %d" % (i+1))
                axs1[i].set_xlabel('Distance threshold')
                axs1[i].set_ylabel('Accuracy')
            else:
                axs2[i-20].scatter(thres_history, acc_history)
                axs2[i-20].set_title("Component = %d" % (i+1))
                axs2[i-20].set_xlabel('Distance threshold')
                axs2[i-20].set_ylabel('Accuracy')

        # Adjust layout to prevent overlap
        fig1.tight_layout()
        fig2.tight_layout()

        # Save the figure as an image
        fig1.savefig('plots_1_eigenface.png')
        fig2.savefig('plots_2_face.png')

        plt.show()

        plt.title("EigenFace_Relationship between component number and acc")
        plt.ylabel('Accuracy')
        plt.xlabel('Component number')
        plt.scatter([i for i, _, _ in component_acc_tuple], [acc for _, acc, _ in component_acc_tuple])
        plt.savefig('Relationship_between_components_accuracy.png')
        plt.show()

        best_acc = 0
        index = 0
        best_threshold = 0
        for i, acc, thres in component_acc_tuple:
            if acc > best_acc:
                index = i
                best_acc = acc
                best_threshold = thres
        print("The number of",index, "components have an accuracy of", best_acc, "with the threshold", best_threshold)
        print("After analyse the graph, pick the best component after a few component of the determined components")
        self.max_acc = best_acc
        self.best_components = int(input("Please input the best number of components: "))
        self.best_thres = float(input("Please input the best distance threshold: "))

    def train_eigenface(self, n_components = 15):
        # Apply PCA on th e data
        if self.best_components == 0:
            pca = PCA(whiten=True, n_components = n_components)
        else:
            n_components = self.best_components
            pca = PCA(whiten=True, n_components = n_components)

        self.x_train_new = pca.fit_transform(self.x_train)
        self.x_test_new = pca.transform(self.x_test)

        print("The Shape of x train data after PCA is", self.x_train_new.shape)
        print("The Shape of x test data after PCA is", self.x_test_new.shape)

        #Calculate the centroids
        self.centroids = {}
        for label in np.unique(self.y_train):
            self.centroids[label] = np.mean(self.x_train_new[self.y_train == label], axis=0)

        result = self.predict(self.x_test_new)
        print("The accuracy of prediction using pca data is", self.acc(result, self.y_test))
        print("The maximum correct distance is", self.get_max_correct_dist(self.x_test_new, self.y_test))
        # self.best_thres = self.get_max_correct_dist(self.x_test_new, self.y_test)
        ran_arr = np.random.rand(100)
        best_thres = 0
        max_acc = 0

        thres_history = []
        acc_history = []

        # Use different threshold to determine the maximum accuracy
        for thres in np.sort(ran_arr):
            result = self.predict(self.x_test_new, thres)
            accuracy = self.acc(result, self.y_test)
            
            thres_history.append(thres)
            acc_history.append(accuracy)
            
            if accuracy > max_acc:
                max_acc = accuracy
                best_thres = thres
            
        print("The best threshold is", best_thres, "and the maximum accuracy is", max_acc)

        # Plot the relationship of threshold and accuracy
        plt.title("component = {}".format(n_components))
        plt.ylabel('accuracy')
        plt.xlabel('distance threshold')
        plt.scatter(thres_history, acc_history)
        plt.show()

        #Save PCA model
        joblib.dump(pca, 'pca_x_train_eigen.joblib')
        #Save centroids
        self.save_centroids()
    
    def performance_analysis(self):
        self.train_eigenface()
        start = time.time()
        result = self.predict(self.x_test_new, self.best_thres)
        end = time.time()
        accuracy = self.acc(result, self.y_test) * 100
        print("The accuracy of the model is {}%".format(accuracy))
        print("prediction time of close test face, ", end-start)
        print('prediction time per face is: ', (end - start)/ len(self.x_test_new))
        target_names = list(self.centroids.keys())
        target_names.append("Unknown")
        print(classification_report(self.y_test, result, labels=np.unique(self.y_test), target_names=target_names))
        print("    accuracy score       {:.2f}".format(accuracy/100))

    # norm L2 distance
    def distance(self,arr1, arr2):
        sub = np.subtract(arr1, arr2)
        return np.sqrt(np.dot(sub, sub)) / len(arr1)

    def predict(self, x_test_, thres = sys.maxsize):
        result = []
        for i in range(len(x_test_)):
            min_dist = sys.maxsize
            distance = {}

            for label, centroid in self.centroids.items():
                distance[label] = self.distance(x_test_[i], centroid)     
                # print("Distance label is" ,distance[label])
                if distance[label] < min_dist:
                    min_dist = distance[label]
            
            predicted_label = min(distance, key = distance.get)
            if min_dist > thres:
                result.append(self.NOBODY)
            else:
                result.append(predicted_label)
        return result
    
    def get_max_correct_dist(self, x_test_, y_test_):
        max_correct_dist = 0
        for i in range(len(x_test_)):
            min_dist = sys.maxsize

            distance = {}
            for label, centroid in self.centroids.items():
                distance[label] = self.distance(x_test_[i], centroid)

                if distance[label] < min_dist:
                    min_dist = distance[label]
            
            predicted_label = min(distance, key = distance.get)
            if predicted_label == y_test_[i]:
                if min_dist > max_correct_dist:
                    max_correct_dist = min_dist
        
        return max_correct_dist
    
    def acc(self, preds, y_test_):
        return np.mean(preds==y_test_)
    
    def save_centroids(self):
        # Save centroids to a file
        with open("centroids_v2.npy", "wb") as f:
            np.save(f, self.centroids)

    # Function to load the trained model for face detection
    def load_centroids(self):
        # Load centroids from file
        with open("centroids_v2.npy", "rb") as f:
            centroids = np.load(f, allow_pickle=True).item()
        print("Face recognition system loaded successfully.")
        self.centroids = centroids
        # return centroids
    
    def load_pca_model(self):
        pca = joblib.load("pca_x_train_eigen.joblib")
        print("Face recognition system loaded successfully.")
        return pca

    # Function to recognize faces using the trained model
    def recognize_faces(self):
        #Load the train model
        self.open_camera()
        self.load_centroids()
        PCA = self.load_pca_model()

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Main loop for real-time face detection
        while True:
            # Capture frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # Iterate over detected faces
            for (x, y, w, h) in faces:
                # Crop face region
                face_region = gray_frame[y:y+h, x:x+w]

                # Resize face region to match training image size
                resized_face = cv2.resize(face_region, (200, 200))  # Change dimensions as needed

                # Flatten the face region for prediction
                flattened_face = resized_face.flatten().reshape(1, -1)

                flattened_face = PCA.transform(flattened_face)

                flattened_face = flattened_face.reshape(-1)

                # Predict the label and confidence of the face using the trained model
                result = []
                thres = sys.maxsize
                min_dist = sys.maxsize
                distance = {}
                for label, centroid in self.centroids.items():
                    
                    distance[label] = self.distance(flattened_face, centroid)
                    # print(distance[label])
                    if distance[label] < min_dist:
                        min_dist = distance[label]
                
                predicted_label = min(distance, key = distance.get)
                if min_dist > self.best_thres:
                    print("Unknown")
                    result.append("Unknown")
                else:
                    print(predicted_label)
                    result.append(predicted_label)

                # Draw rectangle around the face and display name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, result[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display frame with detected faces
            cv2.imshow('Face Detection', frame)

            # Check for key press event to quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # Release the camera and close OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = Improved_PCA_FaceRecognizer()
    print("-"*80)
    print("Description: If you are a new user, please follow our guide by following the steps given.\nThe first step will capture you image to save your face into the database. \nOnce start, the program will take a photo at a 2 second interval. You may do minor face expression\nSecond, will display analyse the number of components then determine the number of components to be chosen and the threshold. \nThird, will train the face recognition system. \nFourth will start the real time face recognition.\nFifth, will learn about the prediction time per image")
    while True:
        print("-"*80)
        print("\t\tIMPROVED EIGENFACE RECOGNITION SYSTEM")
        print("-"*80)
        print("1. Capture Training Image")
        print("2. Display Every Component")
        print("3. Train EigenFace Recognition System")
        print("4. Show performance of face recognition")
        print("5. Recognize face")
        print("6. Exit")
        print("-"*80)
        n = eval(input("Enter your choice: "))
        print("-"*80)

        if n == 1:
            # Call function to capture and add new face images to the dataset
            system.capture_training_image()
        elif n == 2:
            # Display performance analysis
            system.display_every_component_analysis()
        elif n == 3:
            # Train the eigenface recognition system using the captured images
            system.train_eigenface()
        elif n == 4:
            # Show prediction per image
            system.performance_analysis()
        elif n == 5:
            # Recognize faces using the trained model
            system.recognize_faces()
        elif n == 6:
            print("Thank you for using the system")
            print("-"*80)
            break
        else:
            print("Invalid Choice")
