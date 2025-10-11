import cv2
import time
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class DataPreparation:
    def __init__(self):
        self.IMAGE_SIZE = (150,150)
        self.output_folder = "training_images"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.data_folder = "data"
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        self.x = []
        self.y = []

        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

        self.initiate_all()
    
#Save and Return
    def return_separation_train_data(self):
        for img_name in os.listdir(self.output_folder):
            if img_name.endswith(".jpg"):
                label = img_name.split('_')[0]

                img_path = os.path.join(self.output_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.IMAGE_SIZE)

                self.x.append(img.flatten())
                self.y.append(label)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

        # Shuffle the data
        indices = np.arange(len(self.x))
        np.random.seed(42)
        np.random.shuffle(indices)
        self.x = self.x[indices]
        self.y = self.y[indices]

        # Split data into training and testing sets
        split_ratio = 0.8
        split_index = int(split_ratio * len(self.x))
        self.x_train, self.x_test = self.x[:split_index], self.x[split_index:]
        self.y_train, self.y_test = self.y[:split_index], self.y[split_index:]

    def return_mean_face(self):
        if os.path.exists("data/mean_face.xlsx"):
            self.mean_face = pd.read_excel("data/mean_face.xlsx")
            self.mean_face = np.array(self.mean_face)
            self.mean_face = self.mean_face.flatten()

    def return_eigenvalues(self):
        if os.path.exists("data/eigenvalues.xlsx"):
            self.eigenvalues = pd.read_excel("data/eigenvalues.xlsx")
            self.eigenvalues = np.array(self.eigenvalues)

    def return_eigenvectors(self):
        if os.path.exists("data/eigenvectors.xlsx"):
            self.eigenvectors = pd.read_excel("data/eigenvectors.xlsx")
            self.eigenvectors = np.array(self.eigenvectors)
    
    def return_centered_face(self):
        if os.path.exists("data/centered_face.xlsx"):
            self.centered_faces = pd.read_excel("data/centered_face.xlsx")
            self.centered_faces = np.array(self.centered_faces)
            self.centered_faces = self.centered_faces.reshape(-1,self.eigenvectors.shape[0])

    def return_x_train_pca(self):
        if os.path.exists("data/x_train_pca.xlsx"):
            self.x_train_pca = pd.read_excel("data/x_train_pca.xlsx")
            self.x_train_pca = np.array(self.x_train_pca)

    def return_face_recognition(self):
        # Load centroids from file
        with open("centroids.npy", "rb") as f:
            centroids = np.load(f, allow_pickle=True).item()
        print("Face recognition system loaded successfully.")
        return centroids

    def save_mean_face(self):
        mean_pd = pd.DataFrame(self.mean_face)
        mean_pd.to_excel("data/mean_face.xlsx", index=False)  
        
    def save_eigenvalues(self):
        eigenvalues_pd = pd.DataFrame(self.eigenvalues)
        eigenvalues_pd.to_excel("data/eigenvalues.xlsx", index=False)
    
    def save_eigenvectors(self):
        eigenvectors_pd = pd.DataFrame(self.eigenvectors)
        eigenvectors_pd.to_excel("data/eigenvectors.xlsx", index=False)

    def save_centered_face(self):
        center_flatten = self.centered_faces.reshape(-1,5)
        centered_pd = pd.DataFrame(center_flatten)
        centered_pd.to_excel("data/centered_face.xlsx", index=False)

    def save_x_train_pca(self):
        x_train_pca = pd.DataFrame(self.x_train_pca)
        x_train_pca.to_excel("data/x_train_pca.xlsx", index=False)

#Initiate mean, eigenvalues, eigenvectors, centered face
    def initiate_all(self):
        self.return_separation_train_data()
        self.return_mean_face()
        self.return_eigenvalues()
        self.return_eigenvectors()
        self.return_centered_face()
        self.return_x_train_pca()


class FaceRecognitionSystem(DataPreparation):
    def __init__(self):
        super().__init__()
        #0 as default device, 1 as external device
        self.cap = cv2.VideoCapture(1)

        self.NUM_COMPONENTS = 35

        #0 as Singular Value Decomposition, 1 as Eigenvector Decomposition
        self.method_decomp = 0


#Data Analysis
    def display_mean_face(self):
        #Plot mean face
        mean_face = self.mean_face
        mean_face_image = mean_face.reshape(self.IMAGE_SIZE)
        plt.figure(figsize=(6, 6))
        plt.imshow(mean_face_image, cmap='gray')
        plt.title('Mean Face')
        plt.axis('off')
        plt.show()

    def display_centered_face(self):
        #Show centered faces
        # Assuming centered_faces is your array of centered face images
        num_samples = 20
        num_rows = 5
        num_cols = 4

        # Select 20 centered faces randomly
        selected_indices = np.random.choice(len(self.centered_faces), num_samples, replace=False)
        selected_faces = self.centered_faces[selected_indices]

        # Plot the selected centered faces
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 15))

        for i, ax in enumerate(axs.flat):
            ax.imshow(selected_faces[i].reshape((self.IMAGE_SIZE)), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def display_filtered_eigenvectors(self):
        filtered_eigenvectors = pd.DataFrame(self.eigenvectors)
        print(filtered_eigenvectors)

    def display_eigenfaces(self):
        num_faces_to_display = sys.maxsize
        while num_faces_to_display > self.eigenvectors.shape[1]:
            num_faces_to_display = int(input("Input the number of eigenface to be display: "))
        num_rows = num_faces_to_display // 5 + 1  # Adjust the number of rows based on the number of faces to display
        fig, axes = plt.subplots(num_rows, 5, figsize=(12, 12))

        for i in range(num_faces_to_display):
            # Reshape the eigenvector into a 2D image
            eigenvectors = self.eigenvectors
            eigenface = eigenvectors[:, i].reshape(self.IMAGE_SIZE)

            # Plot the eigenface
            row = i // 5
            col = i % 5
            axes[row, col].imshow(eigenface, cmap='gray')
            axes[row, col].set_title(f'Eigenface {i+1}')
            axes[row, col].axis('off')

        # Remove empty subplots
        for j in range(num_faces_to_display, num_rows * 5):
            row = j // 5
            col = j % 5
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()
    
    def display_explained_variance(self):
        # Create a scree plot or explained variance graph
        explained_variance_ratio = self.eigenvalues.real / np.sum(self.eigenvalues.real)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        #Show explained variance graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 51), cumulative_explained_variance[:50], marker='o', linestyle='-')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()


#Recognition
    def capture_training_images(self):
        name = input("Enter your name: ")

        existing_images = [img for img in os.listdir(self.output_folder) if img.endswith(".jpg")]
        img_counter = len(existing_images)

        new_img_counter = 0

        start_time = time.time()

        input("Please press Enter to capture your images\n")
        print()

        while new_img_counter < 10:
            current_time = time.time()
            if current_time - start_time >= 2:
                start_time = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to capture images")
                    break

                cv2.imshow("Video feet", frame)

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_region = gray_frame[y:y+h, x:x+w]

                    resized_face = cv2.resize(face_region, (200,200))

                    img_name = os.path.join(self.output_folder, f"{name}_{img_counter}.jpg")
                    cv2.imwrite(img_name, resized_face)
                    print(f"Face {img_name} saved!")
                    img_counter += 1
                    new_img_counter += 1
            
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

                current_time = time.time()

        print()

        self.cap.release()
        cv2.destroyAllWindows()

    def pca(self):
        # Calculate eigenvectors and eigenvalues with timing
        start_time = time.time()

        if self.method_decomp == 0:
            # Perform Singular Value Decomposition (SVD)
            U, s, Vt = np.linalg.svd(self.centered_faces)
            print("Done SVD")

            # Extract the top 50 highest singular values and corresponding eigenvectors
            self.eigenvalues = s[:50] ** 2
            self.eigenvectors = Vt.T[:, :50]

        if self.method_decomp == 1:
            self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)
            print("Done EigenDecomposition")

        end_time = time.time()
        calculation_time = end_time - start_time
        print(f"Time taken to compute eigenvectors and eigenvalues: {calculation_time:.4f} seconds")

        # Sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(self.eigenvalues)[::-1]  # Get indices to sort eigenvalues in descending order
        sorted_eigenvalues = self.eigenvalues[sorted_indices]
        sorted_eigenvectors = self.eigenvectors[:, sorted_indices]

        # Filter out the first 150 eigenvectors
        filtered_eigenvectors = sorted_eigenvectors[:, :50]
        filtered_eigenvectors = filtered_eigenvectors.reshape(filtered_eigenvectors.shape[0], -1)


        self.eigenvalues = sorted_eigenvalues
        self.eigenvectors = filtered_eigenvectors.real
                
        self.save_eigenvalues()

        # Display Explained Variance aka Scree Plot
        self.display_explained_variance()

        # Perform PCA
        num_components = int(input("Enter the number of components to be used: "))  # Number of principal components to retain
        self.NUM_COMPONENTS = num_components
        self.eigenvectors = self.eigenvectors[:,:self.NUM_COMPONENTS]
        self.x_train_pca = np.dot(self.centered_faces, self.eigenvectors[:, :self.NUM_COMPONENTS])
        self.save_eigenvectors()
        self.save_x_train_pca()

    def ncc(self):
        # Nearest Centroid Classifier (NCC)

        X = self.x_train_pca
        self.centroids = {}
        print(X.shape)
        for label in np.unique(self.y_train):
            self.centroids[label] = np.mean(X[self.y_train == label], axis=0)
        
        start = time.time()
        correct_predictions = 0
        total_predictions = len(self.x_test)
        result = []
        for i in range(total_predictions):
            flattened_face = np.dot(self.x_test[i] - self.mean_face, self.eigenvectors)

            true_label = self.y_test[i]

            distances = {}
            for label, centroid in self.centroids.items():
                distances[label] = np.linalg.norm(flattened_face - centroid)

            predicted_label = min(distances, key=distances.get)
            result.append(predicted_label)
            if predicted_label == true_label:
                correct_predictions += 1

        end = time.time()
        accuracy = correct_predictions / total_predictions * 100
        print(f"Face recognition system trained with accuracy: {accuracy:.2f}%")  
        print("prediction time of close test face, ", end-start) 
        print('prediction time per face is: ', (end - start)/ len(self.x_test))  
        target_names = list(self.centroids.keys())
        # target_names.append("Unknown")
        print(classification_report(self.y_test, result, labels=np.unique(self.y_test), target_names=target_names))


    def train_face_recognition(self):
        # Compute mean face
        self.mean_face = np.mean(self.x_train, axis=0)
        self.save_mean_face()
        print("Done mean face", self.mean_face.shape)

        # Center the faces
        self.centered_faces = self.x_train - self.mean_face
        self.save_centered_face()
        print("Done centered face", self.centered_faces.shape)

        # Compute covariance matrix
        centered_faces = self.centered_faces
        self.covariance_matrix = np.cov(centered_faces.T)
        print("Done covariance matrix", self.covariance_matrix.shape)

        # PCA
        self.pca()

        # Classifier 
        self.ncc()

        # Save centroids to a file
        with open("centroids.npy", "wb") as f:
            np.save(f, self.centroids)

    def recognize_faces(self): 
        centroids = self.return_face_recognition()

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = self.cap.read()
                
            if not ret:
                print("Failed to capture images")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_region = gray_frame[y:y+h, x:x+w]

                resized_face = cv2.resize(face_region, self.IMAGE_SIZE)

                flattened_face = resized_face.flatten()

                flattened_face = np.dot(flattened_face - self.mean_face, self.eigenvectors)

                # Calculate distances to centroids for each face
                distances = {}
                for label, centroid in centroids.items():
                    distances[label] = np.linalg.norm(flattened_face- centroid)

                # Predict label of the face based on the nearest centroid
                predicted_label = min(distances, key=distances.get)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Face_Detection', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    

if __name__ == "__main__":
    #Example usage
    face_recognition_system = FaceRecognitionSystem()

    while True:
        print("-"*80)
        print("\t\tEIGENFACE IN SCRATCH RECOGNITION SYSTEM")
        print("-"*80)
        print("1. Capture Training Image")
        print("2. Train EigenFace Recognition System")
        print("3. Show performance of face recognition")
        print("4. Display Mean Face")
        print("5. Display Centered Face")
        print("6. Display Explained Variance")
        print("7. Display Filtered Eigenvectors")
        print("8. Display Eigenface")
        print("9. Recognize face")
        print("10. Exit")        
        print("-"*80)
        n = eval(input("Enter your choice: "))
        print("-"*80)


        if n == 1:
            face_recognition_system.capture_training_images()

        elif n == 2:
            face_recognition_system.train_face_recognition()

        elif n == 3:
            face_recognition_system.ncc()

        elif n == 4:
            face_recognition_system.display_mean_face()

        elif n == 5:
            face_recognition_system.display_centered_face()

        elif n == 6:
            face_recognition_system.display_explained_variance()

        elif n == 7:
            face_recognition_system.display_filtered_eigenvectors()

        elif n == 8:
            face_recognition_system.display_eigenfaces()

        elif n == 9:
            face_recognition_system.recognize_faces()

        elif n == 10:
            print("Thank you for using our program!")
            print("-"*80)
            break

        else:
            print("Invalid input\n")