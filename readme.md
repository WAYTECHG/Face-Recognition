# Face Recognition System

## Overview

This project is a **Face Recognition System** implemented in Python. It offers three different face recognition methods:

1. **EigenFace Recognition (Scratch Implementation)**

   A fully custom implementation of the Eigenface method. Useful for understanding the underlying mathematics and PCA from scratch.
2. **Improved EigenFace Recognition (scikit-learn)**

   Uses scikit-learn’s PCA to perform Eigenface recognition. Includes support for  **unknown person detection** .
3. **FisherFace Recognition (scikit-learn)**

   Combines PCA + LDA (Fisherfaces) for better performance, especially in distinguishing unknown persons. Ideal for real-world scenarios with multiple classes.

The system provides **real-time face recognition** using a webcam, and includes tools for capturing training images, model training, and performance analysis.

---

## Features

* Real-time face recognition using a webcam.
* Capture and store training images for each person.
* Brightness augmentation for robust recognition.
* Support for unknown person detection.
* Performance analysis with accuracy metrics and prediction time.
* PCA and LDA models saved for re-use.

---

## Requirements

* Python 3.9+
* OpenCV
* NumPy
* scikit-learn
* matplotlib
* joblib

You can install the dependencies using:

<pre class="overflow-visible!" data-start="1417" data-end="1491"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install opencv-python numpy scikit-learn matplotlib joblib
</span></span></code></div></div></pre>

---

## File Structure

<pre class="overflow-visible!" data-start="1517" data-end="2219"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>main_system.py             </span><span># Main menu to select face recognition system</span><span>
eigenface_scratch.py       </span><span># EigenFace implementation from scratch</span><span>
eigenface_sklearn.py       </span><span># EigenFace implementation using scikit-learn PCA</span><span>
fisherface_sklearn.py      </span><span># FisherFace implementation using PCA + LDA</span><span>
training_images/           </span><span># Folder where captured images are stored</span><span>
centroids_v2.npy           </span><span># Saved centroids for eigenface_sklearn</span><span>
centroids_v3.npy           </span><span># Saved centroids for fisherface_sklearn</span><span>
pca_x_train_eigen.joblib   </span><span># Saved PCA model for eigenface_sklearn</span><span>
pca_x_train_fisher.joblib  </span><span># Saved PCA model for fisherface_sklearn</span><span>
lda.joblib                 </span><span># Saved LDA model for fisherface_sklearn</span><span>
</span></span></code></div></div></pre>

---

## Usage

1. Run the main menu:

<pre class="overflow-visible!" data-start="2259" data-end="2292"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python main_system.py
</span></span></code></div></div></pre>

2. You will see the following options:

<pre class="overflow-visible!" data-start="2334" data-end="2480"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>1.</span><span> EigenFace Recognition </span><span>system</span><span></span><span>in</span><span> Scratch
</span><span>2.</span><span> Improved EigenFace Recognition </span><span>system</span><span></span><span>using</span><span> sklearn
</span><span>3.</span><span> FisherFace Recognition </span><span>System</span><span>
</span><span>4.</span><span></span><span>Exit</span><span>
</span></span></code></div></div></pre>

3. Select the desired system by entering the corresponding number:

* **Option 1:** Launches `eigenface_scratch.py`
* **Option 2:** Launches `eigenface_sklearn.py`
* **Option 3:** Launches `fisherface_sklearn.py`
* **Option 4:** Exit the program

4. Follow the on-screen instructions to:
   * Capture training images
   * Analyze components
   * Train the recognition system
   * Perform real-time face recognition

---

## Notes

* Make sure your webcam is connected.
* Training images should be captured in good lighting for better accuracy.
* The first run will require capturing faces for all users you want to recognize.
* Use `q` to quit real-time recognition mode.
