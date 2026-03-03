import os

# Disable Tensorflow roundoff message
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from mtcnn.utils.images import load_images_batch
from mtcnn.utils.plotting import plot

import matplotlib.pyplot as plt
import time
import torch

# Basic Implementation from https://mtcnn.readthedocs.io/en/latest/usage/

if __name__ == "__main__":

    # Start a clock to test optimization
    start = time.time()

    gpu_yn = torch.cuda.is_available() # Check for CUDA GPU
    if gpu_yn == True:
        mtcnn = MTCNN(device="GPU:0")  # Use GPU if computer has
    else: mtcnn = MTCNN(device="CPU:0")   # Use CPU if not. 

    
     # Load image, initialize MTCNN on your CPU. Can use NPU etc
    image = load_image("data/Aiden2.jpg") 

    # Create a bounding box with confidence intervals around the face
    result = mtcnn.detect_faces(image)

    end = time.time()

    print(f"Detection took {end - start:.3f} seconds")
    print(f"Speed: {1/(end-start):.1f} FPS")
    # Display the box
    plt.imshow(plot(image, result))
    plt.show()

