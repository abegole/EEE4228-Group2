from mtcnn import MTCNN

from mtcnn.utils.images import load_image
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt

image = load_image("../data/Aiden1.jpg")

mtcnn = MTCNN(device="CPU:0")   

result = mtcnn.detect_faces(image)