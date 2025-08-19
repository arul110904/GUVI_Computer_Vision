import cv2
import matplotlib.pyplot as plt

img = cv2.imread("download.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,100,300)

plt.imshow(cv2.cvtColor(edges,cv2.COLOR_BGR2RGB))

plt.axis("off")

plt.show()