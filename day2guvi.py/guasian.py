import cv2
import matplotlib.pyplot as plt

img = cv2.imread("download.jpg")
gaussian = cv2.GaussianBlur(img,(5,5),0)
plt.imshow(cv2.cvtColor(gaussian,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()