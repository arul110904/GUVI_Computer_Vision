import cv2
import matplotlib.pyplot as plt

img = cv2.imread("download.jpg")
circle = cv2.circle(img,(160,120),30,(0,0,255),8)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()