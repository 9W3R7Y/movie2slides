import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

cap = cv2.VideoCapture('videos/講義動画.mp4')
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

temp = np.zeros([h, w, 3])
step = 10
coeff = 0.3
RMS = []

print("loading frames")
for i in tqdm(range(n)):
    if i % step == 0:
        ret, frame = cap.read()
        RMS.append(np.sqrt(np.sum(np.power(temp-frame, 2))/w*h))
        temp = frame
    else:
        ret = cap.grab()

mean_RMS = np.mean(RMS)
sum_RMS = 0

RMS_integral = []

for i in range(n//step):
    if RMS[i] > mean_RMS*coeff:
        sum_RMS += RMS[i]
        RMS_integral.append(0)

    else:
        RMS_integral.append(sum_RMS)
        sum_RMS = 0

peaks = np.squeeze(np.where(RMS_integral > mean_RMS*coeff))
n_peaks = len(peaks)

for i in range(n_peaks):
    t = (peaks[i]-1)*step+1
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ret, frame = cap.read()
    cv2.imwrite(str(i).zfill(2)+".png", frame)

plt.plot(RMS[2:], label="RMS")
plt.plot(RMS_integral[2:], label="Piecewise Integral")
plt.legend()
plt.show()
