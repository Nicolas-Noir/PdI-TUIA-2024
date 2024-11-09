import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

monedas = cv2.imread('monedas.jpg')

monedas_copy = monedas.copy()

monedas_rgb = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

monedas_blur = cv2.blur(monedas, (5,5))

img_hls = cv2.cvtColor(monedas_blur, cv2.COLOR_BGR2HLS)

h, l, s = cv2.split(img_hls)

l_smooth = cv2.blur(l, (15, 15))
s_smooth = cv2.blur(s, (15, 15))
h_smooth = cv2.blur(h, (15, 15))

img_smooth_l = cv2.cvtColor(cv2.merge((h, l_smooth, s)), cv2.COLOR_HLS2RGB)

retval, bin_s = cv2.threshold(s_smooth, 14, 255, cv2.THRESH_BINARY)

retval, bin_h = cv2.threshold(h, 60, 255, cv2.THRESH_BINARY)

plt.imshow(bin_s, cmap='gray'),plt.show()

B = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
Aclau = cv2.morphologyEx(bin_s, cv2.MORPH_CLOSE, B)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
Aop = cv2.morphologyEx(Aclau, cv2.MORPH_OPEN, B)


kernel = np.ones((3, 3), np.uint8)
apertura = cv2.morphologyEx(Aop, cv2.MORPH_OPEN, kernel)

plt.imshow(apertura, cmap='gray'),plt.show()

contours, hierarchy = cv2.findContours(Aop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contornos_circulo = []

for contour in contours:
    a = cv2.contourArea(contour)
    p_c = cv2.arcLength(contour, True)**2
    fdf = a/p_c
    if fdf > 0.062:
        print(a)
        contornos_circulo.append(contour)


cv2.drawContours(monedas_rgb, contornos_circulo,-1, (0, 255, 0), 2)
plt.imshow(monedas_rgb),plt.show()


moneditas = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)
moneadas_10=[]
monedas_50=[]
monedas_peso=[]

for monedas in contornos_circulo:
    area = cv2.contourArea(monedas)
    if area < 70000:
        moneadas_10.append(monedas)
    elif area > 95000:
        monedas_50.append(monedas)
    else:
        monedas_peso.append(monedas)



cv2.drawContours(moneditas, monedas_50,-1, (0, 255, 0), 2)
plt.imshow(moneditas),plt.show()



x, y, w, h = cv2.boundingRect(monedas_peso[0])

imagen_recortada = moneditas[y:y+h, x:x+w]

plt.imshow(imagen_recortada), plt.show()

hsv = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2HSV)

plateado_bajo = np.array([0, 0, 80])
plateado_alto = np.array([180, 50, 200])

mascara_plateado = cv2.inRange(hsv, plateado_bajo, plateado_alto)

resultado_plateado = cv2.bitwise_and(imagen_recortada, imagen_recortada, mask=mascara_plateado)

plt.imshow(cv2.cvtColor(resultado_plateado, cv2.COLOR_BGR2RGB)), plt.show()
