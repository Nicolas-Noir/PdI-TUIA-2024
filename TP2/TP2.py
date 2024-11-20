import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

monedas = cv2.imread('monedas.jpg')

monedas_copy = monedas.copy()

monedas_rgb = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

monedas_blur = cv2.blur(monedas, (5,5))

monedas_hls = cv2.cvtColor(monedas_blur, cv2.COLOR_BGR2HLS)

h, l, s = cv2.split(monedas_hls)

monedas_hls_s = cv2.blur(s, (7, 7))

_ , monedas_s_binarizado = cv2.threshold(monedas_hls_s, 14, 255, cv2.THRESH_BINARY)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
monedas_clausura = cv2.morphologyEx(monedas_s_binarizado, cv2.MORPH_CLOSE, B)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
monedas_apertura = cv2.morphologyEx(monedas_clausura, cv2.MORPH_OPEN, B)

contornos_objetos, _ = cv2.findContours(monedas_apertura, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

mascara_zeros = np.zeros_like(monedas_apertura, dtype=np.uint8)

contornos_monedas_areas  = []
contornos_monedas_solos = []

for contorno in contornos_objetos:
    area_objetos = cv2.contourArea(contorno)
    perimetro_cuadrado_objetos = cv2.arcLength(contorno, True)**2
    funcion_de_forma = area_objetos/perimetro_cuadrado_objetos
    if funcion_de_forma > 0.062:
        contornos_monedas_areas.append((contorno, area_objetos))
        contornos_monedas_solos.append(contorno)
    else:
        cv2.drawContours(mascara_zeros, [contorno], -1, 255, thickness=cv2.FILLED)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
dado_clausura = cv2.morphologyEx(mascara_zeros, cv2.MORPH_CLOSE, B)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
dado_apertura = cv2.morphologyEx(dado_clausura, cv2.MORPH_OPEN, B)

dados_contornos, _ = cv2.findContours(dado_apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

moneda_copia_dados = monedas.copy()

lista_cantidad_dados_puntos = []

for dado in dados_contornos:
    x, y, w, h = cv2.boundingRect(dado)

    dado_recortado = moneda_copia_dados[y-10:y+h+10, x-10:x+w+10]
    dado_gris = cv2.cvtColor(dado_recortado, cv2.COLOR_BGR2GRAY)

    dado_blur = cv2.GaussianBlur(dado_gris, (9, 9), 2)
    _, dado_binarizado = cv2.threshold(dado_blur, 170,255, cv2.THRESH_BINARY)

    dado_binarizado_invertido = cv2.bitwise_not(dado_binarizado)

    dados_contornos_puntos, _ = cv2.findContours(dado_binarizado_invertido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cantidad_de_puntos_en_dado = 0
    for puntos_y_formas in dados_contornos_puntos:
        area_puntos = cv2.contourArea(puntos_y_formas)
        perimetro_cuadrado_puntos_dados = cv2.arcLength(puntos_y_formas, True)**2
        funcion_de_forma_puntos_dado = area_puntos/perimetro_cuadrado_puntos_dados
        if funcion_de_forma_puntos_dado > 0.062:
            cantidad_de_puntos_en_dado+=1 

    lista_cantidad_dados_puntos.append(cantidad_de_puntos_en_dado)

print(lista_cantidad_dados_puntos)

#--monedas--


monedas_rgb_copia = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

contornos_monedas_ordenado = sorted(contornos_monedas_areas , key=lambda x: x[1])

max_area = contornos_monedas_ordenado[-1][1]
min_area = contornos_monedas_ordenado[0][1]

if min_area / max_area < 0.85:
    monedas_10_50 = 0
    monedas_10_peso = 0
    monedas_50 = 0
    monedas_peso = 0

    rel_50_10_max , rel_50_10_min = 0.61, 0.4
    rel_50_1_max , rel_50_1_min = 0.83, 0.74
    rel_1_10_max , rel_1_10_min = 0.73, 0.62

    for moneda, area in contornos_monedas_areas:
        if rel_50_10_min <= area/max_area <= rel_50_10_max:
            monedas_10_50 += 1
        elif rel_50_1_min <= area/max_area <= rel_50_1_max:
            monedas_peso += 1
        elif rel_1_10_min <= area/max_area <= rel_1_10_max:
            monedas_10_peso += 1
        elif area/max_area >= 0.84:
            if monedas_10_50 >= 1:
                monedas_50 += 1
            elif monedas_10_peso >= 1:
                monedas_peso +=1
            elif monedas_peso >= 1:
                monedas_50 += 1
    if monedas_10_peso == 0:
        monedas_10_peso = monedas_10_50
    elif monedas_10_50 == 0:
        monedas_10_50 = monedas_10_peso
    
    print(monedas_50, monedas_peso, monedas_10_peso)

else:
    monedas_10_peso = 0
    monedas_50 = 0
    monedas_peso = 0
    rel_recuadro_50_max , rel_recuadro_50_min = 0.97, 0.65

    for contorno_moneda in contornos_monedas_solos:
        x, y, w, h = cv2.boundingRect(contorno_moneda)

        moneda_recortada = monedas_rgb_copia[y-10:y+h, x:x+w]

        moneda_recortada_cielab = cv2.cvtColor(moneda_recortada, cv2.COLOR_BGR2LAB)

        lab_moneda, a_moneda, b_moneda = cv2.split(moneda_recortada_cielab)

        _, b_moneda_binarizada = cv2.threshold(b_moneda, 115, 255, cv2.THRESH_BINARY)
        b_moneda_binarizada_invertida = np.bitwise_not(b_moneda_binarizada)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        b_moneda_binarizada_invertida_clausura = cv2.morphologyEx(b_moneda_binarizada_invertida, cv2.MORPH_CLOSE, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        b_moneda_binarizada_invertida_apertura = cv2.morphologyEx(b_moneda_binarizada_invertida_clausura, cv2.MORPH_OPEN, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        b_moneda_clausura_final = cv2.morphologyEx(b_moneda_binarizada_invertida_apertura, cv2.MORPH_CLOSE, B_monedita)

        contorno_moneda_individual, _ = cv2.findContours(b_moneda_clausura_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contorno_moneda_individual) == 0:
            monedas_10_peso += 1
        else:
            area_moneda_50_o_peso = cv2.contourArea(contorno_moneda_individual[0])
            area_recuadro_moneda_50_o_peso = w*h
            if rel_recuadro_50_min < area_moneda_50_o_peso/area_recuadro_moneda_50_o_peso < rel_recuadro_50_max:
                monedas_50 += 1
            else:
                monedas_peso += 1

    print(monedas_50, monedas_peso, monedas_10_peso)


