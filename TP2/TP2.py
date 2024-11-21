import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def detectar_monedas_y_dados(monedas):
    """
    Función que busca y diferencia entre los tipos de monedas de 1 peso, 50 centavos y 10 centavos, devolviendo la cantidad de cada uno

    También, busca y diferencia la cantidad de puntos en la cara superior de distintos dados, devolviendo la cantidad de puntos de cada dado
    ----------------------------------------------------------------------------------------------------------------------------------------
    Parámetros

    imagen: una imagen que preferentemente contenga monedas de los valores anteriormente dichos y dados  
    """

    monedas_copy = monedas.copy()

    # Se suaviza la imagen para obtener los detalles de los objetos de manera más clara
    monedas_blur = cv2.blur(monedas, (5,5))

    # Se convierte de BGR a HLS para trabajar con el canal de la saturación
    monedas_hls = cv2.cvtColor(monedas_blur, cv2.COLOR_BGR2HLS)

    _, _, s = cv2.split(monedas_hls)

    # Se vuelve a suavizar la imagen, pero ahora sobre la saturación
    monedas_hls_s = cv2.blur(s, (7, 7))
    # Se binariza para obtener los objetos y poder trabajar con ellos
    _ , monedas_s_binarizado = cv2.threshold(monedas_hls_s, 14, 255, cv2.THRESH_BINARY)

    # Se realiza una clausura para poder cerrar los huecos de los diferentes objetos
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
    monedas_clausura = cv2.morphologyEx(monedas_s_binarizado, cv2.MORPH_CLOSE, B)

    # Se realiza apertura para poder eliminar ruido que quede en la imagen y, además, poder redondear los objetos para, así, poder obtener sus formas de manera más clara
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    monedas_apertura = cv2.morphologyEx(monedas_clausura, cv2.MORPH_OPEN, B)

    # Se buscan los contornos
    contornos_objetos, _ = cv2.findContours(monedas_apertura, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Se crea una imagen para así luego poder dibujar los objetos sobrantes
    mascara_zeros = np.zeros_like(monedas_apertura, dtype=np.uint8)

    contornos_monedas_areas  = []
    contornos_monedas_solos = []

    # Se itera sobre los contornos de cada objeto, obteniendo su factor de forma
    # Se busca poder diferenciar las monedas, que son circulares, de los dados, que son objetos con formas cercanas a cuadrados
    # Guardando el área y contorno del dado
    # Por otra parte, los objetos que no sean circulares, es decir, los dados, se dibujan una máscara y se rellenan para su posterior análisis.
    for contorno in contornos_objetos:
        area_objetos = cv2.contourArea(contorno)
        perimetro_cuadrado_objetos = cv2.arcLength(contorno, True)**2
        factor_de_forma = area_objetos/perimetro_cuadrado_objetos
        if factor_de_forma > 0.062:
            contornos_monedas_areas.append((contorno, area_objetos))
            contornos_monedas_solos.append(contorno)
        else:
            cv2.drawContours(mascara_zeros, [contorno], -1, 255, thickness=cv2.FILLED)

    # Se aplica una clausura con un kernel grande, para rellenar los posibles huecos entre los dados, generados por los puntos en los mismos
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
    dado_clausura = cv2.morphologyEx(mascara_zeros, cv2.MORPH_CLOSE, B)

    # Se aplica la apertura para redondear las esquinas y picos de los dados
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    dado_apertura = cv2.morphologyEx(dado_clausura, cv2.MORPH_OPEN, B)

    # Se buscan los contornos
    dados_contornos, _ = cv2.findContours(dado_apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    moneda_copia_dados = monedas.copy()

    lista_cantidad_dados_puntos = []

    # Se itera para cada dado
    for dado in dados_contornos:
        x, y, w, h = cv2.boundingRect(dado)

        # Se recorta la de la imagen original, el área que fue recortada usando boundingbox, agregando un pequeño margen de error
        dado_recortado = moneda_copia_dados[y-10:y+h+10, x-10:x+w+10]

        dado_gris = cv2.cvtColor(dado_recortado, cv2.COLOR_BGR2GRAY)

        # Con la imagen pasada a escala de grises, se busca suavizarla y binarizarla para así poder obtener los puntos dentro de ella
        dado_blur = cv2.GaussianBlur(dado_gris, (9, 9), 2)
       
        _, dado_binarizado = cv2.threshold(dado_blur, 170,255, cv2.THRESH_BINARY)
        
        # Se invierte la imagen, ya que para buscar los contornos de los dados, necesitamos que estos puntos sean blancos
        dado_binarizado_invertido = cv2.bitwise_not(dado_binarizado)
        dados_contornos_puntos, _ = cv2.findContours(dado_binarizado_invertido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Se itera sobre cada punto o forma que fue encontrada en el dado, ya que al invertir la imagen, el fondo también es encontrado por find countorns
        # Se buscan las formas que sean circulares, al igual que se realizó anteriormente para las monedas, y se suma 1 por cada punto encontrado
        cantidad_de_puntos_en_dado = 0
        for puntos_y_formas in dados_contornos_puntos:
            area_puntos = cv2.contourArea(puntos_y_formas)
            perimetro_cuadrado_puntos_dados = cv2.arcLength(puntos_y_formas, True)**2
            factor_de_forma_puntos_dado = area_puntos/perimetro_cuadrado_puntos_dados
            if factor_de_forma_puntos_dado > 0.062:
                cantidad_de_puntos_en_dado+=1 

        lista_cantidad_dados_puntos.append(cantidad_de_puntos_en_dado)

    #-------------------------------------------------------------------------------------------------------------------------------------

    # Proceso para encontrar y diferenciar las monedas
    

    monedas_rgb_copia = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

    # Se ordena el área de las monedas de forma ascendente para facilitar la diferenciación entre ellas 
    contornos_monedas_ordenado = sorted(contornos_monedas_areas , key=lambda x: x[1])

    max_area = contornos_monedas_ordenado[-1][1]
    min_area = contornos_monedas_ordenado[0][1]

    # En este caso, se busca saber si la moneda con menor área es diferente a la de mayor
    # Si esto es verdad, estamos en el caso donde puede llegar a haber 2 o 3 tipos de monedas en la imagen 
    # ya que si el área entre la menor y la mayor es mayor a 0.85, esto quiere decir que, dentro del cierto margen, son el mismo tipo de moneda
    if min_area / max_area < 0.85:
        monedas_10_50 = 0
        monedas_10_peso = 0
        monedas_50 = 0
        monedas_peso = 0

        # Rango de relaciones de área entre monedas
        rel_50_10_max , rel_50_10_min = 0.61, 0.4
        rel_50_1_max , rel_50_1_min = 0.83, 0.74
        rel_1_10_max , rel_1_10_min = 0.73, 0.62

        # Se clasifican las monedas según su relación de área con la moneda más grande, siendo estas relaciones declaradas anteriormente
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

    # En el caso de que tengamos 1 solo tipo de moneda
    else:
        monedas_10_peso = 0
        monedas_50 = 0
        monedas_peso = 0
        # Rango de proporciones de área de las monedas de 50 centavos
        rel_recuadro_50_max , rel_recuadro_50_min = 0.97, 0.65

        # Se trabaja sobre la primera moneda, ya que llegado a este punto, todas las monedas son iguales
        contorno_moneda = contornos_monedas_solos[0]
        x, y, w, h = cv2.boundingRect(contorno_moneda)

        # Se recorta la imagen y se convierte en espacio CIElab
        moneda_recortada = monedas_rgb_copia[y-10:y+h, x:x+w]
        moneda_recortada_cielab = cv2.cvtColor(moneda_recortada, cv2.COLOR_BGR2LAB)

        # Se separa al canal b, ya que para CIElab el canal b, no está tan afectado por la iluminación y específicamente el b se centra en los colores amarillos
        _, _, b_moneda = cv2.split(moneda_recortada_cielab)
        _, b_moneda_binarizada = cv2.threshold(b_moneda, 115, 255, cv2.THRESH_BINARY)

        b_moneda_binarizada_invertida = np.bitwise_not(b_moneda_binarizada)

        # Se aplica la clausura, apertura y clausura nuevamente para redondear y sacar ruido ante cualquier tipo de problema y deformación de las monedas
        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        b_moneda_binarizada_invertida_clausura = cv2.morphologyEx(b_moneda_binarizada_invertida, cv2.MORPH_CLOSE, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        b_moneda_binarizada_invertida_apertura = cv2.morphologyEx(b_moneda_binarizada_invertida_clausura, cv2.MORPH_OPEN, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        b_moneda_clausura_final = cv2.morphologyEx(b_moneda_binarizada_invertida_apertura, cv2.MORPH_CLOSE, B_monedita)

        contorno_moneda_individual, _ = cv2.findContours(b_moneda_clausura_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Si el contorno es 0, eso quiere decir que la moneda no tenía colores amarillos o dorados
        # Por lo tanto, la imagen resultante va a ser totalmente negra, esto quiere decir que es la moneda de 10 centavos
        # ya que find_countorns no es capaz de encontrar ningún valor
        if len(contorno_moneda_individual) == 0:
            monedas_10_peso = len(contornos_monedas_solos)
        else:
            # Para las otras monedas se busca el área resultante del fondo de las monedas de 50 y 1 peso 
            # Resultando en que si el área está dentro de la relación para la moneda de 50
            # Eso quiere decir que es la moneda de un 50
            # En contra parte, para la moneda de 1 peso, su borde dorado reduce el área de la moneda, ampliando por contra parte el área del fondo
            area_moneda_50_o_peso = cv2.contourArea(contorno_moneda_individual[0])
            area_recuadro_moneda_50_o_peso = w*h
            if rel_recuadro_50_min < area_moneda_50_o_peso/area_recuadro_moneda_50_o_peso < rel_recuadro_50_max:
                monedas_50 = len(contornos_monedas_solos)
            else:
                monedas_peso = len(contornos_monedas_solos)

    return print(
        f'Hay "{monedas_50}" monedas de 50 centavos, "{monedas_peso}" monedas de 1 peso y "{monedas_10_peso}" monedas de 10 centavos\n'
        f'Hay "{len(lista_cantidad_dados_puntos)}" dados y sus respectivos valores son:\n' + "\n".join(f"  - En el dado número {i + 1}, tiene {puntos} puntos" for i, puntos in enumerate(lista_cantidad_dados_puntos)))


monedas = cv2.imread('monedas.jpg')

detectar_monedas_y_dados(monedas)