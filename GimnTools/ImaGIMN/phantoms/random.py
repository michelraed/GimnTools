from scipy.ndimage import gaussian_filter
import numpy as np
from PIL import Image, ImageDraw
from GimnTools.ImaGIMN.phantoms.geometries.figures import create_star_image, create_ellipse_image, create_polygon_image, create_circle_image


def generate_random_image(size,number_of_figures):
  #cria a imagem
  image = np.zeros([size, size], dtype=np.int64)

  for i in range(number_of_figures):
    #gera um seletor aleatorio para o tipo de figura que será criado
    selector = np.random.randint(0,100)

    if selector>=0 and selector<25.0:
      #Estrela é selecionada
      
      num_points = np.random.randint(5,12)
      x = np.random.randint(0,size)  # posição x do centro da estrela
      y = np.random.randint(0,size)  # posição y do centro da estrela
      star_radius = np.random.randint(0,size/3) # raio da estrela
      star = create_star_image(size, num_points, x, y, star_radius,1)
      #bota a estrela dentro de um determinado valor de atividade
      star = star/10
      multiplier = np.random.randint(10,400)
      phantom = star*multiplier

      #gera borramento
      sigma = 3  # Desvio padrão
      phantom = gaussian_filter(phantom, sigma)

      #gera ruido aleatorio
      image += np.random.poisson(phantom)

    elif selector>=25 and selector<50.0:
      #seleciona elipse
      size = size
      x = np.random.randint(0,size)    # posição x do centro da elipse
      y = np.random.randint(0,size)    # posição y do centro da elipse
      width = np.random.randint(1,size/2)    # largura da elipse
      height = np.random.randint(1,size/2)  # altura da elipse
      elipse = create_ellipse_image(size, x, y, width, height,1)
      elipse = elipse/10
      multiplier = np.random.randint(10,400)
      phantom = elipse*multiplier

      #gera borramento
      sigma = 3  # Desvio padrão
      phantom = gaussian_filter(phantom, sigma)

      #gera ruido aleatorio
      image += np.random.poisson(phantom)
    elif selector>=50 and selector<75.0:
      #seleciona poligono
      size = size
      num_sides = np.random.randint(3,8)  # número de lados do polígono
      x = np.random.randint(0,size)   # posição x do centro do polígono
      y = np.random.randint(0,size)   # posição y do centro do polígono
      radius = np.random.randint(0,size/3)   # raio do polígono
      pol = create_polygon_image(size, num_sides, x, y, radius,1)
      pol = pol/10
      multiplier = np.random.randint(10,400)
      phantom = pol*multiplier
      #gera borramento
      sigma = 3  # Desvio padrão
      phantom = gaussian_filter(phantom, sigma)

      #gera ruido aleatorio
      image += np.random.poisson(phantom)
    elif selector>=75 and selector<=100:
      #seleciona circulo
      size = size
      x = np.random.randint(0,size)   # posição x do centro do círculo
      y = np.random.randint(0,size)   # posição y do centro do círculo
      radius = np.random.randint(0,size/3)   # raio do círculo
      circ = create_circle_image(size, x, y, radius,1)
      circ = circ/10
      multiplier = np.random.randint(10,400)
      phantom =circ*multiplier

      #gera borramento
      sigma = 1  # Desvio padrão
      phantom = gaussian_filter(phantom, sigma)

      #gera ruido aleatorio
      image += np.random.poisson(phantom)

    else:
        print("something went wront")

  return image