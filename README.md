py: archivos de python
  filter_unimorph para extraer categorías específicas de archivos de UniMorph.
  main_triplets/biplets.py: código principal. Triplets es para analizar archivos de tripletes, biplets para archivos de parejas (base+construct/verbo+flexión)
  
  datasets: # derinet debería estar en esta carpeta
    spa:
      spa.txt: archivo de unimorph
      filtered: solo los tiempos presente, pasado imperfecto y futuro.
      _small: dataset pequeño para testear el código.
      50_triplets.csv: archivo de tripletes que hice para el trabajo de clase de NLP.
      
    pol:
      pol.txt: archivo de unimorph
      filtered:

  embeddings: modelos de vectores (W2V y FT) separados por idioma. No están subidos porque son bastante pesados.
  results: archivos csv con los resultados del código. Aparecen todas las palabras.
