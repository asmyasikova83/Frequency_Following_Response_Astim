import numpy as np
from pathlib import Path

data = [ 40. ,  42.5  ,45.   ,
         47.5 , 50.  , 52.5 , 55. ,  57.5 , 60.  , 62.5  ,65.,   67.5,
  70. ,  72.5 , 75.  , 77.5 , 80.  , 82.5 , 85. ,  87.5 , 90. ,  92.5 , 95. ,  97.5 ]

index = next(i for i, value in enumerate(data) if value > 70)
print(f"Индекс первого элемента > 70: {index}")  # Вывод: 13


F = [- 10,  -5,  0,  10]
F = np.array(F)
print(1 < F < 1000)
