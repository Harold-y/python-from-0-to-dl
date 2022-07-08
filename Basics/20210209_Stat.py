import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('McDonald\'s', 'Burger King', 'Wendy\'s', 'KFC', 'Pizza Hut', 'Taco Bell', 'Starbucks')
y_pos = np.arange(len(objects))
performance = [41, 11.3, 9.4, 8.2, 8, 4.3, 4.1]

plt.bar(y_pos, performance, align='center', alpha=1.0)
plt.xticks(y_pos, objects, fontsize=7)
plt.ylabel('Billions of Dollars')
plt.title('Yearly Produced Value')
plt.show()
