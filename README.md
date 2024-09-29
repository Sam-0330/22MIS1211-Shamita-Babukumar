# SOM-on-country-data
# Countries Clustering using Self-Organizing Maps (SOM) and KMeans

This project aims to cluster countries based on various socio-economic indicators using Self-Organizing Maps (SOM) and KMeans clustering. The clustering results are visualized in a 2D grid, providing insights into the similarities between countries.

## Dataset

The dataset used for this project can be found at the following link: [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data).

The dataset includes socio-economic and demographic indicators for approximately 167 countries, such as:
- GDP per capita
- Life expectancy
- Population
- Literacy rates
- Healthcare spending
- Child mortage per capita
- Exports and imports per capita
- Income
- Inflation
- Fertility rate



## Requirements

### Software Requirements

- **Programming Language**: Python 3.x
- **Libraries**:
  - `pandas` (for data handling)
  - `numpy` (for numerical computations)
  - `matplotlib` (for visualization)
  - `sklearn` (for scaling and clustering)
  - `minisom` (for the Self-Organizing Map)

### No Hardware Requirements

You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn minisom
```
You can run the below code in google collab :

```bash

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from minisom import MiniSom

data = pd.read_csv('/content/Country-data.csv')
countries = data['country'].values
data = data.drop('country', axis=1).to_numpy()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
som = MiniSom(x=10, y=10, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_scaled)
som.train_random(data_scaled, 1000)
win_map = np.array([som.winner(d) for d in data_scaled])
win_map_flat = np.array([x * 10 + y for x, y in win_map])
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(win_map_flat.reshape(-1, 1))
plt.figure(figsize=(8, 8))
markers = ['o', 's', 'D']
colors = ['r', 'b', 'g']
for i, x in enumerate(data_scaled):
    winner = som.winner(x)
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, markers[clusters[i]],
             markerfacecolor='None', markeredgecolor=colors[clusters[i]],
             markersize=12, markeredgewidth=2)
for i, centroid in enumerate(kmeans.cluster_centers_):
    plt.plot(som.winner(centroid)[0] + 0.5, som.winner(centroid)[1] + 0.5, 'X',
             markerfacecolor=colors[i], markeredgewidth=2, markersize=16)
plt.title('Countries - income and expectancy mapping')
plt.grid()
plt.show()
```

This code does not require an executable file!!!

