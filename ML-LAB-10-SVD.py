print("RAMYA R - 24BAD096")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
data = pd.read_csv(r"D:\ratings.csv")
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = user_item_matrix.values/5.0
user_mean = np.sum(matrix, axis=1) / np.count_nonzero(matrix, axis=1)
user_mean = user_mean.reshape(-1,1)
matrix_centered = matrix - user_mean
U, sigma, Vt = np.linalg.svd(matrix_centered, full_matrices=False)
k = 20
sigma_k = np.diag(sigma[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]
reconstructed = np.dot(np.dot(U_k, sigma_k), Vt_k) + user_mean
reconstructed = np.clip(reconstructed, 0, 1)
predicted = pd.DataFrame(reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)
mask = matrix > 0
actual = matrix[mask]
pred = reconstructed[mask]
rmse = np.sqrt(mean_squared_error(actual, pred))
mae = mean_absolute_error(actual, pred)
print("RMSE:", rmse)
print("MAE:", mae)
plt.scatter(actual[:200], pred[:200])
plt.plot([0, 1], [0, 1], color='red')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()
user_id = user_item_matrix.index[0]
user_ratings = predicted.loc[user_id]
top_movies = user_ratings.sort_values(ascending=False).head(10)
errors = []
k_values = [5,10,15,20]
for k in k_values:
    sigma_k = np.diag(sigma[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    recon = np.dot(np.dot(U_k, sigma_k), Vt_k) + user_mean
    pred_k = recon[mask]
    errors.append(np.sqrt(mean_squared_error(actual, pred_k)))
sample_users = user_item_matrix.iloc[:20, :20]
sample_predicted = predicted.iloc[:20, :20]
plt.figure(figsize=(10,6))
sns.heatmap(sample_users, cmap='viridis')
plt.title("Original User-Item Matrix (Sample)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(sample_predicted, cmap='viridis',vmin = 0, vmax = 1)
plt.title("Reconstructed Matrix using SVD (Sample)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()
plt.plot(k_values, errors, marker='o')
plt.xlabel("Number of Latent Factors (k)")
plt.ylabel("RMSE")
plt.title("Error vs Number of Latent Factors")
plt.show()
plt.figure()
plt.bar(range(len(top_movies)), top_movies.values)
plt.xticks(range(len(top_movies)), top_movies.index, rotation=45)
plt.xlabel("Movie ID")
plt.ylabel("Predicted Rating")
plt.title("Top Recommended Movies")
plt.show()
