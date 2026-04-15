SCENARIO 1 – MATRIX FACTORIZATION USING SVD

Dataset (Kaggle – Public)
MovieLens Dataset
Dataset Link: https://www.kaggle.com/datasets/abhikjha/movielens-100k

This project implements a movie recommendation system using Singular Value Decomposition (SVD), a matrix factorization technique that captures hidden relationships between users and movies. The MovieLens dataset is used, which contains User ID, Movie ID, and ratings. First, the dataset is loaded and preprocessed, and a user-item interaction matrix is created where rows represent users and columns represent movies. Missing values are handled, and optionally the matrix is normalized using mean centering to remove user bias. SVD is then applied to decompose the matrix into three components, and only the top k latent factors are selected to reduce dimensionality. The matrix is reconstructed using these reduced components to predict missing ratings. Based on the predicted values, Top-N movie recommendations are generated for each user. The model performance is evaluated using RMSE and MAE, where lower values indicate better accuracy. Further analysis is done to study the impact of different k values, comparing predicted and actual ratings, and understanding overfitting and underfitting. Visualizations such as heatmaps and error vs k graphs help in understanding how dimensionality reduction improves recommendation quality. Overall, SVD provides an efficient and accurate way to build recommendation systems by learning latent user preferences.


SCENARIO 2 – MATRIX FACTORIZATION USING NMF

Dataset (Same / Alternative Dataset)
Same MovieLens Dataset (or any rating dataset)

This project uses Non-negative Matrix Factorization (NMF) to build a movie recommendation system based on latent features. The same MovieLens dataset is used, containing User ID, Movie ID, and ratings. The data is loaded and converted into a user-item matrix, where missing values are handled by filling them with zero or mean values, since NMF requires non-negative inputs. The NMF model factorizes the matrix into two smaller matrices: a user-feature matrix and an item-feature matrix, representing hidden patterns in user preferences and movie characteristics. These matrices are multiplied to reconstruct the original matrix and predict missing ratings. Based on the predicted ratings, Top-N recommendations are generated for users. The performance of the model is evaluated using RMSE, Precision@K, and Recall@K, which measure both accuracy and relevance of recommendations. NMF is especially useful because it produces interpretable features, making it easier to understand user behavior. Compared to SVD, NMF is more interpretable but may be slightly less accurate. Visualizations such as heatmaps and feature-based plots help analyze the results. Overall, NMF is a simple and effective method for recommendation systems when interpretability is important and data is non-negative.

# 24ADI003_24BAD096_RAMYA-R_EX_10
