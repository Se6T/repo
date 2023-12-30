import os
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE


class CustomDownprojection:
    def __init__(self, data, dim=2, perplexity=100, random_state=42):
        drop_cols = [
            col for col in data.columns
            if re.search(r'returns_\d+d$', col)
            or col.endswith('Open')
            or col.endswith('High')
            or col.endswith('Low')
        ]
        self.dim = dim
        self.perplexity = perplexity
        self.random_state = random_state

        return_cols = [col for col in data.columns if re.search(r'returns_\d+d$', col)]
        self.data = data.drop(columns=drop_cols)
        self.returns = data[return_cols]
        self.idx = [i / len(self.data) for i in range(len(self.data))]

        if os.path.exists("models/downprojection") and os.listdir("models/downprojection"):
            self._load_downprojection()
        else:
            self._perform_downprojection()

    def _perform_downprojection(self):
        # Perform PCA
        self.pca = PCA(self.dim)
        self.principal_components = self.pca.fit_transform(self.data)
        
        # Perform t-SNE
        self.tsne = TSNE(n_components=self.dim, perplexity=self.perplexity, random_state=self.random_state)
        self.tsne_components = self.tsne.fit_transform(self.data)
    
        # Perform Kernel PCA
        self.rbf_kernel_pca = KernelPCA(n_components=self.dim, kernel='rbf')
        self.rbf_kpca = self.rbf_kernel_pca.fit_transform(self.data)
        
        self.cos_kernel_pca = KernelPCA(n_components=self.dim, kernel='cosine')
        self.cos_kpca = self.cos_kernel_pca.fit_transform(self.data)

        self._store_downprojection()
    
    def _store_downprojection(self):
        # Create the directory if it doesn't exist
        if not os.path.exists("models/downprojection"):
            os.makedirs("models/downprojection")

        # Store each downprojection
        np.save("models/downprojection/principal_components.npy", self.principal_components)
        np.save("models/downprojection/tsne_components.npy", self.tsne_components)
        np.save("models/downprojection/rbf_kpca.npy", self.rbf_kpca)
        np.save("models/downprojection/cos_kpca.npy", self.cos_kpca)
    
    def _load_downprojection(self):
        # Load each downprojection
        self.principal_components = np.load("models/downprojection/principal_components.npy")
        self.tsne_components = np.load("models/downprojection/tsne_components.npy")
        self.rbf_kpca = np.load("models/downprojection/rbf_kpca.npy")
        self.cos_kpca = np.load("models/downprojection/cos_kpca.npy")
            
    def get_close_points(self, method='cos_kpca', num_last_points=20, k=50):
        if method == 'pca':
            downprojection = self.principal_components
        elif method == 'tsne':
            downprojection = self.tsne_components
        elif method == 'rbf_kpca':
            downprojection = self.rbf_kpca
        elif method == 'cos_kpca':
            downprojection = self.cos_kpca
        else:
            raise ValueError(f"Method parameter was {method}, but needs to be 'pca', 'tsne', or some 'kpca'.")
        
        # Get the last 'num_last_points' data points
        last_data_points = downprojection[-num_last_points:]

         # Calculate the average of the last points
        average_last_points = np.mean(last_data_points, axis=0)

        # Calculate Euclidean distances between all data points except the most recent data points and the average
        distances = np.linalg.norm(downprojection[:-180] - average_last_points, axis=1)
        # Normalize distances
        distances_min = np.min(distances)
        distances_max = np.max(distances)
        normalized_distances = (distances - distances_min) / (distances_max - distances_min)
        # Find the indices of the 'k' closest points to the weighted average
        closest_points_indices = np.argpartition(normalized_distances, k)[:k]
        closest_distances = normalized_distances[closest_points_indices]
        
        dates = [self.data.index[index] for index in closest_points_indices]
        dates_df = pd.DataFrame({
            "distance": closest_distances,
        }, index = dates)
        for col in self.returns.columns:
            dates_df[col] = self.returns[col].loc[dates]

        return dates_df
    
    def get_closest_times(self):
        combined_df = pd.DataFrame()
        
        for method in ['tsne', 'pca', 'rbf_kpca', 'cos_kpca']:
            df = self.get_close_points(method=method)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_df = combined_df.sort_values(by="distance", ascending=True)

        return combined_df
    
    def plot_closest_times(self):
        close_times = self.get_closest_times()
        index = self.data.index
        
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(12, 1))

        # Plot the timeline as a line
        ax.plot(index, [0] * len(index), label='Timeline', linestyle='-', linewidth=2)

        # Plot the highlighted points with sizes
        ax.scatter(close_times['date'], [0] * len(close_times), s=close_times['counts']*100, alpha=0.15, label='Counts', color='red')

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([])

        # Show the plot
        plt.title("Timeline with Similar Macro-Economic Conditions to Today")
        plt.show()
