import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

class DimensionalityReducer:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def apply_svd(self, X, n_components=100, random_state=42):
        """Применение SVD (LSA) для снижения размерности"""
        print(f"Применение SVD для снижения до {n_components} компонент...")
        
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=random_state
        )
        
        X_reduced = svd.fit_transform(X)
        
        # Сохранение модели и результатов
        self.models['svd'] = svd
        self.results['svd'] = {
            'X_reduced': X_reduced,
            'explained_variance': svd.explained_variance_ratio_,
            'total_variance': np.sum(svd.explained_variance_ratio_)
        }
        
        print(f"Сохраненная дисперсия: {np.sum(svd.explained_variance_ratio_):.4f}")
        
        return X_reduced, svd
    
    def find_optimal_components(self, X, max_components=200):
        """Поиск оптимального числа компонент"""
        print("Поиск оптимального числа компонент...")
        
        components_range = range(10, min(max_components, X.shape[1]-1), 10)
        explained_variances = []
        
        for n_comp in components_range:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd.fit(X)
            explained_variances.append(np.sum(svd.explained_variance_ratio_))
        
        # Построение графика локтя
        plt.figure(figsize=(10, 6))
        plt.plot(components_range, explained_variances, 'bo-')
        plt.xlabel('Количество компонент')
        plt.ylabel('Объясненная дисперсия')
        plt.title('График локтя для выбора числа компонент SVD')
        plt.grid(True)
        plt.savefig('svd_elbow_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Нахождение точки, где прирост дисперсии замедляется
        differences = np.diff(explained_variances)
        optimal_idx = np.argmax(differences < 0.01) + 1 if any(differences < 0.01) else len(components_range) - 1
        
        optimal_components = components_range[optimal_idx]
        print(f"Рекомендуемое число компонент: {optimal_components}")
        
        return optimal_components, explained_variances
    
    def visualize_components(self, X_reduced, labels, method='svd', n_components=2):
        """Визуализация компонент"""
        if n_components == 2:
            self._visualize_2d(X_reduced, labels, method)
        elif n_components == 3:
            self._visualize_3d(X_reduced, labels, method)
    
    def _visualize_2d(self, X_reduced, labels, method):
        """2D визуализация"""
        plt.figure(figsize=(12, 8))
        
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       c=[colors[i]], label=label, alpha=0.7)
        
        plt.xlabel('Компонента 1')
        plt.ylabel('Компонента 2')
        plt.title(f'2D визуализация {method.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{method}_2d_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_3d(self, X_reduced, labels, method):
        """3D визуализация"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                      c=[colors[i]], label=label, alpha=0.7)
        
        ax.set_xlabel('Компонента 1')
        ax.set_ylabel('Компонента 2')
        ax.set_zlabel('Компонента 3')
        ax.set_title(f'3D визуализация {method.upper()}')
        ax.legend()
        plt.savefig(f'{method}_3d_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def apply_tsne(self, X, n_components=2, perplexity=30, random_state=42):
        """Применение t-SNE для визуализации"""
        print("Применение t-SNE...")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000
        )
        
        X_tsne = tsne.fit_transform(X)
        self.results['tsne'] = {'X_reduced': X_tsne}
        
        return X_tsne
    
    def apply_umap(self, X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        """Применение UMAP для визуализации"""
        print("Применение UMAP...")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        
        X_umap = reducer.fit_transform(X)
        self.results['umap'] = {'X_reduced': X_umap}
        
        return X_umap
    
    def interpret_components(self, svd_model, feature_names, n_top_words=10):
        """Интерпретация компонент через ключевые слова"""
        components = svd_model.components_
        
        print("\nИНТЕРПРЕТАЦИЯ КОМПОНЕНТ SVD")
        print("="*50)
        
        for i, component in enumerate(components):
            top_indices = component.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            top_scores = [component[idx] for idx in top_indices]
            
            print(f"\nКомпонента {i+1}:")
            for word, score in zip(top_words, top_scores):
                print(f"  {word}: {score:.4f}")
            
            if i >= 4:  # Показываем только первые 5 компонент
                print("...")
                break

# Пример использования
def demonstrate_dimensionality_reduction():
    """Демонстрация работы модуля снижения размерности"""
    from classical_vectorizers_2 import ClassicalVectorizers
    
    # Загрузка и векторизация данных
    vectorizer = ClassicalVectorizers()
    corpus, labels = vectorizer.load_corpus()
    X_tfidf = vectorizer.tfidf_vectorizer(ngram_range=(1, 2))
    
    # Снижение размерности
    reducer = DimensionalityReducer()
    
    # Поиск оптимального числа компонент
    optimal_comp, variances = reducer.find_optimal_components(X_tfidf)
    
    # Применение SVD
    X_svd, svd_model = reducer.apply_svd(X_tfidf, n_components=optimal_comp)
    
    # Визуализация
    reducer.visualize_components(X_svd[:, :2], labels, 'svd', 2)
    
    # Интерпретация компонент
    feature_names = vectorizer.vocabularies['tfidf']
    reducer.interpret_components(svd_model, feature_names)
    
    # Альтернативные методы визуализации
    X_tsne = reducer.apply_tsne(X_svd)
    reducer.visualize_components(X_tsne, labels, 'tsne', 2)
    
    X_umap = reducer.apply_umap(X_svd)
    reducer.visualize_components(X_umap, labels, 'umap', 2)

if __name__ == "__main__":
    demonstrate_dimensionality_reduction()