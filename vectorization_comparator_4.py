import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from classical_vectorizers_2 import ClassicalVectorizers

class VectorizationComparator:
    def __init__(self):
        self.results = {}
        self.comparison_df = pd.DataFrame()
    
    def load_data(self):
        """Загрузка данных для сравнения"""
        vectorizer = ClassicalVectorizers()
        self.corpus, self.labels = vectorizer.load_corpus()
        self.vectorizer = vectorizer
        return self.corpus, self.labels
    
    def evaluate_semantic_coherence(self, X, method_name):
        """Оценка семантической согласованности"""
        print(f"Оценка семантической согласованности для {method_name}...")
        
        # Выбираем документы одной категории для оценки сходства
        unique_categories = list(set(self.labels))
        intra_similarities = []
        
        for category in unique_categories:
            # Индексы документов этой категории
            category_indices = [i for i, label in enumerate(self.labels) if label == category]
            
            if len(category_indices) > 1:
                # Выбираем случайные пары документов из одной категории
                np.random.shuffle(category_indices)
                pairs = [(category_indices[i], category_indices[i+1]) 
                        for i in range(0, len(category_indices)-1, 2)]
                
                for i, j in pairs[:10]:  # Ограничиваем количество пар для эффективности
                    if i < X.shape[0] and j < X.shape[0]:
                        similarity = cosine_similarity(
                            X[i].reshape(1, -1), 
                            X[j].reshape(1, -1)
                        )[0][0]
                        intra_similarities.append(similarity)
        
        mean_similarity = np.mean(intra_similarities) if intra_similarities else 0
        std_similarity = np.std(intra_similarities) if intra_similarities else 0
        
        return {
            'mean_intra_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'n_pairs_evaluated': len(intra_similarities)
        }
    
    def evaluate_computational_efficiency(self, vectorization_func, method_name, **kwargs):
        """Оценка вычислительной эффективности"""
        print(f"Оценка вычислительной эффективности для {method_name}...")
        
        # Измерение времени выполнения
        start_time = time.time()
        X = vectorization_func(**kwargs)
        execution_time = time.time() - start_time
        
        # Оценка использования памяти
        memory_usage = X.data.nbytes if hasattr(X, 'data') else X.nbytes
        
        return {
            'execution_time': execution_time,
            'memory_usage_mb': memory_usage / (1024 * 1024),
            'sparsity': 1.0 - (X.nnz / (X.shape[0] * X.shape[1])) if hasattr(X, 'nnz') else 0
        }
    
    def evaluate_classification_performance(self, X, method_name):
        """Оценка производительности в задаче классификации"""
        print(f"Оценка классификации для {method_name}...")
        
        # Преобразование в плотную матрицу если нужно и возможно
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_dense, self.labels, test_size=0.3, random_state=42, stratify=self.labels
        )
        
        # Обучение простого классификатора
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Предсказание и оценка
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'classification_accuracy': accuracy,
            'n_features': X.shape[1]
        }
    
    def comprehensive_comparison(self):
        """Всестороннее сравнение методов векторизации"""
        print("ЗАПУСК ВСЕСТОРОННЕГО СРАВНЕНИЯ МЕТОДОВ ВЕКТОРИЗАЦИИ")
        print("="*60)
        
        # Конфигурации методов
        methods_config = [
            {
                'name': 'One-Hot Uni',
                'func': self.vectorizer.one_hot_encoding,
                'kwargs': {'ngram_range': (1, 1), 'max_features': 10000}
            },
            {
                'name': 'BoW Uni',
                'func': self.vectorizer.bag_of_words,
                'kwargs': {'ngram_range': (1, 1), 'max_features': 10000, 'weighting': 'count'}
            },
            {
                'name': 'TF-IDF Uni',
                'func': self.vectorizer.tfidf_vectorizer,
                'kwargs': {'ngram_range': (1, 1), 'max_features': 10000}
            },
            {
                'name': 'TF-IDF Bi',
                'func': self.vectorizer.tfidf_vectorizer,
                'kwargs': {'ngram_range': (1, 2), 'max_features': 15000}
            },
            {
                'name': 'TF-IDF Tri',
                'func': self.vectorizer.tfidf_vectorizer,
                'kwargs': {'ngram_range': (1, 3), 'max_features': 20000}
            }
        ]
        
        results = []
        
        for config in methods_config:
            print(f"\n--- Оценка {config['name']} ---")
            
            # Применение векторизации
            X = config['func'](**config['kwargs'])
            
            # Оценка различных аспектов
            efficiency = self.evaluate_computational_efficiency(
                config['func'], config['name'], **config['kwargs']
            )
            
            semantics = self.evaluate_semantic_coherence(X, config['name'])
            
            classification = self.evaluate_classification_performance(X, config['name'])
            
            # Сбор результатов
            method_results = {
                'Method': config['name'],
                'Dimensionality': X.shape[1],
                'Sparsity': efficiency['sparsity'],
                'Execution_Time_s': efficiency['execution_time'],
                'Memory_Usage_MB': efficiency['memory_usage_mb'],
                'Mean_Intra_Similarity': semantics['mean_intra_similarity'],
                'Classification_Accuracy': classification['classification_accuracy'],
                'Ngram_Range': str(config['kwargs'].get('ngram_range', (1, 1)))
            }
            
            results.append(method_results)
            self.results[config['name']] = {
                'X': X,
                'metrics': method_results
            }
        
        # Создание сводной таблицы
        self.comparison_df = pd.DataFrame(results)
        
        # Сохранение результатов
        self.comparison_df.to_csv('vectorization_comprehensive_comparison.csv', 
                                 index=False, encoding='utf-8')
        
        return self.comparison_df
    
    def visualize_comparison(self):
        """Визуализация результатов сравнения"""
        if self.comparison_df.empty:
            print("Сначала выполните comprehensive_comparison()")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Сравнительный анализ методов векторизации', fontsize=16)
        
        # 1. Точность классификации
        axes[0, 0].bar(self.comparison_df['Method'], 
                      self.comparison_df['Classification_Accuracy'])
        axes[0, 0].set_title('Точность классификации')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Размерность
        axes[0, 1].bar(self.comparison_df['Method'], 
                      self.comparison_df['Dimensionality'])
        axes[0, 1].set_title('Размерность признаков')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Время выполнения
        axes[0, 2].bar(self.comparison_df['Method'], 
                      self.comparison_df['Execution_Time_s'])
        axes[0, 2].set_title('Время выполнения (сек)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Использование памяти
        axes[1, 0].bar(self.comparison_df['Method'], 
                      self.comparison_df['Memory_Usage_MB'])
        axes[1, 0].set_title('Использование памяти (МБ)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Семантическое сходство
        axes[1, 1].bar(self.comparison_df['Method'], 
                      self.comparison_df['Mean_Intra_Similarity'])
        axes[1, 1].set_title('Среднее семантическое сходство')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Разреженность
        axes[1, 2].bar(self.comparison_df['Method'], 
                      self.comparison_df['Sparsity'])
        axes[1, 2].set_title('Разреженность')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('vectorization_comparison_visualization.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Генерация отчета по сравнению"""
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЙ ОТЧЕТ ПО СРАВНЕНИЮ МЕТОДОВ ВЕКТОРИЗАЦИИ")
        print("="*80)
        
        # Лучший метод по каждой метрике
        metrics_to_maximize = ['Classification_Accuracy', 'Mean_Intra_Similarity']
        metrics_to_minimize = ['Execution_Time_s', 'Memory_Usage_MB', 'Dimensionality', 'Sparsity']
        
        print("\nЛУЧШИЕ МЕТОДЫ ПО МЕТРИКАМ:")
        print("-" * 50)
        
        for metric in metrics_to_maximize:
            best_method = self.comparison_df.loc[self.comparison_df[metric].idxmax(), 'Method']
            best_value = self.comparison_df[metric].max()
            print(f"{metric}: {best_method} ({best_value:.4f})")
        
        for metric in metrics_to_minimize:
            best_method = self.comparison_df.loc[self.comparison_df[metric].idxmin(), 'Method']
            best_value = self.comparison_df[metric].min()
            print(f"{metric}: {best_method} ({best_value:.4f})")
        
        # Рекомендации
        print("\nРЕКОМЕНДАЦИИ:")
        print("-" * 30)
        
        # Для высокой точности классификации
        best_accuracy = self.comparison_df.loc[self.comparison_df['Classification_Accuracy'].idxmax()]
        print(f"Для максимальной точности: {best_accuracy['Method']} "
              f"(точность: {best_accuracy['Classification_Accuracy']:.3f})")
        
        # Для баланса производительности и качества
        balanced_score = (self.comparison_df['Classification_Accuracy'] * 0.6 + 
                         (1 - self.comparison_df['Execution_Time_s'] / 
                          self.comparison_df['Execution_Time_s'].max()) * 0.4)
        best_balanced = self.comparison_df.loc[balanced_score.idxmax()]
        print(f"Для баланса качества и скорости: {best_balanced['Method']}")
        
        # Для семантических задач
        best_semantics = self.comparison_df.loc[self.comparison_df['Mean_Intra_Similarity'].idxmax()]
        print(f"Для семантического анализа: {best_semantics['Method']} "
              f"(сходство: {best_semantics['Mean_Intra_Similarity']:.3f})")

# Запуск сравнительного анализа
def run_comparison_analysis():
    comparator = VectorizationComparator()
    comparator.load_data()
    results_df = comparator.comprehensive_comparison()
    comparator.visualize_comparison()
    comparator.generate_report()
    
    return comparator

if __name__ == "__main__":
    run_comparison_analysis()