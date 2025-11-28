import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import time
import json
from collections import defaultdict

class ClassicalVectorizers:
    def __init__(self):
        self.vectorizers = {}
        self.vocabularies = {}
        self.metrics = defaultdict(dict)
    
    def load_corpus(self, file_path='corpus.jsonl'):
        """Загрузка предобработанного корпуса"""
        self.corpus = []
        self.labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.corpus.append(item['text'])
                self.labels.append(item['category'])
        
        print(f"Загружено {len(self.corpus)} документов")
        return self.corpus, self.labels
    
    def one_hot_encoding(self, ngram_range=(1, 1), max_features=10000):
        """One-Hot Encoding с поддержкой n-грамм"""
        print("Применение One-Hot Encoding...")
        
        start_time = time.time()
        
        vectorizer = CountVectorizer(
            binary=True,
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r'\b\w+\b'
        )
        
        X = vectorizer.fit_transform(self.corpus)
        
        # Сохранение метрик
        exec_time = time.time() - start_time
        self._save_metrics('one_hot', X, exec_time, ngram_range)
        self.vectorizers['one_hot'] = vectorizer
        self.vocabularies['one_hot'] = vectorizer.get_feature_names_out()
        
        return X
    
    def bag_of_words(self, ngram_range=(1, 1), max_features=10000, 
                    binary=False, weighting='count'):
        """Bag of Words с различными схемами взвешивания"""
        print("Применение Bag of Words...")
        
        start_time = time.time()
        
        if weighting == 'binary':
            binary = True
        
        vectorizer = CountVectorizer(
            binary=binary,
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r'\b\w+\b'
        )
        
        X = vectorizer.fit_transform(self.corpus)
        
        exec_time = time.time() - start_time
        self._save_metrics('bow', X, exec_time, ngram_range)
        self.vectorizers['bow'] = vectorizer
        self.vocabularies['bow'] = vectorizer.get_feature_names_out()
        
        return X
    
    def tfidf_vectorizer(self, ngram_range=(1, 1), max_features=10000,
                        smooth_idf=True, sublinear_tf=False):
        """TF-IDF векторизатор с настройкой параметров"""
        print("Применение TF-IDF...")
        
        start_time = time.time()
        
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            token_pattern=r'\b\w+\b'
        )
        
        X = vectorizer.fit_transform(self.corpus)
        
        exec_time = time.time() - start_time
        self._save_metrics('tfidf', X, exec_time, ngram_range)
        self.vectorizers['tfidf'] = vectorizer
        self.vocabularies['tfidf'] = vectorizer.get_feature_names_out()
        
        return X
    
    def _save_metrics(self, method, X, exec_time, ngram_range):
        """Сохранение метрик для метода векторизации"""
        n_samples, n_features = X.shape
        sparsity = 1.0 - (X.nnz / (n_samples * n_features))
        
        self.metrics[method] = {
            'dimensionality': n_features,
            'sparsity': sparsity,
            'execution_time': exec_time,
            'n_samples': n_samples,
            'n_nonzero': X.nnz,
            'ngram_range': ngram_range
        }
    
    def analyze_sparsity(self, X, method_name):
        """Анализ разреженности матрицы"""
        density = X.nnz / (X.shape[0] * X.shape[1])
        sparsity = 1 - density
        
        print(f"\n--- Анализ разреженности для {method_name} ---")
        print(f"Размерность: {X.shape}")
        print(f"Количество ненулевых элементов: {X.nnz}")
        print(f"Разреженность: {sparsity:.4f}")
        print(f"Плотность: {density:.6f}")
        
        return sparsity
    
    def compare_methods(self):
        """Сравнительный анализ всех методов векторизации"""
        print("\n" + "="*50)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ ВЕКТОРИЗАЦИИ")
        print("="*50)
        
        results = []
        
        for method, metrics in self.metrics.items():
            results.append({
                'Method': method,
                'Dimensionality': metrics['dimensionality'],
                'Sparsity': f"{metrics['sparsity']:.4f}",
                'Execution Time (s)': f"{metrics['execution_time']:.3f}",
                'N-gram Range': metrics['ngram_range'],
                'Non-zero Elements': metrics['n_nonzero']
            })
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        # Сохранение результатов в CSV
        df.to_csv('vectorization_metrics.csv', index=False, encoding='utf-8')
        print(f"\nРезультаты сохранены в vectorization_metrics.csv")
        
        return df
    
    def experiment_with_ngrams(self):
        """Эксперименты с различными n-граммами"""
        ngram_configs = [
            (1, 1),  # униграммы
            (1, 2),  # униграммы + биграммы
            (1, 3),  # униграммы + биграммы + триграммы
            (2, 2),  # только биграммы
            (3, 3)   # только триграммы
        ]
        
        print("\nЭКСПЕРИМЕНТЫ С N-ГРАММАМИ")
        print("="*40)
        
        for ngram_range in ngram_configs:
            print(f"\nN-gram range: {ngram_range}")
            print("-" * 20)
            
            # TF-IDF с разными n-gram диапазонами
            X_tfidf = self.tfidf_vectorizer(ngram_range=ngram_range)
            self.analyze_sparsity(X_tfidf, f"TF-IDF {ngram_range}")

# Пример использования
if __name__ == "__main__":
    vectorizer = ClassicalVectorizers()
    corpus, labels = vectorizer.load_corpus()
    
    # Применение различных методов векторизации
    X_one_hot = vectorizer.one_hot_encoding(ngram_range=(1, 1))
    X_bow = vectorizer.bag_of_words(ngram_range=(1, 1))
    X_tfidf = vectorizer.tfidf_vectorizer(ngram_range=(1, 1))
    
    # Сравнение методов
    comparison_df = vectorizer.compare_methods()
    
    # Эксперименты с n-граммами
    vectorizer.experiment_with_ngrams()