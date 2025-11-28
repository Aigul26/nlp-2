import gensim
from gensim.models import Word2Vec, FastText as GensimFastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json
import matplotlib.pyplot as plt
import seaborn as sns

class DistributedRepresentations:
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        
    def prepare_word_level_data(self, corpus):
        """Подготовка данных для word-level моделей"""
        tokenized_corpus = [doc.split() for doc in corpus]
        return tokenized_corpus
    
    def prepare_doc_level_data(self, corpus, labels=None):
        """Подготовка данных для doc-level моделей"""
        tagged_docs = []
        for i, doc in enumerate(corpus):
            tokens = doc.split()
            if labels:
                tag = f"{labels[i]}_{i}"
            else:
                tag = f"DOC_{i}"
            tagged_docs.append(TaggedDocument(tokens, [tag]))
        return tagged_docs
    
    def train_word2vec(self, corpus, vector_size=100, window=5, 
                      min_count=5, sg=0, workers=4, **kwargs):
        """Обучение Word2Vec модели"""
        print(f"Обучение Word2Vec (sg={sg})...")
        
        tokenized_corpus = self.prepare_word_level_data(corpus)
        
        start_time = time.time()
        
        model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,  # 0 для CBOW, 1 для Skip-gram
            workers=workers,
            **kwargs
        )
        
        training_time = time.time() - start_time
        
        model_name = f"word2vec_{'skipgram' if sg == 1 else 'cbow'}_{vector_size}d"
        self.models[model_name] = {
            'model': model,
            'type': 'word2vec',
            'training_time': training_time,
            'vocab_size': len(model.wv.key_to_index)
        }
        
        print(f"Обучение завершено за {training_time:.2f} сек")
        print(f"Размер словаря: {len(model.wv.key_to_index)}")
        
        return model
    
    def train_fasttext(self, corpus, vector_size=100, window=5,
                      min_count=5, sg=0, workers=4, **kwargs):
        """Обучение FastText модели через gensim"""
        print(f"Обучение FastText через gensim (sg={sg})...")
        
        tokenized_corpus = self.prepare_word_level_data(corpus)
        
        start_time = time.time()
        
        model = GensimFastText(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            **kwargs
        )
        
        training_time = time.time() - start_time
        
        model_name = f"fasttext_{'skipgram' if sg == 1 else 'cbow'}_{vector_size}d"
        self.models[model_name] = {
            'model': model,
            'type': 'fasttext',
            'training_time': training_time,
            'vocab_size': len(model.wv.key_to_index)
        }
        
        print(f"Обучение завершено за {training_time:.2f} сек")
        print(f"Размер словаря: {len(model.wv.key_to_index)}")
        
        return model
    
    def train_doc2vec(self, corpus, labels, vector_size=100, window=5,
                     min_count=5, dm=1, workers=4, **kwargs):
        """Обучение Doc2Vec модели"""
        print(f"Обучение Doc2Vec (dm={dm})...")
        
        tagged_docs = self.prepare_doc_level_data(corpus, labels)
        
        start_time = time.time()
        
        model = Doc2Vec(
            documents=tagged_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            dm=dm,  # 1 для PV-DM, 0 для PV-DBOW
            workers=workers,
            **kwargs
        )
        
        training_time = time.time() - start_time
        
        model_name = f"doc2vec_{'pvdm' if dm == 1 else 'pvdbow'}_{vector_size}d"
        self.models[model_name] = {
            'model': model,
            'type': 'doc2vec',
            'training_time': training_time
        }
        
        print(f"Обучение завершено за {training_time:.2f} сек")
        
        return model
    
    def evaluate_word_analogies(self, model, model_name, analogies_file=None):
        """Оценка точности аналогий"""
        print(f"Оценка точности аналогий для {model_name}...")
        
        # Создание тестовых аналогий для русского языка
        russian_analogies = [
            # Столицы
            ['москва', 'россия', 'киев', 'украина'],
            ['берлин', 'германия', 'париж', 'франция'],
            ['токио', 'япония', 'пекин', 'китай'],
            # Гендерные аналогии
            ['король', 'королева', 'мужчина', 'женщина'],
            ['актер', 'актриса', 'отец', 'мать'],
            ['принц', 'принцесса', 'юноша', 'девушка'],
            # Синтаксические
            ['хороший', 'лучше', 'плохой', 'хуже'],
            ['быстро', 'быстрее', 'медленно', 'медленнее'],
            ['большой', 'больше', 'маленький', 'меньше'],
            # Профессии
            ['врач', 'больница', 'учитель', 'школа'],
            ['повар', 'ресторан', 'библиотекарь', 'библиотека']
        ]
        
        correct = 0
        total = len(russian_analogies)
        detailed_results = []
        
        for analogy in russian_analogies:
            try:
                predicted = model.wv.most_similar(
                    positive=[analogy[1], analogy[2]],
                    negative=[analogy[0]],
                    topn=3
                )
                
                top_prediction = predicted[0][0]
                similarity_score = predicted[0][1]
                
                is_correct = top_prediction == analogy[3]
                if is_correct:
                    correct += 1
                
                detailed_results.append({
                    'analogy': f"{analogy[0]} : {analogy[1]} = {analogy[2]} : {analogy[3]}",
                    'predicted': top_prediction,
                    'similarity': similarity_score,
                    'correct': is_correct,
                    'top_3': predicted
                })
                    
            except KeyError as e:
                # Если какое-то слово отсутствует в словаре
                detailed_results.append({
                    'analogy': f"{analogy[0]} : {analogy[1]} = {analogy[2]} : {analogy[3]}",
                    'predicted': 'OOV',
                    'similarity': 0,
                    'correct': False,
                    'error': f"Отсутствует слово: {e}"
                })
                continue
        
        accuracy = correct / total if total > 0 else 0
        
        self.evaluation_results[model_name] = {
            'word_analogy_accuracy': accuracy,
            'n_analogies_tested': total,
            'n_correct': correct,
            'detailed_analogies': detailed_results
        }

        
        return accuracy
    
    def evaluate_semantic_similarity(self, model, model_name, word_pairs=None):
        """Оценка семантического сходства"""
        print(f"Оценка семантического сходства для {model_name}...")
        
        if word_pairs is None:
            word_pairs = [
                ('город', 'мегаполис'),
                ('компьютер', 'ноутбук'),
                ('счастье', 'радость'),
                ('дом', 'здание'),
                ('быстро', 'стремительно'),
                ('холодный', 'ледяной'),
                ('говорить', 'разговаривать'),
                ('красивый', 'привлекательный')
            ]
        
        similarities = []
        valid_pairs = 0
        detailed_similarities = []
        
        for word1, word2 in word_pairs:
            try:
                similarity = model.wv.similarity(word1, word2)
                similarities.append(similarity)
                valid_pairs += 1
                detailed_similarities.append({
                    'pair': f"{word1} - {word2}",
                    'similarity': similarity
                })
            except KeyError as e:
                detailed_similarities.append({
                    'pair': f"{word1} - {word2}",
                    'similarity': 0,
                    'error': f"Отсутствует слово: {e}"
                })
                continue
        
        if valid_pairs > 0:
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            max_similarity = np.max(similarities)
            min_similarity = np.min(similarities)
        else:
            mean_similarity = 0
            std_similarity = 0
            max_similarity = 0
            min_similarity = 0
        
        # Сохранение результатов
        if model_name in self.evaluation_results:
            self.evaluation_results[model_name].update({
                'semantic_similarity_mean': mean_similarity,
                'semantic_similarity_std': std_similarity,
                'semantic_similarity_max': max_similarity,
                'semantic_similarity_min': min_similarity,
                'n_valid_similarity_pairs': valid_pairs,
                'detailed_similarities': detailed_similarities
            })
        else:
            self.evaluation_results[model_name] = {
                'semantic_similarity_mean': mean_similarity,
                'semantic_similarity_std': std_similarity,
                'semantic_similarity_max': max_similarity,
                'semantic_similarity_min': min_similarity,
                'n_valid_similarity_pairs': valid_pairs,
                'detailed_similarities': detailed_similarities
            }
        
        print(f"Среднее семантическое сходство: {mean_similarity:.3f} (±{std_similarity:.3f})")
        print(f"Диапазон: [{min_similarity:.3f}, {max_similarity:.3f}]")
        print(f"Валидных пар: {valid_pairs}/{len(word_pairs)}")
        
        return mean_similarity
    
    def evaluate_oov_coverage(self, model, model_name, test_words=None):
        """Оценка покрытия OOV слов"""
        
        if test_words is None:
            test_words = [
                'город', 'страна', 'человек', 'работа', 'время', 
                'деньги', 'политика', 'экономика', 'спорт', 'культура',
                'развитие', 'технология', 'образование', 'здоровье', 'природа',
                'искусство', 'музыка', 'литература', 'наука', 'исследование'
            ]
        
        oov_words = []
        oov_details = []
        
        for word in test_words:
            if word not in model.wv:
                oov_words.append(word)
                oov_details.append({
                    'word': word,
                    'status': 'OOV'
                })
            else:
                oov_details.append({
                    'word': word,
                    'status': 'В словаре',
                    'vector_norm': np.linalg.norm(model.wv[word])
                })
        
        coverage = 1 - (len(oov_words) / len(test_words))
        
        if model_name in self.evaluation_results:
            self.evaluation_results[model_name].update({
                'oov_coverage': coverage,
                'n_oov_words': len(oov_words),
                'n_test_words': len(test_words),
                'oov_details': oov_details
            })
        else:
            self.evaluation_results[model_name] = {
                'oov_coverage': coverage,
                'n_oov_words': len(oov_words),
                'n_test_words': len(test_words),
                'oov_details': oov_details
            }
        
        return coverage
    
    def evaluate_morphological_robustness(self, model, model_name, test_words=None):
        """Оценка морфологической устойчивости (особенно для FastText)"""
        print(f"Оценка морфологической устойчивости для {model_name}...")
        
        if test_words is None:
            test_words = ['город', 'страна', 'человек', 'работа', 'время']
        
        # Тестируем способность модели работать с OOV словами через n-граммы
        oov_words = []
        morphological_tests = []
        
        for base_word in test_words:
            # Создаем варианты слов с разными окончаниями
            variants = [
                f"{base_word}ый",  # прилагательное
                f"{base_word}а",   # женский род
                f"{base_word}ов",  # родительный падеж
                f"{base_word}ам",  # дательный падеж
                f"{base_word}ами", # творительный падеж
                f"{base_word}ный", # другое окончание
                f"{base_word}ение" # существительное
            ]
            
            for variant in variants:
                if variant not in model.wv:
                    oov_words.append(variant)
                morphological_tests.append({
                    'base_word': base_word,
                    'variant': variant,
                    'in_vocab': variant in model.wv
                })
        
        # Для FastText проверяем, можем ли мы получить векторы для OOV слов
        successful_oov = 0
        oov_vectors = []
        
        for word in oov_words[:10]:  # Проверяем только первые 10
            try:
                vector = model.wv[word]  # FastText должен уметь это
                successful_oov += 1
                oov_vectors.append({
                    'word': word,
                    'vector_norm': np.linalg.norm(vector),
                    'status': 'Успех'
                })
            except KeyError:
                oov_vectors.append({
                    'word': word,
                    'status': 'Неудача'
                })
                pass
        
        oov_success_rate = successful_oov / min(10, len(oov_words)) if oov_words else 0
        
        if model_name in self.evaluation_results:
            self.evaluation_results[model_name].update({
                'morphological_oov_success': oov_success_rate,
                'n_morphological_tests': len(morphological_tests),
                'n_oov_variants': len(oov_words),
                'morphological_details': morphological_tests,
                'oov_vector_results': oov_vectors
            })
        else:
            self.evaluation_results[model_name] = {
                'morphological_oov_success': oov_success_rate,
                'n_morphological_tests': len(morphological_tests),
                'n_oov_variants': len(oov_words),
                'morphological_details': morphological_tests,
                'oov_vector_results': oov_vectors
            }
        
        return oov_success_rate
    
    def evaluate_document_embeddings(self, model, corpus, labels, model_name):
        """Оценка качества document embeddings"""
        print(f"Оценка document embeddings для {model_name}...")
        
        # Извлечение векторов документов
        doc_vectors = []
        valid_labels = []
        extraction_details = []
        
        for i, doc in enumerate(corpus):
            try:
                if hasattr(model, 'dv'):
                    # Для Doc2Vec
                    tag = f"{labels[i]}_{i}" if labels else f"DOC_{i}"
                    vector = model.dv[tag]
                    method = 'doc2vec_direct'
                else:
                    # Для word-level моделей (усреднение векторов слов)
                    tokens = doc.split()
                    word_vectors = []
                    for word in tokens:
                        if word in model.wv:
                            word_vectors.append(model.wv[word])
                    
                    if word_vectors:
                        vector = np.mean(word_vectors, axis=0)
                        method = 'word_avg'
                    else:
                        extraction_details.append({
                            'doc_id': i,
                            'status': 'No valid words',
                            'method': 'word_avg'
                        })
                        continue
                
                doc_vectors.append(vector)
                valid_labels.append(labels[i])
                extraction_details.append({
                    'doc_id': i,
                    'status': 'Success',
                    'method': method,
                    'vector_norm': np.linalg.norm(vector)
                })
                
            except (KeyError, IndexError, AttributeError) as e:
                extraction_details.append({
                    'doc_id': i,
                    'status': f'Error: {str(e)}',
                    'method': 'unknown'
                })
                continue
        
        if not doc_vectors:
            print("Не удалось извлечь векторы документов")
            ari_score = 0
            accuracy = 0
        else:
            doc_vectors = np.array(doc_vectors)
            
            # Кластеризация
            n_clusters = min(len(set(valid_labels)), len(doc_vectors))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(doc_vectors)
                ari_score = adjusted_rand_score(valid_labels, clusters)
            else:
                ari_score = 0
            
            # Классификация
            if len(set(valid_labels)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    doc_vectors, valid_labels, test_size=0.3, random_state=42, stratify=valid_labels
                )
                
                clf = LogisticRegression(random_state=42, max_iter=1000)
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
            else:
                accuracy = 0
        
        if model_name in self.evaluation_results:
            self.evaluation_results[model_name].update({
                'clustering_ari': ari_score,
                'classification_accuracy': accuracy,
                'n_documents_processed': len(doc_vectors),
                'document_extraction_details': extraction_details
            })
        else:
            self.evaluation_results[model_name] = {
                'clustering_ari': ari_score,
                'classification_accuracy': accuracy,
                'n_documents_processed': len(doc_vectors),
                'document_extraction_details': extraction_details
            }
        
        print(f"Качество кластеризации (ARI): {ari_score:.3f}")
        print(f"Точность классификации: {accuracy:.3f}")
        print(f"Обработано документов: {len(doc_vectors)}/{len(corpus)}")
        
        return ari_score, accuracy
    
    def get_model_performance(self, model_name):
        """Получение метрик производительности модели"""
        if model_name not in self.models:
            print(f"Модель {model_name} не найдена")
            return None
        
        model_info = self.models[model_name]
        evaluation = self.evaluation_results.get(model_name, {})
        
        performance = {
            'model_name': model_name,
            'model_type': model_info['type'],
            'training_time': model_info['training_time'],
            'vocab_size': model_info.get('vocab_size', 'N/A')
        }
        
        performance.update(evaluation)
        return performance
    
    def comprehensive_training(self, corpus, labels):
        """Всестороннее обучение различных моделей"""
        print("ЗАПУСК ВСЕСТОРОННЕГО ОБУЧЕНИЯ МОДЕЛЕЙ")
        print("="*50)
        
        # Конфигурации для экспериментов
        configs = [
            # Word2Vec
            {'method': 'word2vec', 'vector_size': 100, 'window': 5, 'sg': 0, 'min_count': 3},
            {'method': 'word2vec', 'vector_size': 100, 'window': 5, 'sg': 1, 'min_count': 3},
            {'method': 'word2vec', 'vector_size': 200, 'window': 8, 'sg': 0, 'min_count': 3},
            {'method': 'word2vec', 'vector_size': 200, 'window': 8, 'sg': 1, 'min_count': 3},
            
            # FastText через gensim
            {'method': 'fasttext', 'vector_size': 100, 'window': 5, 'sg': 0, 'min_count': 3},
            {'method': 'fasttext', 'vector_size': 100, 'window': 5, 'sg': 1, 'min_count': 3},
            {'method': 'fasttext', 'vector_size': 200, 'window': 8, 'sg': 0, 'min_count': 3},
            
            # Doc2Vec
            {'method': 'doc2vec', 'vector_size': 100, 'window': 5, 'dm': 1, 'min_count': 3},
            {'method': 'doc2vec', 'vector_size': 100, 'window': 5, 'dm': 0, 'min_count': 3},
        ]
        
        # Тестовые слова для оценки
        test_words = ['город', 'страна', 'человек', 'работа', 'время', 
                     'деньги', 'политика', 'экономика', 'спорт', 'культура']
        
        for config in configs:
            method = config.pop('method')
            
            print(f"\n{'='*40}")
            print(f"ОБУЧЕНИЕ: {method.upper()} с параметрами {config}")
            print(f"{'='*40}")
            
            try:
                if method == 'word2vec':
                    model = self.train_word2vec(corpus, **config)
                    model_name = f"word2vec_{'skipgram' if config['sg'] == 1 else 'cbow'}_{config['vector_size']}d"
                    
                elif method == 'fasttext':
                    model = self.train_fasttext(corpus, **config)
                    model_name = f"fasttext_{'skipgram' if config['sg'] == 1 else 'cbow'}_{config['vector_size']}d"
                    
                elif method == 'doc2vec':
                    model = self.train_doc2vec(corpus, labels, **config)
                    model_name = f"doc2vec_{'pvdm' if config['dm'] == 1 else 'pvdbow'}_{config['vector_size']}d"
                
                # Оценка для word-level моделей
                if method in ['word2vec', 'fasttext']:
                    self.evaluate_word_analogies(model, model_name)
                    self.evaluate_semantic_similarity(model, model_name)
                    self.evaluate_oov_coverage(model, model_name, test_words)
                    self.evaluate_morphological_robustness(model, model_name, test_words)
                
                # Оценка document embeddings для всех моделей
                self.evaluate_document_embeddings(model, corpus, labels, model_name)
                
                # Сохранение модели
                model.save(f"{model_name}.model")
                print(f"✅ Модель сохранена как {model_name}.model")
                
            except Exception as e:
                print(f"❌ Ошибка при обучении {method}: {str(e)}")
                continue
    
    def generate_comparison_report(self):
        """Генерация отчета по сравнению моделей"""
        print("\n" + "="*80)
        print("ОТЧЕТ ПО СРАВНЕНИЮ МОДЕЛЕЙ РАСПРЕДЕЛЕННЫХ ПРЕДСТАВЛЕНИЙ")
        print("="*80)
        
        results_list = []
        for model_name, metrics in self.evaluation_results.items():
            # Получаем базовую информацию о модели
            model_info = self.models.get(model_name, {})
            row = {
                'Model': model_name,
                'Type': model_info.get('type', 'unknown'),
                'Training_Time': model_info.get('training_time', 0),
                'Vocab_Size': model_info.get('vocab_size', 0)
            }
            
            # Добавляем метрики оценки
            for key, value in metrics.items():
                if key not in ['detailed_analogies', 'detailed_similarities', 
                             'oov_details', 'morphological_details', 
                             'document_extraction_details', 'oov_vector_results']:
                    row[key] = value
            
            results_list.append(row)
        
        comparison_df = pd.DataFrame(results_list)
        
        # Заполняем пропущенные значения
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns
        comparison_df[numeric_columns] = comparison_df[numeric_columns].fillna(0)
        
        # Сохранение результатов
        comparison_df.to_csv('distributed_models_comparison.csv', index=False, encoding='utf-8')
        
        # Выбираем ключевые колонки для отображения
        key_columns = ['Model', 'word_analogy_accuracy', 'semantic_similarity_mean', 
                      'oov_coverage', 'morphological_oov_success', 'classification_accuracy', 
                      'clustering_ari', 'Training_Time']
        
        available_columns = [col for col in key_columns if col in comparison_df.columns]
        display_df = comparison_df[available_columns]
        
        # Визуализация результатов
        self._visualize_results(comparison_df)
        
        return comparison_df
    
    def _visualize_results(self, df):
        """Визуализация результатов сравнения"""
        try:
            # Создаем фигуру с несколькими subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Сравнение моделей распределенных представлений', fontsize=16, fontweight='bold')
            
            metrics_to_plot = [
                ('word_analogy_accuracy', 'Точность аналогий'),
                ('semantic_similarity_mean', 'Семантическое сходство'),
                ('classification_accuracy', 'Точность классификации'),
                ('clustering_ari', 'Качество кластеризации (ARI)'),
                ('oov_coverage', 'Покрытие словаря'),
                ('morphological_oov_success', 'Успешность OOV')
            ]
            
            for idx, (metric, title) in enumerate(metrics_to_plot):
                if metric in df.columns:
                    ax = axes[idx // 3, idx % 3]
                    
                    # Сортируем по метрике
                    plot_data = df[['Model', metric]].dropna().sort_values(metric, ascending=True)
                    
                    if not plot_data.empty:
                        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
                        bars = ax.barh(plot_data['Model'], plot_data[metric], color=colors)
                        ax.set_title(title, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Значение метрики')
                        
                        # Добавляем значения на барчарты
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.3f}', ha='left', va='center', fontsize=9)
                        
                        ax.grid(True, alpha=0.3, axis='x')
            
            # Убираем пустые subplots
            for idx in range(len(metrics_to_plot), 6):
                axes[idx // 3, idx % 3].set_visible(False)
            
            
            # Дополнительная визуализация: время обучения vs качество
            if 'Training_Time' in df.columns and 'classification_accuracy' in df.columns:
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(df['Training_Time'], df['classification_accuracy'], 
                                    s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
                
                for i, model_name in enumerate(df['Model']):
                    plt.annotate(model_name, 
                               (df['Training_Time'].iloc[i], df['classification_accuracy'].iloc[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                plt.xlabel('Время обучения (сек)')
                plt.ylabel('Точность классификации')
                plt.title('Соотношение времени обучения и качества моделей')
                plt.colorbar(scatter, label='Индекс модели')
                plt.grid(True, alpha=0.3)
                plt.savefig('training_time_vs_accuracy.png', dpi=300, bbox_inches='tight')
                
        except Exception as e:
            print(f"Ошибка при визуализации: {e}")
    
    def get_best_model(self, metric='classification_accuracy'):
        """Получение лучшей модели по указанной метрике"""
        if not self.evaluation_results:
            print("Нет результатов для сравнения")
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, metrics in self.evaluation_results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        if best_model:
            print(f"Лучшая модель по метрике '{metric}': {best_model} (score: {best_score:.3f})")
            return best_model, best_score
        else:
            print(f"Метрика '{metric}' не найдена в результатах")
            return None

# Пример использования
def run_distributed_models_experiment():
    """Запуск эксперимента с распределенными представлениями"""
    # Загрузка данных
    try:
        with open('corpus.jsonl', 'r', encoding='utf-8') as f:
            corpus = []
            labels = []
            for line in f:
                item = json.loads(line)
                corpus.append(item['text'])
                labels.append(item['category'])
        
        print(f"Загружено {len(corpus)} документов")
        print(f"Категории: {set(labels)}")
        
    except FileNotFoundError:
        print("Файл corpus_processed.jsonl не найден. Создаем демо-данные...")
        # Создаем демо-данные если файл не найден
        corpus = ["это пример текста о политике " * 10,
                 "новости спорта и результаты матчей " * 10,
                 "экономические показатели и рынки " * 10,
                 "культурные события и искусство " * 10] * 25
        labels = ['политика', 'спорт', 'экономика', 'культура'] * 25
    
    # Обучение моделей
    dr = DistributedRepresentations()
    dr.comprehensive_training(corpus, labels)
    
    # Генерация отчета
    report_df = dr.generate_comparison_report()
    
    # Поиск лучшей модели
    dr.get_best_model('classification_accuracy')
    dr.get_best_model('word_analogy_accuracy')
    
    return dr, report_df

if __name__ == "__main__":
    dr, report = run_distributed_models_experiment()