import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import random
from typing import List, Dict, Any

class VectorSpaceExplorer:
    """Веб-интерфейс для исследования векторных пространств"""
    
    def __init__(self):
        self.models = {}  # Словарь для хранения всех моделей
        self.current_model = None
        self.current_model_name = None
        self.vocab = []
        
        # Создаем все демо-модели для прототипа
        self._create_all_demo_models()
    
    def _create_all_demo_models(self):
        """Создание всех демонстрационных моделей"""
        try:
            sentences = self._generate_training_sentences()
            
            st.info("🔄 Создание демонстрационных моделей...")
            
            # Прогресс-бар для отслеживания создания моделей
            progress_bar = st.progress(0)
            status_text = st.empty()
            # 1. Word2Vec Skip-gram
            status_text.text("Создание Word2Vec Skip-gram...")
            self.models['word2vec_sg'] = Word2Vec(
                sentences=sentences,
                vector_size=150,
                window=8,
                min_count=1,
                workers=4,
                epochs=50,
                sg=1  # skip-gram
            )
            progress_bar.progress(20)
            
            # 2. Word2Vec CBOW
            status_text.text("Создание Word2Vec CBOW...")
            self.models['word2vec_cbow'] = Word2Vec(
                sentences=sentences,
                vector_size=150,
                window=8,
                min_count=1,
                workers=4,
                epochs=50,
                sg=0  # CBOW
            )
            progress_bar.progress(40)
            
            # 3. FastText Skip-gram
            status_text.text("Создание FastText Skip-gram...")
            self.models['fasttext_sg'] = FastText(
                sentences=sentences,
                vector_size=150,
                window=8,
                min_count=1,
                workers=4,
                epochs=50,
                sg=1  # skip-gram
            )
            progress_bar.progress(60)
            
            # 4. Doc2Vec PV-DM (Distributed Memory)
            status_text.text("Создание Doc2Vec PV-DM...")
            tagged_documents = [TaggedDocument(words=doc, tags=[f'doc_{i}']) 
                              for i, doc in enumerate(sentences)]
            
            self.models['doc2vec_dm'] = Doc2Vec(
                documents=tagged_documents,
                vector_size=150,
                window=8,
                min_count=1,
                workers=4,
                epochs=50,
                dm=1  # PV-DM
            )
            progress_bar.progress(80)
            
            # 5. Doc2Vec PV-DBOW (Distributed Bag of Words)
            status_text.text("Создание Doc2Vec PV-DBOW...")
            self.models['doc2vec_dbow'] = Doc2Vec(
                documents=tagged_documents,
                vector_size=150,
                window=8,
                min_count=1,
                workers=4,
                epochs=50,
                dm=0  # PV-DBOW
            )
            progress_bar.progress(100)
            
            # Устанавливаем модель по умолчанию
            self.current_model = self.models['word2vec_sg']
            self.current_model_name = "Word2Vec Skip-gram"
            
            # Создаем общий словарь
            self._create_combined_vocabulary()
            
            status_text.text("✅ Все модели успешно созданы!")
            st.success(f"✅ Создано {len(self.models)} моделей! Словарь: {len(self.vocab)} слов")
            
            # Небольшая задержка перед очисткой статуса
            import time
            time.sleep(2)
            status_text.empty()
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ Ошибка создания моделей: {e}")
            # Создаем резервный словарь
            self._create_fallback_vocabulary()
    
    def _generate_training_sentences(self) -> List[List[str]]:
        """Генерация тренировочных предложений для всех моделей"""
        sentences = []
        
        # География и города
        geo_sentences = [
            ["россия", "москва", "столица", "город", "кремль", "река", "волга"],
            ["санкт-петербург", "питер", "город", "нева", "эрмитаж", "культура"],
            ["париж", "франция", "европа", "город", "лувр", "эйфелева", "башня"],
            ["нью-йорк", "сша", "америка", "город", "небоскреб", "статуя", "свободы"],
            ["китай", "пекин", "азия", "страна", "великая", "стена", "экономика"],
            ["япония", "токио", "азия", "страна", "технологии", "сакура", "культура"],
            ["германия", "берлин", "европа", "страна", "автомобили", "бмв", "мерседес"],
            ["италия", "рим", "европа", "страна", "искусство", "пицца", "паста"],
            ["англия", "лондон", "европа", "страна", "королева", "биг-бен", "традиции"],
            ["испания", "мадрид", "европа", "страна", "футбол", "коррида", "танцы"]
        ]
        
        # Политика и власть
        politics_sentences = [
            ["путин", "президент", "власть", "кремль", "правительство", "политика"],
            ["байден", "сша", "президент", "америка", "белый", "дом", "демократия"],
            ["правительство", "министерство", "бюджет", "налоги", "законы", "государство"],
            ["парламент", "дума", "депутаты", "законы", "выборы", "голосование"],
            ["оппозиция", "протест", "митинг", "демонстрация", "требования", "власть"],
            ["дипломатия", "переговоры", "международные", "отношения", "посольство"],
            ["санкции", "экономика", "международные", "ограничения", "торговля"],
            ["война", "конфликт", "армия", "солдаты", "оружие", "безопасность"]
        ]
        
        # Культура и искусство
        culture_sentences = [
            ["кино", "фильм", "актер", "режиссер", "фестиваль", "премия", "оскар"],
            ["культура", "искусство", "музей", "театр", "выставка", "картина", "скульптура"],
            ["литература", "книга", "писатель", "поэт", "роман", "стихи", "проза"],
            ["музыка", "песня", "исполнитель", "композитор", "концерт", "альбом"],
            ["танец", "балет", "хореография", "движение", "ритм", "выступление"],
            ["архитектура", "здание", "строительство", "дизайн", "проект", "чертеж"],
            ["фотография", "камера", "снимок", "объектив", "композиция", "свет"],
            ["живопись", "художник", "краски", "полотно", "пейзаж", "портрет"]
        ]
        
        # Экономика и бизнес
        economy_sentences = [
            ["экономика", "деньги", "рубль", "доллар", "бизнес", "компания", "рынок"],
            ["банк", "кредит", "вклад", "процент", "ипотека", "финансы", "счет"],
            ["инфляция", "цены", "рост", "падение", "курс", "валюты", "обмен"],
            ["инвестиции", "капитал", "прибыль", "убыток", "акции", "биржа", "трейдинг"],
            ["нефть", "газ", "ресурсы", "добыча", "энергия", "топливо", "экспорт"],
            ["технологии", "инновации", "стартап", "венчурный", "капитал", "разработка"],
            ["торговля", "магазин", "покупка", "продажа", "клиент", "услуга", "товар"],
            ["работа", "зарплата", "карьера", "профессия", "навыки", "образование"]
        ]
        
        # Наука и образование
        science_sentences = [
            ["наука", "исследование", "ученый", "открытие", "лаборатория", "эксперимент"],
            ["образование", "университет", "студент", "преподаватель", "лекция", "экзамен"],
            ["школа", "учитель", "ученик", "урок", "домашнее", "задание", "оценка"],
            ["технологии", "компьютер", "программирование", "алгоритм", "данные", "информация"],
            ["медицина", "врач", "больница", "лечение", "диагноз", "здоровье", "пациент"],
            ["математика", "числа", "формулы", "уравнения", "теорема", "доказательство"],
            ["физика", "атом", "энергия", "законы", "эксперимент", "исследование"],
            ["химия", "элементы", "реакции", "молекулы", "лаборатория", "опыты"]
        ]
        
        # Спорт
        sport_sentences = [
            ["спорт", "футбол", "хоккей", "игра", "команда", "соревнование", "победа"],
            ["олимпиада", "медаль", "чемпионат", "рекорд", "атлет", "тренировка"],
            ["баскетбол", "мяч", "корзина", "площадка", "команда", "очки"],
            ["теннис", "ракетка", "мяч", "сет", "подача", "турнир"],
            ["плавание", "бассейн", "вода", "стиль", "дистанция", "рекорд"],
            ["бокс", "ринг", "перчатки", "поединок", "нокаут", "чемпион"],
            ["автоспорт", "гонки", "трасса", "скорость", "пилот", "победа"],
            ["шахматы", "доска", "фигуры", "ход", "стратегия", "турнир"]
        ]
        
        # Объединяем все предложения
        all_sentences = (geo_sentences + politics_sentences + culture_sentences + 
                       economy_sentences + science_sentences + sport_sentences)
        
        # Добавляем вариации для увеличения словаря
        for sentence in all_sentences:
            sentences.append(sentence)
            # Создаем вариации с синонимами
            if "россия" in sentence:
                sentences.append([w.replace("россия", "родина") if w == "россия" else w for w in sentence])
            if "город" in sentence:
                sentences.append([w.replace("город", "мегаполис") if w == "город" else w for w in sentence])
            if "страна" in sentence:
                sentences.append([w.replace("страна", "государство") if w == "страна" else w for w in sentence])
        
        # Добавляем случайные комбинации для лучшей семантики
        for _ in range(500):
            base1 = random.choice(all_sentences)
            base2 = random.choice(all_sentences)
            if random.random() > 0.7:  # 30% chance to combine
                new_sentence = list(set(base1[:3] + base2[:3]))
                if len(new_sentence) >= 3:
                    sentences.append(new_sentence)
        
        return sentences
    
    def _create_combined_vocabulary(self):
        """Создание объединенного словаря из всех моделей"""
        all_words = set()
        
        # Собираем слова из всех моделей Word2Vec и FastText
        for model_name, model in self.models.items():
            if hasattr(model, 'wv') and hasattr(model.wv, 'key_to_index'):
                all_words.update(model.wv.key_to_index.keys())
        
        # Фильтруем слова
        self.vocab = [word for word in all_words 
                     if len(word) > 2 and word.isalpha() and not word.isdigit()]
        
        # Сортируем по алфавиту для удобства
        self.vocab.sort()
    
    def _create_fallback_vocabulary(self):
        """Создание резервного словаря"""
        self.vocab = [
            "россия", "москва", "санкт-петербург", "новосибирск", "екатеринбург", "казань",
            "нижний новгород", "челябинск", "самара", "омск", "ростов", "уфа", "красноярск",
            "пермь", "воронеж", "волгоград", "краснодар", "саратов", "тюмень", "ижевск",
            "барнаул", "ульяновск", "владивосток", "ярославль", "иркутск", "томск", "оренбург",
            "кемерово", "новокузнецк", "рязань", "астрахань", "пенза", "липецк", "киров",
            "чебоксары", "калининград", "курск", "тверь", "ставрополь", "магнитогорск",
            "сочи", "тула", "брянск", "белгород", "курган", "архангельск", "владимир",
            "севастополь", "симферополь", "сургут", "чебоксары", "вологда", "саранск",
            "чебоксары", "мурманск", "калуга", "орёл", "смоленск", "чита", "владикавказ",
            "якутск", "харьков", "киев", "минск", "астана", "ташкент", "баку", "ереван",
            "тбилиси", "вильнюс", "рига", "таллин", "варшава", "прага", "будапешт", "бухарест",
            "софия", "белград", "загреб", "сараево", "подгорица", "приштина", "кишинев",
            "кишинев", "токио", "пекин", "сеул", "бангкок", "джакарта", "манила", "ханой",
            "куала-лумпур", "сингапур", "тайбэй", "гонконг", "макао", "дебрейт", "катманду",
            "коломбо", "дакка", "исламабад", "кабул", "тегеран", "багдад", "эрь-рияд", "дубай",
            "каир", "рабат", "алжир", "тунис", "триполи", "хараре", "дакар", "лагос", "аккра",
            "найроби", "аддис-абеба", "антананариву", "кампала", "дарес-салам", "луанда",
            "киншаса", "абуджа", "бамако", "ужгород", "ужгород", "ужгород", "ужгород"
        ]
    
    def render_sidebar(self):
        """Боковая панель с настройками"""
        st.sidebar.title("🔍 Анализ векторных пространств")
        
        # Выбор модели
        st.sidebar.markdown("### Выбор модели")
        
        model_options = {
            "Word2Vec Skip-gram": "word2vec_sg",
            "Word2Vec CBOW": "word2vec_cbow", 
            "FastText Skip-gram": "fasttext_sg",
            "Doc2Vec PV-DM": "doc2vec_dm",
            "Doc2Vec PV-DBOW": "doc2vec_dbow"
        }
        
        selected_model_name = st.sidebar.selectbox(
            "Модель:",
            list(model_options.keys()),
            index=0
        )
        
        # Обновляем текущую модель
        model_key = model_options[selected_model_name]
        if model_key in self.models:
            self.current_model = self.models[model_key]
            self.current_model_name = selected_model_name
        
        if self.current_model:
            st.sidebar.success(f"✅ {self.current_model_name}")
            st.sidebar.info(f"Размерность: {self._get_model_vector_size()}D")
            st.sidebar.info(f"Слов в словаре: {len(self.vocab)}")
        else:
            st.sidebar.warning("⚠️ Модель не загружена")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Настройки визуализации")
        
        viz_method = st.sidebar.selectbox(
            "Метод визуализации:",
            ["t-SNE", "PCA", "UMAP"]
        )
        
        num_words = st.sidebar.slider(
            "Количество слов для визуализации:",
            min_value=50,
            max_value=1000,
            value=300
        )
        
        return viz_method, num_words
    
    def _get_model_vector_size(self) -> int:
        """Получение размерности векторов модели"""
        if not self.current_model:
            return 0
        
        if hasattr(self.current_model, 'vector_size'):
            return self.current_model.vector_size
        elif hasattr(self.current_model, 'wv') and hasattr(self.current_model.wv, 'vector_size'):
            return self.current_model.wv.vector_size
        else:
            return 0
    
    def _get_word_vector(self, word: str):
        """Получение вектора слова из текущей модели"""
        if not self.current_model:
            return None
        
        try:
            # Для Word2Vec и FastText
            if hasattr(self.current_model, 'wv'):
                return self.current_model.wv[word]
            # Для Doc2Vec (работа с документами)
            elif hasattr(self.current_model, 'dv'):
                return self.current_model.dv[word]
            else:
                return self.current_model[word]
        except:
            return None
    
    def _word_in_vocabulary(self, word: str) -> bool:
        """Проверка наличия слова в словаре текущей модели"""
        if not self.current_model:
            return False
        
        try:
            # Для Word2Vec и FastText
            if hasattr(self.current_model, 'wv') and hasattr(self.current_model.wv, 'key_to_index'):
                return word in self.current_model.wv.key_to_index
            # Для Doc2Vec
            elif hasattr(self.current_model, 'dv') and hasattr(self.current_model.dv, 'key_to_index'):
                return word in self.current_model.dv.key_to_index
            else:
                return word in self.current_model.key_to_index
        except:
            return False
    
    def _get_most_similar(self, word: str, topn: int = 10):
        """Поиск наиболее похожих слов"""
        if not self.current_model:
            return None
        
        try:
            # Для Word2Vec и FastText
            if hasattr(self.current_model, 'wv'):
                return self.current_model.wv.most_similar(word, topn=topn)
            # Для Doc2Vec
            elif hasattr(self.current_model, 'dv'):
                return self.current_model.dv.most_similar(word, topn=topn)
            else:
                return self.current_model.most_similar(word, topn=topn)
        except Exception as e:
            st.error(f"Ошибка поиска похожих слов: {e}")
            return None
    
    def _compute_similarity(self, word1: str, word2: str) -> float:
        """Вычисление семантического сходства"""
        if not self.current_model:
            raise Exception("Модель не загружена")
        
        try:
            # Для Word2Vec и FastText
            if hasattr(self.current_model, 'wv'):
                return self.current_model.wv.similarity(word1, word2)
            # Для Doc2Vec
            elif hasattr(self.current_model, 'dv'):
                return self.current_model.dv.similarity(word1, word2)
            else:
                return self.current_model.similarity(word1, word2)
        except Exception as e:
            raise Exception(f"Ошибка вычисления сходства: {e}")
    
    def render_vector_arithmetic(self):
        """Интерфейс векторной арифметики"""
        st.header("🧮 Векторная арифметика")
        
        if not self.current_model:
            st.warning("Модель не загружена. Невозможно выполнить вычисления.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Калькулятор аналогий")
            expression = st.text_input(
                "Введите выражение:",
                value="россия - москва + париж",
                help="Формат: слово1 - слово2 + слово3"
            )
            
            topn = st.slider("Количество результатов:", 1, 20, 10, key="arithmetic_topn")
            
            if st.button("Вычислить", type="primary", key="calc_btn"):
                if expression:
                    with st.spinner("Вычисление..."):
                        try:
                            result = self._compute_vector_arithmetic(expression, topn)
                            if result:
                                self._display_arithmetic_results(expression, result)
                            else:
                                st.error("Не удалось вычислить выражение")
                        except Exception as e:
                            st.error(f"Ошибка: {e}")
        
        with col2:
            st.markdown("### Примеры")
            examples = [
                "россия - москва + париж",
                "путин - россия + сша",
                "рубль - россия + доллар",
                "кино - россия + франция",
                "футбол - россия + бразилия"
            ]
            
            for example in examples:
                if st.button(example, key=f"ex_{hash(example)}"):
                    st.session_state.expression = example
    
    def _compute_vector_arithmetic(self, expression, topn=10):
        """Вычисление векторной арифметики"""
        if not self.current_model:
            return None
            
        parts = expression.split()
        
        # Проверяем корректный формат: слово1 - слово2 + слово3
        if len(parts) == 5 and parts[1] == '-' and parts[3] == '+':
            try:
                word1, word2, word3 = parts[0], parts[2], parts[4]
                
                # Проверяем наличие слов в словаре
                for word in [word1, word2, word3]:
                    if not self._word_in_vocabulary(word):
                        raise Exception(f"Слово '{word}' не найдено в словаре")
                
                # Вычисляем аналогию
                if hasattr(self.current_model, 'wv'):
                    result = self.current_model.wv.most_similar(
                        positive=[word3, word2],
                        negative=[word1],
                        topn=topn
                    )
                else:
                    # Для моделей без прямого метода most_similar
                    vec1 = self._get_word_vector(word1)
                    vec2 = self._get_word_vector(word2) 
                    vec3 = self._get_word_vector(word3)
                    
                    if vec1 is None or vec2 is None or vec3 is None:
                        raise Exception("Не удалось получить векторы слов")
                    
                    # Вычисляем: word3 + word2 - word1
                    result_vector = vec3 + vec2 - vec1
                    
                    # Ищем ближайшие векторы
                    if hasattr(self.current_model, 'wv'):
                        result = self.current_model.wv.similar_by_vector(result_vector, topn=topn)
                    else:
                        raise Exception("Модель не поддерживает векторные операции")
                
                return result
            except Exception as e:
                raise Exception(f"Ошибка вычисления: {e}")
        else:
            raise Exception("Неверный формат выражения. Используйте: слово1 - слово2 + слово3")
    
    def _display_arithmetic_results(self, expression, results):
        """Отображение результатов векторной арифметики"""
        st.success("✅ Результаты вычисления:")
        
        # Таблица результатов
        df = pd.DataFrame(results, columns=["Слово", "Сходство"])
        st.dataframe(df, use_container_width=True)
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 6))
        words = [word for word, score in results]
        scores = [score for word, score in results]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
        bars = ax.barh(words, scores, color=colors)
        
        ax.set_xlabel("Косинусное сходство")
        ax.set_title(f"Результаты: {expression}")
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        st.pyplot(fig)
    
    def render_semantic_similarity(self):
        """Интерфейс анализа семантического сходства"""
        st.header("📊 Анализ семантического сходства")
        
        if not self.current_model:
            st.warning("Модель не загружена. Невозможно вычислить сходство.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            word1 = st.text_input("Первое слово:", value="россия", key="sim_word1")
            word2 = st.text_input("Второе слово:", value="москва", key="sim_word2")
            
            if st.button("Вычислить сходство", key="sim_btn"):
                if word1 and word2:
                    try:
                        similarity = self._compute_similarity(word1, word2)
                        self._display_similarity_results(word1, word2, similarity)
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
        
        with col2:
            st.markdown("### Популярные пары")
            pairs = [
                ("россия", "москва"),
                ("кино", "фестиваль"),
                ("культура", "искусство"),
                ("путин", "президент"),
                ("футбол", "спорт"),
                ("экономика", "деньги")
            ]
            
            for w1, w2 in pairs:
                if st.button(f"{w1} - {w2}", key=f"pair_{w1}_{w2}"):
                    st.session_state.sim_word1 = w1
                    st.session_state.sim_word2 = w2
    
    def _display_similarity_results(self, word1, word2, similarity):
        """Отображение результатов сходства"""
        # Датчик сходства
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=similarity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Сходство: {word1} - {word2}"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightcoral"},
                    {'range': [0.3, 0.7], 'color': "lightyellow"},
                    {'range': [0.7, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Оценка сходства
        if similarity > 0.7:
            st.success("✅ Высокое семантическое сходство")
        elif similarity > 0.4:
            st.warning("⚠️ Умеренное семантическое сходство")
        else:
            st.error("❌ Низкое семантическое сходство")
    
    def render_nearest_neighbors(self):
        """Интерфейс поиска ближайших соседей"""
        st.header("🔍 Поиск похожих слов")
        
        if not self.current_model:
            st.warning("Модель не загружена. Невозможно найти похожие слова.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            word = st.text_input("Введите слово:", value="культура", key="neighbors_word")
            topn = st.slider("Количество соседей:", 1, 20, 10, key="neighbors_topn")
            
            if st.button("Найти похожие слова", key="neighbors_btn"):
                if word:
                    with st.spinner("Поиск..."):
                        try:
                            neighbors = self._find_nearest_neighbors(word, topn)
                            self._display_neighbors_results(word, neighbors)
                        except Exception as e:
                            st.error(f"Ошибка: {e}")
        
        with col2:
            st.markdown("### Примеры слов")
            test_words = ["культура", "искусство", "кино", "россия", "путин", "футбол", "экономика"]
            
            for test_word in test_words:
                if st.button(test_word, key=f"btn_{test_word}"):
                    st.session_state.neighbors_word = test_word
    
    def _find_nearest_neighbors(self, word, topn):
        """Поиск ближайших соседей"""
        if not self.current_model:
            raise Exception("Модель не загружена")
            
        if not self._word_in_vocabulary(word):
            raise Exception(f"Слово '{word}' не найдено в словаре")
        
        return self._get_most_similar(word, topn)
    
    def _display_neighbors_results(self, word, neighbors):
        """Отображение результатов поиска соседей"""
        if not neighbors:
            st.warning("Не найдено похожих слов")
            return
            
        st.success(f"✅ Слова, похожие на '{word}':")
        
        # Таблица результатов
        df = pd.DataFrame(neighbors, columns=["Слово", "Сходство"])
        st.dataframe(df, use_container_width=True)
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(12, 6))
        
        words = [word for word, score in neighbors]
        scores = [score for word, score in neighbors]
        y_pos = np.arange(len(words))
        
        bars = ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.set_xlabel("Косинусное сходство")
        ax.set_title(f"Слова, похожие на '{word}'")
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        st.pyplot(fig)

    def render_model_info(self):
        """Информация о моделях"""
        st.header("ℹ️ Информация о моделях")
        
        if not self.models:
            st.warning("Модели не загружены")
            return
        
        # Общая информация
        st.subheader("📊 Обзор всех моделей")
        
        model_info = []
        for model_name, model in self.models.items():
            info = {
                "Модель": model_name,
                "Тип": self._get_model_type(model_name),
                "Размерность": self._get_model_vector_size_for(model),
                "Архитектура": self._get_model_architecture(model_name),
                "Слов в словаре": len(self.vocab)
            }
            model_info.append(info)
        
        df_models = pd.DataFrame(model_info)
        st.dataframe(df_models, use_container_width=True)
        
        # Детальная информация о текущей модели
        st.subheader(f"🔍 Детали текущей модели: {self.current_model_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Размерность", f"{self._get_model_vector_size()}D")
            st.metric("Размер словаря", f"{len(self.vocab)}")
        
        with col2:
            st.metric("Архитектура", self._get_model_architecture(self.current_model_name))
            st.metric("Окно контекста", "8")
        
        with col3:
            st.metric("Эпох обучения", "50")
            st.metric("Min Count", "1")
        
        # Примеры работы текущей модели
        st.subheader("🔍 Примеры работы текущей модели")
        
        test_words = ["культура", "россия", "кино", "путин", "футбол", "экономика"]
        for word in test_words:
            if self._word_in_vocabulary(word):
                try:
                    similar = self._get_most_similar(word, topn=3)
                    if similar:
                        st.write(f"**{word}**: {[w for w, s in similar]}")
                    else:
                        st.write(f"**{word}**: ошибка получения похожих слов")
                except:
                    st.write(f"**{word}**: ошибка получения похожих слов")
    
    def _get_model_type(self, model_name: str) -> str:
        """Получение типа модели"""
        if "word2vec" in model_name.lower():
            return "Word2Vec"
        elif "fasttext" in model_name.lower():
            return "FastText"
        elif "doc2vec" in model_name.lower():
            return "Doc2Vec"
        else:
            return "Unknown"
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Получение архитектуры модели"""
        if "skip-gram" in model_name.lower() or "sg" in model_name.lower():
            return "Skip-gram"
        elif "cbow" in model_name.lower():
            return "CBOW"
        elif "dm" in model_name.lower():
            return "PV-DM"
        elif "dbow" in model_name.lower():
            return "PV-DBOW"
        else:
            return "Unknown"
    
    def _get_model_vector_size_for(self, model) -> int:
        """Получение размерности для конкретной модели"""
        if hasattr(model, 'vector_size'):
            return model.vector_size
        elif hasattr(model, 'wv') and hasattr(model.wv, 'vector_size'):
            return model.wv.vector_size
        else:
            return 0
    
    def render_dashboard(self):
        """Основной метод рендеринга дашборда"""
        st.set_page_config(
            page_title="Анализ векторных пространств",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 Анализ векторных пространств")
        st.markdown("Интерактивное исследование семантических представлений слов")
        
        # Боковая панель
        viz_method, num_words = self.render_sidebar()
        
        # Основное содержимое
        tab1, tab2, tab3 = st.tabs([
            "🧮 Векторная арифметика", 
            "🔍 Похожие слова",
            "📊 Семантическое сходство", 
        ])
        
        with tab1:
            self.render_vector_arithmetic()
        
        with tab2:
            self.render_nearest_neighbors()
        
        with tab3:
            self.render_semantic_similarity()

# Основное приложение Streamlit
def main():
    try:
        explorer = VectorSpaceExplorer()
        explorer.render_dashboard()
    except Exception as e:
        st.error(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main()