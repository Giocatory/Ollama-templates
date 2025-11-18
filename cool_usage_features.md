## 🚀 Продвинутые техники работы с моделями

### 1. **Ансамблирование моделей (Model Ensembling)**

```python
import ollama
import asyncio
from typing import List, Dict
import numpy as np

class ModelEnsemble:
    def __init__(self):
        self.models = [
            'llama3.1:8b',      # Для общих знаний
            'qwen2.5:7b',       # Для русского языка  
            'deepseek-r1:8b',   # Для рассуждений
            'codellama:7b'      # Для технических вопросов
        ]
    
    async def get_ensemble_response(self, prompt: str, strategy: str = "vote") -> str:
        """Получить ответ от ансамбля моделей"""
        responses = []
        
        for model in self.models:
            try:
                response = ollama.generate(model=model, prompt=prompt)
                responses.append({
                    'model': model,
                    'response': response['response'],
                    'confidence': self.calculate_confidence(response['response'])
                })
            except Exception as e:
                print(f"Ошибка в модели {model}: {e}")
        
        if strategy == "vote":
            return self.majority_vote(responses)
        elif strategy == "confidence":
            return self.highest_confidence(responses)
        elif strategy == "combined":
            return self.combine_responses(responses)
    
    def calculate_confidence(self, response: str) -> float:
        """Простая эвристика для оценки уверенности ответа"""
        confidence_indicators = [
            len(response) > 50,                    # Длинные ответы обычно увереннее
            '?' not in response[-10:],             # Отсутствие вопросов в конце
            not any(word in response.lower() for word in ['не знаю', 'возможно', 'наверное'])
        ]
        return sum(confidence_indicators) / len(confidence_indicators)
    
    def majority_vote(self, responses: List[Dict]) -> str:
        """Выбор наиболее частого ответа (упрощенно)"""
        # Здесь можно реализовать сложную логику голосования
        return max(responses, key=lambda x: x['confidence'])['response']
    
    def highest_confidence(self, responses: List[Dict]) -> str:
        """Выбор ответа с наибольшей уверенностью"""
        return max(responses, key=lambda x: x['confidence'])['response']
    
    def combine_responses(self, responses: List[Dict]) -> str:
        """Комбинирование лучших частей всех ответов"""
        combined = "На основе анализа нескольких моделей:\n\n"
        for i, resp in enumerate(sorted(responses, key=lambda x: x['confidence'], reverse=True), 1):
            combined += f"Модель {i} ({resp['model'].split(':')[0]}): {resp['response']}\n\n"
        return combined

# Использование
ensemble = ModelEnsemble()
response = asyncio.run(ensemble.get_ensemble_response("Объясни квантовую запутанность"))
print(response)
```

### 2. **Цепочки размышлений (Chain of Thought)**

```python
class AdvancedReasoning:
    def __init__(self, model: str = "deepseek-r1:8b"):
        self.model = model
    
    def complex_reasoning(self, problem: str) -> str:
        """Решение сложных задач через цепочку размышлений"""
        
        # Шаг 1: Анализ проблемы
        analysis_prompt = f"""
        Проанализируй следующую проблему и разбей её на подзадачи:
        {problem}
        
        Выведи только список подзадач без дополнительных объяснений.
        """
        
        analysis = ollama.generate(
            model=self.model,
            prompt=analysis_prompt,
            options={'temperature': 0.3}
        )
        
        # Шаг 2: Решение каждой подзадачи
        solution_prompt = f"""
        Проблема: {problem}
        
        Подзадачи для решения:
        {analysis['response']}
        
        Реши каждую подзададу последовательно, показывая ход мыслей.
        """
        
        solution = ollama.generate(
            model=self.model,
            prompt=solution_prompt,
            options={'temperature': 0.7}
        )
        
        # Шаг 3: Синтез финального ответа
        final_prompt = f"""
        На основе решения подзадач сформулируй итоговый ответ:
        
        Исходная проблема: {problem}
        Решение подзадач: {solution['response']}
        
        Дай краткий, но полный итоговый ответ.
        """
        
        final = ollama.generate(
            model=self.model,
            prompt=final_prompt,
            options={'temperature': 0.5}
        )
        
        return final['response']
    
    def socratic_dialogue(self, question: str, max_turns: int = 5) -> str:
        """Сократический диалог для глубокого понимания"""
        conversation = [{'role': 'user', 'content': question}]
        
        for turn in range(max_turns):
            # Модель задает уточняющие вопросы
            clarification = ollama.chat(
                model=self.model,
                messages=conversation + [{
                    'role': 'system', 
                    'content': 'Задай уточняющий вопрос чтобы лучше понять проблему. Не давай ответ сразу.'
                }]
            )
            
            clarification_question = clarification['message']['content']
            print(f"🤔 Уточняющий вопрос: {clarification_question}")
            
            # Пользователь отвечает (в реальном приложении - получаем от пользователя)
            user_answer = input("Ваш ответ: ")
            
            conversation.extend([
                {'role': 'assistant', 'content': clarification_question},
                {'role': 'user', 'content': user_answer}
            ])
        
        # Финальный ответ
        final_response = ollama.chat(
            model=self.model,
            messages=conversation + [{
                'role': 'system', 
                'content': 'Теперь дай развернутый итоговый ответ на основе всей полученной информации.'
            }]
        )
        
        return final_response['message']['content']

# Использование
reasoner = AdvancedReasoning()
result = reasoner.complex_reasoning("Как оптимизировать бизнес-процессы в стартапе?")
print(result)
```

### 3. **Динамическое переключение моделей**

```python
class SmartModelRouter:
    def __init__(self):
        self.model_map = {
            'coding': ['codellama:7b', 'deepseek-coder:6.7b'],
            'creative': ['llama3.1:8b', 'qwen2.5:7b'],
            'reasoning': ['deepseek-r1:8b', 'llama3.1:8b'],
            'russian': ['qwen2.5:7b', 'saiga:7b'],
            'math': ['wizardmath:7b', 'mathstral:7b'],
            'general': ['llama3.1:8b', 'qwen2.5:7b']
        }
    
    def detect_intent(self, message: str) -> str:
        """Определение намерения пользователя"""
        intent_prompt = f"""
        Определи тип запроса пользователя. Варианты: coding, creative, reasoning, russian, math, general.
        
        Сообщение: {message}
        
        Верни только одно слово - тип запроса.
        """
        
        response = ollama.generate(
            model='llama3.1:8b',
            prompt=intent_prompt,
            options={'temperature': 0.1}
        )
        
        intent = response['response'].strip().lower()
        return intent if intent in self.model_map else 'general'
    
    def get_best_model(self, intent: str, available_models: List[str] = None) -> str:
        """Выбор лучшей модели для задачи"""
        if available_models is None:
            available_models = self.get_available_models()
        
        candidates = [model for model in self.model_map[intent] if model in available_models]
        return candidates[0] if candidates else available_models[0]
    
    def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except:
            return ['llama3.1:8b']  # fallback
    
    def route_request(self, message: str) -> str:
        """Маршрутизация запроса к оптимальной модели"""
        intent = self.detect_intent(message)
        best_model = self.get_best_model(intent)
        
        print(f"🎯 Определен интент: {intent}, выбрана модель: {best_model}")
        
        response = ollama.generate(model=best_model, prompt=message)
        return f"🤖 [{best_model}]: {response['response']}"

# Использование
router = SmartModelRouter()
response = router.route_request("Напиши функцию Python для быстрой сортировки")
print(response)
```

## 🔧 Крутые идеи для проектов

### 4. **AI-репетитор с адаптивным обучением**

```python
class AITutor:
    def __init__(self, subject: str = "программирование"):
        self.subject = subject
        self.student_level = "beginner"
        self.learning_style = "практический"
        self.progress = []
    
    def assess_student_level(self, student_response: str) -> str:
        """Оценка уровня студента"""
        assessment_prompt = f"""
        Оцени уровень знаний студента по его ответу:
        
        Ответ студента: {student_response}
        Предмет: {self.subject}
        
        Варианты уровней: beginner, intermediate, advanced.
        Верни только уровень.
        """
        
        response = ollama.generate(
            model='deepseek-r1:8b',
            prompt=assessment_prompt,
            options={'temperature': 0.1}
        )
        
        self.student_level = response['response'].strip().lower()
        return self.student_level
    
    def generate_lesson(self, topic: str) -> dict:
        """Генерация адаптивного урока"""
        lesson_prompt = f"""
        Создай урок по теме: {topic}
        Уровень студента: {self.student_level}
        Стиль обучения: {self.learning_style}
        Предмет: {self.subject}
        
        Структура:
        1. Краткая теория
        2. Практический пример
        3. Упражнение для закрепления
        4. Вопрос для проверки понимания
        
        Верни в формате JSON.
        """
        
        response = ollama.generate(
            model='qwen2.5:7b',
            prompt=lesson_prompt,
            options={'temperature': 0.7}
        )
        
        # Парсинг JSON ответа
        try:
            import json
            lesson_data = json.loads(response['response'])
            return lesson_data
        except:
            return {
                'theory': response['response'],
                'example': 'Пример будет в следующем уроке',
                'exercise': 'Попрактикуйтесь с полученной теорией',
                'question': 'Что вы поняли из этого урока?'
            }
    
    def provide_feedback(self, student_answer: str, correct_answer: str) -> str:
        """Предоставление обратной связи"""
        feedback_prompt = f"""
        Дай конструктивную обратную связь студенту:
        
        Ответ студента: {student_answer}
        Правильный ответ: {correct_answer}
        Уровень студента: {self.student_level}
        
        Будь поддерживающим, укажи на ошибки и предложи улучшения.
        """
        
        response = ollama.generate(
            model='deepseek-r1:8b',
            prompt=feedback_prompt,
            options={'temperature': 0.8}
        )
        
        return response['response']

# Использование
tutor = AITutor("Python")
lesson = tutor.generate_lesson("функции в Python")
print(f"📚 Теория: {lesson['theory']}")
print(f"💡 Пример: {lesson['example']}")
```

### 5. **Интеллектуальный анализатор кода**

```python
class CodeAnalyzer:
    def __init__(self):
        self.metrics = {}
    
    def analyze_code_quality(self, code: str, language: str = "python") -> dict:
        """Анализ качества кода"""
        analysis_prompt = f"""
        Проанализируй следующий код на {language} и оцени:
        
        Код:
        ```{language}
        {code}
        ```
        
        Оцени по шкале 1-10:
        - Читаемость
        - Эффективность  
        - Соответствие best practices
        - Безопасность
        - Модульность
        
        Укажи конкретные проблемы и предложи улучшения.
        Верни в формате JSON.
        """
        
        response = ollama.generate(
            model='codellama:7b',
            prompt=analysis_prompt,
            options={'temperature': 0.3}
        )
        
        try:
            import json
            return json.loads(response['response'])
        except:
            return {'error': 'Не удалось проанализировать код'}
    
    def suggest_optimizations(self, code: str, language: str = "python") -> list:
        """Предложение оптимизаций"""
        optimization_prompt = f"""
        Предложи оптимизации для кода:
        
        ```{language}
        {code}
        ```
        
        Сфокусируйся на:
        - Производительности
        - Использовании памяти
        - Улучшении читаемости
        - Современных подходах
        
        Верни список конкретных предложений.
        """
        
        response = ollama.generate(
            model='deepseek-coder:6.7b',
            prompt=optimization_prompt,
            options={'temperature': 0.5}
        )
        
        return [suggestion.strip() for suggestion in response['response'].split('\n') if suggestion.strip()]
    
    def generate_tests(self, code: str, language: str = "python") -> str:
        """Генерация unit-тестов"""
        test_prompt = f"""
        Сгенерируй unit-тесты для следующего кода:
        
        ```{language}
        {code}
        ```
        
        Верни готовый код тестов с комментариями.
        """
        
        response = ollama.generate(
            model='codellama:7b',
            prompt=test_prompt,
            options={'temperature': 0.4}
        )
        
        return response['response']

# Использование
analyzer = CodeAnalyzer()
code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

quality_report = analyzer.analyze_code_quality(code)
optimizations = analyzer.suggest_optimizations(code)
tests = analyzer.generate_tests(code)

print("📊 Отчет о качестве:", quality_report)
print("⚡ Оптимизации:", optimizations)
print("🧪 Тесты:", tests)
```

### 6. **Персонализированный контент-генератор**

```python
class ContentGenerator:
    def __init__(self, brand_voice: str = "профессиональный", tone: str = "дружелюбный"):
        self.brand_voice = brand_voice
        self.tone = tone
        self.content_history = []
    
    def analyze_audience(self, audience_description: str) -> dict:
        """Анализ целевой аудитории"""
        audience_prompt = f"""
        Проанализируй целевую аудиторию: {audience_description}
        
        Определи:
        - Боли и потребности
        - Язык общения
        - Интересы
        - Уровень знаний
        
        Верни анализ в структурированном виде.
        """
        
        response = ollama.generate(
            model='qwen2.5:7b',
            prompt=audience_prompt,
            options={'temperature': 0.6}
        )
        
        return {'audience_analysis': response['response']}
    
    def generate_content_strategy(self, topic: str, platform: str, audience: dict) -> list:
        """Генерация контентной стратегии"""
        strategy_prompt = f"""
        Создай контентную стратегию для:
        Тема: {topic}
        Платформа: {platform}
        Аудитория: {audience['audience_analysis']}
        Голос бренда: {self.brand_voice}
        Тон: {self.tone}
        
        Предложи 5 идей контента с описанием.
        """
        
        response = ollama.generate(
            model='llama3.1:8b',
            prompt=strategy_prompt,
            options={'temperature': 0.8}
        )
        
        ideas = [idea.strip() for idea in response['response'].split('\n') if idea.strip()]
        return ideas[:5]
    
    def create_content_piece(self, idea: str, word_count: int = 500) -> str:
        """Создание конкретного контента"""
        content_prompt = f"""
        Напиши контент на основе идеи: {idea}
        
        Требования:
        - Голос бренда: {self.brand_voice}
        - Тон: {self.tone}
        - Объем: {word_count} слов
        - Структурированный текст
        - Призыв к действию
        
        Создай готовый к публикации материал.
        """
        
        response = ollama.generate(
            model='qwen2.5:7b',
            prompt=content_prompt,
            options={'temperature': 0.7, 'num_predict': 1000}
        )
        
        self.content_history.append({
            'idea': idea,
            'content': response['response'],
            'timestamp': '2024-01-01'  # В реальности использовать datetime
        })
        
        return response['response']
    
    def a_b_test_content(self, content_a: str, content_b: str, metric: str = "engagement") -> dict:
        """A/B тестирование контента"""
        test_prompt = f"""
        Проанализируй два варианта контента для A/B тестирования по метрике: {metric}
        
        Контент A:
        {content_a}
        
        Контент B:
        {content_b}
        
        Предскажи, какой контент покажет лучшие результаты и почему.
        """
        
        response = ollama.generate(
            model='deepseek-r1:8b',
            prompt=test_prompt,
            options={'temperature': 0.5}
        )
        
        return {
            'predicted_winner': 'A' if 'контент a' in response['response'].lower() else 'B',
            'analysis': response['response']
        }

# Использование
content_gen = ContentGenerator("инновационный", "вдохновляющий")
audience = content_gen.analyze_audience("стартаперы в IT, 25-35 лет")
strategy = content_gen.generate_content_strategy("искусственный интеллект", "LinkedIn", audience)

for i, idea in enumerate(strategy, 1):
    print(f"🎯 Идея {i}: {idea}")
    content = content_gen.create_content_piece(idea)
    print(f"📝 Контент: {content[:200]}...\n")
```

## 🎯 Дообучение и кастомизация

### 7. **Создание кастомных моделей через Modelfile**

```python
import subprocess
import tempfile
import os

class ModelCustomizer:
    def __init__(self):
        self.template_modelfile = """
FROM {base_model}

SYSTEM \"\"\"
{system_prompt}
\"\"\"

TEMPLATE \"\"\"{template}\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER num_predict {max_tokens}
"""
    
    def create_custom_model(self, 
                          model_name: str,
                          base_model: str,
                          system_prompt: str,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          max_tokens: int = 2048) -> bool:
        """Создание кастомной модели"""
        
        template = """{% for message in messages %}
{% if message['role'] == 'user' %}
### Пользователь: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
### Ассистент: {{ message['content'] }}
{% endif %}
{% endfor %}"""
        
        modelfile_content = self.template_modelfile.format(
            base_model=base_model,
            system_prompt=system_prompt,
            template=template,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Создание временного файла
        with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
            f.write(modelfile_content)
            temp_path = f.name
        
        try:
            # Создание модели через Ollama CLI
            result = subprocess.run([
                'ollama', 'create', model_name, '-f', temp_path
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ Модель {model_name} успешно создана!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка создания модели: {e}")
            return False
        finally:
            os.unlink(temp_path)
    
    def create_specialized_assistant(self, specialization: str, style: str = "профессиональный") -> str:
        """Создание специализированного ассистента"""
        
        specializations = {
            'legal': {
                'base_model': 'llama3.1:8b',
                'system_prompt': f'''Ты - опытный юрист. Отвечай на вопросы точно, ссылайся на законодательство.
                Стиль общения: {style}. Будь максимально точен в формулировках.'''
            },
            'medical': {
                'base_model': 'qwen2.5:7b', 
                'system_prompt': f'''Ты - медицинский консультант. Давай общие рекомендации, но напоминай о необходимости обращения к врачу.
                Стиль: {style}. Будь осторожен в рекомендациях.'''
            },
            'technical': {
                'base_model': 'codellama:7b',
                'system_prompt': f'''Ты - senior разработчик. Помогай с техническими вопросами, давай примеры кода.
                Стиль: {style}. Объясняй сложные концепции просто.'''
            }
        }
        
        if specialization not in specializations:
            raise ValueError(f"Специализация {specialization} не поддерживается")
        
        config = specializations[specialization]
        model_name = f"{specialization}-assistant-{style}"
        
        success = self.create_custom_model(
            model_name=model_name,
            base_model=config['base_model'],
            system_prompt=config['system_prompt']
        )
        
        return model_name if success else None

# Использование
customizer = ModelCustomizer()
legal_model = customizer.create_specialized_assistant('legal', 'формальный')
if legal_model:
    response = ollama.generate(model=legal_model, prompt="Какие документы нужны для открытия ООО?")
    print(response['response'])
```

### 8. **Контекстное обучение с few-shot примерами**

```python
class FewShotLearner:
    def __init__(self, base_model: str = "llama3.1:8b"):
        self.base_model = base_model
        self.examples = {}
    
    def add_examples(self, task_type: str, examples: List[dict]):
        """Добавление few-shot примеров"""
        if task_type not in self.examples:
            self.examples[task_type] = []
        
        self.examples[task_type].extend(examples)
    
    def generate_with_examples(self, task_type: str, new_input: str, max_examples: int = 3) -> str:
        """Генерация с использованием few-shot обучения"""
        if task_type not in self.examples or not self.examples[task_type]:
            # Если примеров нет, используем обычную генерацию
            return ollama.generate(model=self.base_model, prompt=new_input)['response']
        
        # Выбираем случайные примеры
        import random
        selected_examples = random.sample(self.examples[task_type], 
                                        min(max_examples, len(self.examples[task_type])))
        
        # Строим промпт с примерами
        prompt = "Вот несколько примеров выполнения задачи:\n\n"
        
        for i, example in enumerate(selected_examples, 1):
            prompt += f"Пример {i}:\n"
            prompt += f"Вход: {example['input']}\n"
            prompt += f"Выход: {example['output']}\n\n"
        
        prompt += f"Теперь выполни задачу для нового входа:\n"
        prompt += f"Вход: {new_input}\n"
        prompt += f"Выход:"
        
        response = ollama.generate(
            model=self.base_model,
            prompt=prompt,
            options={'temperature': 0.3}
        )
        
        return response['response']

# Использование
learner = FewShotLearner()

# Добавляем примеры для классификации тональности
sentiment_examples = [
    {
        'input': 'Этот продукт просто потрясающий!',
        'output': 'POSITIVE'
    },
    {
        'input': 'Ужасное качество, никогда больше не куплю',
        'output': 'NEGATIVE'  
    },
    {
        'input': 'Нормальный товар за свои деньги',
        'output': 'NEUTRAL'
    }
]

learner.add_examples('sentiment', sentiment_examples)

# Тестируем на новом тексте
result = learner.generate_with_examples(
    'sentiment', 
    'Качество хорошее, но доставка подвела'
)
print(f"Тональность: {result}")
```

## 🔮 Продвинутые концепции

### 9. **Мультиагентные системы**

```python
class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
    
    def create_agent(self, name: str, role: str, expertise: str, model: str = "llama3.1:8b"):
        """Создание AI-агента"""
        self.agents[name] = {
            'role': role,
            'expertise': expertise,
            'model': model,
            'memory': []
        }
    
    def agent_discussion(self, topic: str, participants: List[str]) -> str:
        """Организация дискуссии между агентами"""
        discussion = f"Тема дискуссии: {topic}\n\n"
        
        for round_num in range(3):  # 3 раунда дискуссии
            discussion += f"--- Раунд {round_num + 1} ---\n"
            
            for agent_name in participants:
                if agent_name not in self.agents:
                    continue
                
                agent = self.agents[agent_name]
                
                # Каждый агент получает контекст дискуссии
                prompt = f"""
                Ты - {agent['role']} с экспертизой в {agent['expertise']}.
                
                Контекст дискуссии:
                {discussion}
                
                Добавь свой вклад в дискуссию. Будь конкретен и опирайся на свою экспертизу.
                Твоя роль: {agent['role']}
                """
                
                response = ollama.generate(
                    model=agent['model'],
                    prompt=prompt,
                    options={'temperature': 0.8}
                )
                
                agent_contribution = response['response']
                discussion += f"{agent_name} ({agent['role']}): {agent_contribution}\n\n"
                
                # Сохраняем в память агента
                agent['memory'].append({
                    'round': round_num + 1,
                    'contribution': agent_contribution
                })
        
        # Финальный синтез
        synthesis_prompt = f"""
        На основе дискуссии сформулируй итоговый консенсус:
        
        {discussion}
        
        Выдели ключевые выводы и рекомендации.
        """
        
        final_response = ollama.generate(
            model='deepseek-r1:8b',
            prompt=synthesis_prompt,
            options={'temperature': 0.5}
        )
        
        return final_response['response']

# Использование
system = MultiAgentSystem()

# Создаем агентов
system.create_agent("tech_lead", "Технический руководитель", "архитектура систем", "codellama:7b")
system.create_agent("product_manager", "Продуктовый менеджер", "потребности пользователей", "llama3.1:8b") 
system.create_agent("ux_designer", "UX дизайнер", "пользовательский опыт", "qwen2.5:7b")

# Запускаем дискуссию
result = system.agent_discussion(
    "Разработка нового мобильного приложения для банкинга",
    ["tech_lead", "product_manager", "ux_designer"]
)

print("🤝 Результат дискуссии:")
print(result)
```

### 10. **Динамическая оценка качества ответов**

```python
class QualityEvaluator:
    def __init__(self):
        self.metrics = ['relevance', 'accuracy', 'completeness', 'clarity']
    
    def evaluate_response(self, question: str, response: str, context: str = "") -> dict:
        """Оценка качества ответа по нескольким метрикам"""
        
        evaluation_prompt = f"""
        Оцени ответ ассистента по следующим метрикам (1-10):
        
        Вопрос: {question}
        Контекст: {context}
        Ответ: {response}
        
        Метрики:
        - Relevance (релевантность ответа вопросу)
        - Accuracy (точность информации) 
        - Completeness (полнота ответа)
        - Clarity (ясность изложения)
        
        Верни оценку в формате JSON.
        """
        
        evaluation = ollama.generate(
            model='deepseek-r1:8b',
            prompt=evaluation_prompt,
            options={'temperature': 0.1}
        )
        
        try:
            import json
            scores = json.loads(evaluation['response'])
            
            # Добавляем итоговую оценку
            total_score = sum(scores.values()) / len(scores)
            scores['overall'] = round(total_score, 2)
            
            return scores
        except:
            return {metric: 5 for metric in self.metrics + ['overall']}
    
    def provide_feedback(self, question: str, response: str, scores: dict) -> str:
        """Генерация конструктивной обратной связи"""
        
        feedback_prompt = f"""
        На основе оценки ответа сгенерируй конструктивную обратную связь:
        
        Вопрос: {question}
        Ответ: {response}
        Оценки: {scores}
        
        Предложи конкретные улучшения для каждого аспекта.
        Будь поддерживающим и профессиональным.
        """
        
        feedback = ollama.generate(
            model='qwen2.5:7b',
            prompt=feedback_prompt,
            options={'temperature': 0.7}
        )
        
        return feedback['response']
    
    def track_improvement(self, evaluations: List[dict]) -> dict:
        """Отслеживание улучшения качества с течением времени"""
        if not evaluations:
            return {}
        
        improvement = {}
        for metric in self.metrics + ['overall']:
            first_score = evaluations[0].get(metric, 5)
            last_score = evaluations[-1].get(metric, 5)
            improvement[metric] = {
                'start': first_score,
                'current': last_score,
                'improvement': last_score - first_score,
                'trend': 'positive' if last_score > first_score else 'negative'
            }
        
        return improvement

# Использование
evaluator = QualityEvaluator()

question = "Что такое машинное обучение?"
response = "Машинное обучение - это подраздел искусственного интеллекта, который позволяет компьютерам обучаться на данных."

scores = evaluator.evaluate_response(question, response)
feedback = evaluator.provide_feedback(question, response, scores)

print("📊 Оценки:", scores)
print("💡 Обратная связь:", feedback)
```