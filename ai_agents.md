# Построение AI-агентов с Ollama: полное руководство

## 🏗️ Архитектура AI-агентов

### Базовый класс агента

```python
import ollama
from typing import Dict, List, Any, Callable
import json
import asyncio
from datetime import datetime
import uuid

class BaseAgent:
    """Базовый класс для всех AI-агентов"""
    
    def __init__(self, 
                 name: str,
                 role: str,
                 model: str = "llama3.1:8b",
                 system_prompt: str = "",
                 tools: List[Callable] = None):
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.tools = tools or []
        self.memory = []
        self.conversation_history = []
        self.agent_id = str(uuid.uuid4())[:8]
        
    def _default_system_prompt(self) -> str:
        return f"""Ты - {self.role}. Выполняй свои задачи профессионально и эффективно.
Всегда думай шаг за шагом. Если нужны дополнительные данные - запрашивай их.
Будь точным в своих ответах."""
    
    def add_memory(self, content: str, memory_type: str = "observation"):
        """Добавление в память агента"""
        self.memory.append({
            'timestamp': datetime.now().isoformat(),
            'type': memory_type,
            'content': content,
            'agent': self.name
        })
    
    def get_relevant_memory(self, query: str, limit: int = 5) -> List[str]:
        """Получение релевантных воспоминаний (упрощенная версия)"""
        # В реальной системе здесь бы использовалось векторное поисковое пространство
        return [m['content'] for m in self.memory[-limit:]]
    
    def think(self, prompt: str, use_memory: bool = True) -> str:
        """Процесс мышления агента"""
        
        # Собираем контекст
        context_parts = [self.system_prompt]
        
        if use_memory and self.memory:
            relevant_memories = self.get_relevant_memory(prompt)
            if relevant_memories:
                context_parts.append("Релевантные воспоминания:")
                context_parts.extend(relevant_memories)
        
        context_parts.append(f"\nТекущая задача: {prompt}")
        full_prompt = "\n".join(context_parts)
        
        # Генерируем ответ
        response = ollama.generate(
            model=self.model,
            prompt=full_prompt,
            options={'temperature': 0.7, 'num_predict': 1024}
        )
        
        # Сохраняем в память
        self.add_memory(f"Задача: {prompt}\nОтвет: {response['response']}", "thought")
        
        return response['response']
    
    def reflect(self, outcome: str) -> str:
        """Рефлексия агента над своими действиями"""
        reflection_prompt = f"""
        Проанализируй результат своих действий:
        
        Результат: {outcome}
        
        Что прошло хорошо? Что можно улучшить? Какие уроки можно извлечь?
        """
        
        reflection = self.think(reflection_prompt)
        self.add_memory(f"Рефлексия: {reflection}", "reflection")
        return reflection
    
    def communicate(self, message: str, recipient: 'BaseAgent' = None) -> str:
        """Коммуникация между агентами"""
        if recipient:
            comm_prompt = f"""
            Сообщение для {recipient.name} ({recipient.role}):
            {message}
            
            Сформулируй ответ, учитывая свою роль и экспертизу.
            """
            return recipient.think(comm_prompt)
        else:
            # Общее сообщение
            return self.think(f"Ответь на общее сообщение: {message}")
```

## 🤖 Специализированные агенты

### 1. **Агент-исследователь**

```python
class ResearchAgent(BaseAgent):
    """Агент для исследования и сбора информации"""
    
    def __init__(self, name: str = "Исследователь", domains: List[str] = None):
        super().__init__(
            name=name,
            role="AI исследователь и аналитик",
            model="qwen2.5:7b",
            system_prompt="""Ты - опытный исследователь. Твоя задача - находить, анализировать и синтезировать информацию.
            Всегда проверяй факты, будь критичным к источникам и предоставляй структурированные выводы.
            Используй различные методы анализа и всегда указывай уровень достоверности информации."""
        )
        self.domains = domains or ["наука", "технологии", "бизнес"]
        self.sources = []
    
    def research_topic(self, topic: str, depth: str = "базовый") -> Dict:
        """Исследование темы с заданной глубиной"""
        research_prompt = f"""
        Проведи исследование по теме: {topic}
        Глубина исследования: {depth}
        Домены экспертизы: {', '.join(self.domains)}
        
        Структура ответа:
        1. Ключевые факты
        2. Основные тенденции
        3. Противоречия и дебаты
        4. Будущие перспективы
        5. Рекомендации для дальнейшего изучения
        
        Будь максимально объективным и точным.
        """
        
        research_result = self.think(research_prompt)
        
        # Анализ достоверности
        credibility_analysis = self.analyze_credibility(research_result)
        
        result = {
            'topic': topic,
            'depth': depth,
            'content': research_result,
            'credibility_score': credibility_analysis['score'],
            'confidence_level': credibility_analysis['level'],
            'timestamp': datetime.now().isoformat(),
            'agent': self.name
        }
        
        self.add_memory(f"Исследование: {topic} - {credibility_analysis['level']}", "research")
        return result
    
    def analyze_credibility(self, content: str) -> Dict:
        """Анализ достоверности информации"""
        analysis_prompt = f"""
        Проанализируй достоверность следующей информации:
        
        {content}
        
        Оцени по шкале 1-10:
        - Фактическая точность
        - Объективность
        - Полнота информации
        - Наличие подтверждающих данных
        
        Верни оценку и уровень достоверности (низкий/средний/высокий).
        """
        
        response = self.think(analysis_prompt)
        
        # Упрощенный парсинг ответа
        if "высок" in response.lower():
            score = 8
            level = "высокий"
        elif "средн" in response.lower():
            score = 6
            level = "средний"
        else:
            score = 4
            level = "низкий"
            
        return {'score': score, 'level': level, 'analysis': response}
    
    def compare_sources(self, topic: str, perspectives: List[str]) -> Dict:
        """Сравнение различных точек зрения"""
        compare_prompt = f"""
        Сравни различные точки зрения на тему: {topic}
        
        Перспективы для сравнения:
        {chr(10).join([f'{i+1}. {p}' for i, p in enumerate(perspectives)])}
        
        Проанализируй:
        - Общие моменты
        - Ключевые различия
        - Сильные и слабые стороны каждой позиции
        - Возможности синтеза
        
        Представь анализ в структурированном виде.
        """
        
        comparison = self.think(compare_prompt)
        return {'topic': topic, 'comparison': comparison, 'perspectives': perspectives}
```

### 2. **Агент-аналитик**

```python
class AnalyticsAgent(BaseAgent):
    """Агент для анализа данных и выявления закономерностей"""
    
    def __init__(self, name: str = "Аналитик"):
        super().__init__(
            name=name,
            role="Старший аналитик данных",
            model="deepseek-r1:8b",
            system_prompt="""Ты - опытный аналитик данных. Твоя задача - находить закономерности, 
            строить гипотезы и предоставлять data-driven инсайты.
            Всегда используй логические цепочки, проверяй гипотезы и визуализируй выводы там, где это возможно."""
        )
        self.analysis_methods = [
            "статистический анализ",
            "тренд-анализ", 
            "корреляционный анализ",
            "причинно-следственный анализ",
            "прогнозное моделирование"
        ]
    
    def analyze_dataset(self, data_description: str, objectives: List[str]) -> Dict:
        """Анализ набора данных"""
        analysis_prompt = f"""
        Проанализируй следующий набор данных:
        
        Описание данных: {data_description}
        
        Цели анализа: {', '.join(objectives)}
        
        Доступные методы: {', '.join(self.analysis_methods)}
        
        Предоставь:
        1. Выбор методов анализа
        2. Ключевые метрики
        3. Обнаруженные закономерности
        4. Практические инсайты
        5. Рекомендации для действий
        
        Будь конкретным и используй data-driven подход.
        """
        
        analysis_result = self.think(analysis_prompt)
        
        # Генерация гипотез
        hypotheses = self.generate_hypotheses(analysis_result)
        
        return {
            'objectives': objectives,
            'analysis': analysis_result,
            'hypotheses': hypotheses,
            'recommendations': self.extract_recommendations(analysis_result)
        }
    
    def generate_hypotheses(self, analysis: str) -> List[str]:
        """Генерация гипотез на основе анализа"""
        hypothesis_prompt = f"""
        На основе анализа сгенерируй 3 проверяемые гипотезы:
        
        Анализ: {analysis}
        
        Гипотезы должны быть:
        - Конкретными и измеримыми
        - Проверяемыми на данных
        - Практически значимыми
        
        Верни только список гипотез.
        """
        
        response = self.think(hypothesis_prompt)
        return [h.strip() for h in response.split('\n') if h.strip() and h.strip()[0].isdigit()]
    
    def extract_recommendations(self, analysis: str) -> List[str]:
        """Извлечение рекомендаций из анализа"""
        rec_prompt = f"""
        Извлеки практические рекомендации из анализа:
        
        {analysis}
        
        Верни список конкретных, выполнимых рекомендаций.
        """
        
        response = self.think(rec_prompt)
        return [r.strip() for r in response.split('\n') if r.strip() and any(marker in r for marker in ['•', '-', '—'])]
    
    def predictive_analysis(self, current_state: str, timeframe: str = "краткосрочный") -> Dict:
        """Прогнозный анализ"""
        predict_prompt = f"""
        Проведи прогнозный анализ:
        
        Текущее состояние: {current_state}
        Временной горизонт: {timeframe}
        
        Рассмотри:
        - Вероятные сценарии развития
        - Ключевые драйверы изменений
        - Потенциальные риски
        - Возможности для вмешательства
        
        Оцени вероятность каждого сценария.
        """
        
        prediction = self.think(predict_prompt)
        return {
            'timeframe': timeframe,
            'current_state': current_state,
            'predictions': prediction,
            'timestamp': datetime.now().isoformat()
        }
```

### 3. **Агент-творец**

```python
class CreativeAgent(BaseAgent):
    """Агент для творческих задач и генерации контента"""
    
    def __init__(self, name: str = "Творец", style: str = "инновационный"):
        super().__init__(
            name=name,
            role="Креативный директор и создатель контента",
            model="llama3.1:8b",
            system_prompt=f"""Ты - творческий гений со стилем: {style}. 
            Твоя задача - создавать оригинальный, вдохновляющий контент и инновационные идеи.
            Будь смелым в своих творческих подходах, но сохраняй практическую применимость."""
        )
        self.style = style
        self.creative_methods = [
            "мозговой штурм",
            "латеральное мышление", 
            "синектика",
            "SCAMPER",
            "шесть шляп мышления"
        ]
    
    def brainstorm_ideas(self, topic: str, constraints: List[str] = None) -> Dict:
        """Генерация идей через мозговой штурм"""
        constraints = constraints or []
        
        brainstorm_prompt = f"""
        Проведи мозговой штурм на тему: {topic}
        
        Ограничения: {', '.join(constraints) if constraints else 'нет'}
        Стиль: {self.style}
        Методы: {', '.join(self.creative_methods)}
        
        Сгенерируй:
        - 5 радикальных идей
        - 5 практичных идей  
        - 5 инновационных идей
        - 1 прорывную идею
        
        Для каждой идеи укажи:
        - Краткое описание
        - Потенциальное влияние
        - Сложность реализации
        """
        
        ideas = self.think(brainstorm_prompt)
        
        # Оценка идей
        evaluation = self.evaluate_ideas(ideas)
        
        return {
            'topic': topic,
            'constraints': constraints,
            'ideas': ideas,
            'evaluation': evaluation,
            'best_idea': evaluation.get('best_idea', '')
        }
    
    def evaluate_ideas(self, ideas: str) -> Dict:
        """Оценка сгенерированных идей"""
        eval_prompt = f"""
        Оцени сгенерированные идеи по критериям:
        - Инновационность (1-10)
        - Практическая реализуемость (1-10)
        - Потенциальное влияние (1-10)
        - Соответствие бренду (1-10)
        
        Идеи для оценки:
        {ideas}
        
        Выбери лучшую идею и обоснуй выбор.
        """
        
        evaluation = self.think(eval_prompt)
        
        # Упрощенный парсинг оценок
        scores = {}
        lines = evaluation.split('\n')
        for line in lines:
            if 'инновационность' in line.lower():
                scores['innovation'] = self._extract_score(line)
            elif 'реализуемость' in line.lower():
                scores['feasibility'] = self._extract_score(line)
            elif 'влияние' in line.lower():
                scores['impact'] = self._extract_score(line)
        
        return {'scores': scores, 'detailed_evaluation': evaluation}
    
    def _extract_score(self, text: str) -> int:
        """Извлечение числовой оценки из текста"""
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 5
    
    def create_content(self, 
                     content_type: str, 
                     topic: str, 
                     target_audience: str,
                     tone: str = "профессиональный") -> Dict:
        """Создание контента различных типов"""
        content_prompt = f"""
        Создай {content_type} на тему: {topic}
        
        Целевая аудитория: {target_audience}
        Тон: {tone}
        Стиль: {self.style}
        
        Требования:
        - Соответствие бренду
        - Вовлекающий характер
        - Четкая структура
        - Призыв к действию
        
        Создай готовый к использованию контент.
        """
        
        content = self.think(content_prompt)
        
        return {
            'type': content_type,
            'topic': topic,
            'audience': target_audience,
            'tone': tone,
            'content': content,
            'length': len(content),
            'created_at': datetime.now().isoformat()
        }
```

### 4. **Агент-критик**

```python
class CriticAgent(BaseAgent):
    """Агент для критического анализа и оценки"""
    
    def __init__(self, name: str = "Критик"):
        super().__init__(
            name=name,
            role="Старший критик и рецензент",
            model="deepseek-r1:8b",
            system_prompt="""Ты - проницательный критик. Твоя задача - объективно оценивать работы, 
            находить слабые места и предлагать конструктивные улучшения.
            Будь честным, но справедливым. Критикуй идеи, а не людей."""
        )
        self.evaluation_framework = {
            'content': ['актуальность', 'глубина', 'оригинальность'],
            'structure': ['логика', 'организация', 'связность'],
            'style': ['ясность', 'выразительность', 'соответствие аудитории'],
            'impact': ['практическая ценность', 'эмоциональное воздействие', 'долгосрочная значимость']
        }
    
    def critical_review(self, work: str, context: str = "") -> Dict:
        """Критический обзор работы"""
        review_prompt = f"""
        Проведи критический обзор следующей работы:
        
        Работа: {work}
        Контекст: {context}
        
        Используй framework оценки:
        {json.dumps(self.evaluation_framework, ensure_ascii=False, indent=2)}
        
        Предоставь:
        1. Общую оценку
        2. Сильные стороны
        3. Области для улучшения
        4. Конкретные рекомендации
        5. Итоговый вердикт
        
        Будь конструктивным и конкретным.
        """
        
        review = self.think(review_prompt)
        
        # Извлечение оценки
        rating = self._extract_rating(review)
        
        return {
            'work_preview': work[:200] + "...",
            'review': review,
            'rating': rating,
            'recommendations': self._extract_recommendations(review)
        }
    
    def _extract_rating(self, review: str) -> float:
        """Извлечение числовой оценки из обзора"""
        import re
        # Ищем оценки в формате X/10 или X из 10
        patterns = [r'(\d+)\s*\/\s*10', r'(\d+)\s*из\s*10', r'оценка\s*[:\-]\s*(\d+)']
        
        for pattern in patterns:
            matches = re.findall(pattern, review.lower())
            if matches:
                return min(10, max(1, int(matches[0])))
        
        return 7.0  # Средняя оценка по умолчанию
    
    def _extract_recommendations(self, review: str) -> List[str]:
        """Извлечение рекомендаций из обзора"""
        lines = review.split('\n')
        recommendations = []
        
        for line in lines:
            line_lower = line.lower()
            if any(marker in line_lower for marker in ['рекомендация', 'совет', 'улучшить', 'стоит']):
                if len(line.strip()) > 20:  # Достаточно длинная для содержательной рекомендации
                    recommendations.append(line.strip())
        
        return recommendations[:5]  # Возвращаем до 5 рекомендаций
    
    def compare_works(self, works: List[Dict]) -> Dict:
        """Сравнение нескольких работ"""
        works_text = "\n\n".join([f"Работа {i+1}:\n{w['content']}\nКонтекст: {w.get('context', '')}" 
                                for i, w in enumerate(works)])
        
        compare_prompt = f"""
        Сравни следующие работы:
        
        {works_text}
        
        Проанализируй:
        - Относительные достоинства и недостатки
        - Уникальные особенности каждой работы
        - Соответствие целям и аудитории
        - Рекомендации по выбору
        
        Представь сравнение в табличной форме.
        """
        
        comparison = self.think(compare_prompt)
        
        # Оценка каждой работы
        ratings = {}
        for i, work in enumerate(works):
            review = self.critical_review(work['content'], work.get('context', ''))
            ratings[f"work_{i+1}"] = review['rating']
        
        return {
            'comparison': comparison,
            'ratings': ratings,
            'best_work': max(ratings, key=ratings.get)
        }
```

## 🏢 Система управления агентами

```python
class AgentOrchestrator:
    """Оркестратор для управления множеством агентов"""
    
    def __init__(self):
        self.agents = {}
        self.workflows = {}
        self.communication_log = []
    
    def register_agent(self, agent: BaseAgent):
        """Регистрация агента в системе"""
        self.agents[agent.name] = agent
        print(f"✅ Агент {agent.name} зарегистрирован как {agent.role}")
    
    def create_workflow(self, name: str, steps: List[Dict]):
        """Создание рабочего процесса"""
        self.workflows[name] = {
            'steps': steps,
            'created_at': datetime.now().isoformat(),
            'executions': 0
        }
    
    def execute_workflow(self, workflow_name: str, initial_input: str) -> Dict:
        """Выполнение рабочего процесса"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} не найден")
        
        workflow = self.workflows[workflow_name]
        workflow['executions'] += 1
        
        current_input = initial_input
        results = {}
        
        print(f"🚀 Запуск workflow: {workflow_name}")
        
        for step in workflow['steps']:
            agent_name = step['agent']
            task = step['task']
            
            if agent_name not in self.agents:
                print(f"⚠️ Агент {agent_name} не найден, пропускаем шаг")
                continue
            
            agent = self.agents[agent_name]
            print(f"🔧 {agent.name} выполняет: {task}")
            
            # Выполнение задачи агентом
            result = agent.think(f"{task}\n\nВходные данные: {current_input}")
            results[step['name']] = result
            current_input = result
            
            # Логирование
            self.communication_log.append({
                'timestamp': datetime.now().isoformat(),
                'workflow': workflow_name,
                'step': step['name'],
                'agent': agent_name,
                'input': current_input,
                'output': result
            })
        
        print(f"✅ Workflow {workflow_name} завершен")
        return results
    
    def facilitate_discussion(self, topic: str, participant_agents: List[str], rounds: int = 3) -> Dict:
        """Фасилитация дискуссии между агентами"""
        print(f"💬 Начинаем дискуссию: {topic}")
        
        discussion_log = []
        current_topic = topic
        
        for round_num in range(rounds):
            print(f"🔄 Раунд {round_num + 1}")
            round_log = {'round': round_num + 1, 'contributions': []}
            
            for agent_name in participant_agents:
                if agent_name not in self.agents:
                    continue
                
                agent = self.agents[agent_name]
                
                # Агент вносит свой вклад
                prompt = f"""
                Тема дискуссии: {current_topic}
                
                Текущий раунд: {round_num + 1}
                Твоя роль: {agent.role}
                
                Внеси свой вклад в дискуссию. Учитывай предыдущие высказывания и продвигай обсуждение вперед.
                """
                
                contribution = agent.think(prompt)
                round_log['contributions'].append({
                    'agent': agent_name,
                    'role': agent.role,
                    'contribution': contribution
                })
                
                print(f"   {agent.name}: {contribution[:100]}...")
                
                # Обновляем тему для следующего агента
                current_topic = contribution
            
            discussion_log.append(round_log)
        
        # Синтез результатов
        synthesis_prompt = f"""
        Синтезируй результаты дискуссии:
        
        Исходная тема: {topic}
        Участники: {', '.join(participant_agents)}
        
        Лог дискуссии: {json.dumps(discussion_log, ensure_ascii=False)}
        
        Предоставь:
        - Ключевые выводы
        - Области согласия
        - Оставшиеся разногласия  
        - Рекомендации для дальнейших действий
        """
        
        # Используем первого агента для синтеза
        synthesizer = self.agents[participant_agents[0]]
        synthesis = synthesizer.think(synthesis_prompt)
        
        return {
            'topic': topic,
            'participants': participant_agents,
            'rounds': rounds,
            'discussion_log': discussion_log,
            'synthesis': synthesis
        }
    
    def get_system_status(self) -> Dict:
        """Получение статуса системы агентов"""
        return {
            'total_agents': len(self.agents),
            'agent_names': list(self.agents.keys()),
            'total_workflows': len(self.workflows),
            'workflow_names': list(self.workflows.keys()),
            'total_communications': len(self.communication_log),
            'system_uptime': datetime.now().isoformat()
        }
```

## 🎯 Практические примеры использования

### Пример 1: Система для создания контента

```python
def setup_content_creation_system():
    """Настройка системы для создания контента"""
    orchestrator = AgentOrchestrator()
    
    # Создаем агентов
    researcher = ResearchAgent("Исследователь-контента")
    analyst = AnalyticsAgent("Аналитик-трендов")
    creator = CreativeAgent("Главный-редактор", "инновационный")
    critic = CriticAgent("Редактор-критик")
    
    # Регистрируем агентов
    orchestrator.register_agent(researcher)
    orchestrator.register_agent(analyst)
    orchestrator.register_agent(creator)
    orchestrator.register_agent(critic)
    
    # Создаем workflow для создания статьи
    orchestrator.create_workflow("content_creation", [
        {
            'name': 'research',
            'agent': 'Исследователь-контента',
            'task': 'Проведи исследование по теме и собери ключевые факты'
        },
        {
            'name': 'analysis', 
            'agent': 'Аналитик-трендов',
            'task': 'Проанализируй собранные данные и выяви ключевые тренды'
        },
        {
            'name': 'creation',
            'agent': 'Главный-редактор', 
            'task': 'Напиши статью на основе исследования и анализа'
        },
        {
            'name': 'review',
            'agent': 'Редактор-критик',
            'task': 'Проверить статью и предложить улучшения'
        }
    ])
    
    return orchestrator

# Использование
orchestrator = setup_content_creation_system()

# Запуск workflow
topic = "Влияние искусственного интеллекта на современное образование"
results = orchestrator.execute_workflow("content_creation", topic)

print("📊 Результаты создания контента:")
for step_name, result in results.items():
    print(f"\n--- {step_name} ---")
    print(result[:500] + "...")
```

### Пример 2: Система принятия решений

```python
def setup_decision_system():
    """Настройка системы для принятия решений"""
    orchestrator = AgentOrchestrator()
    
    # Создаем агентов с разными перспективами
    optimist = BaseAgent("Оптимист", "Оптимистичный аналитик", 
                        system_prompt="Фокусируйся на возможностях и позитивных аспектах")
    
    pessimist = BaseAgent("Пессимист", "Критически настроенный аналитик",
                         system_prompt="Выявляй риски и потенциальные проблемы") 
    
    realist = BaseAgent("Реалист", "Прагматичный аналитик",
                       system_prompt="Балансируй между возможностями и рисками")
    
    strategist = BaseAgent("Стратег", "Стратегический планировщик",
                          system_prompt="Разрабатывай долгосрочные стратегии")
    
    # Регистрируем агентов
    orchestrator.register_agent(optimist)
    orchestrator.register_agent(pessimist) 
    orchestrator.register_agent(realist)
    orchestrator.register_agent(strategist)
    
    return orchestrator

# Использование
decision_system = setup_decision_system()

# Дискуссия по важному решению
decision_topic = "Стоит ли инвестировать в разработку нового AI-продукта в 2024?"
discussion = decision_system.facilitate_discussion(
    decision_topic,
    ["Оптимист", "Пессимист", "Реалист", "Стратег"],
    rounds=2
)

print("🎯 Результаты дискуссии:")
print(discussion['synthesis'])
```

### Пример 3: Агент с инструментами

```python
class ToolEnhancedAgent(BaseAgent):
    """Агент с расширенными инструментами"""
    
    def __init__(self, name: str, role: str):
        super().__init__(name, role)
        self.tools = {
            'calculator': self.calculate,
            'web_search': self.simulate_web_search,
            'data_analyzer': self.analyze_data,
            'code_executor': self.execute_code
        }
    
    def calculate(self, expression: str) -> str:
        """Калькулятор (упрощенная версия)"""
        try:
            # Безопасное вычисление
            safe_dict = {'__builtins__': None}
            result = eval(expression, safe_dict, {})
            return f"Результат: {result}"
        except:
            return "Ошибка вычисления"
    
    def simulate_web_search(self, query: str) -> str:
        """Имитация поиска в интернете"""
        search_prompt = f"""
        Представь, что ты ищешь информацию в интернете по запросу: {query}
        
        Верни 3 наиболее релевантных результата с кратким описанием.
        """
        
        return self.think(search_prompt)
    
    def analyze_data(self, data_description: str) -> str:
        """Анализ данных"""
        analysis_prompt = f"""
        Проанализируй следующие данные:
        {data_description}
        
        Предоставь основные инсайты и закономерности.
        """
        
        return self.think(analysis_prompt)
    
    def execute_code(self, code: str, language: str = "python") -> str:
        """Имитация выполнения кода"""
        execution_prompt = f"""
        Проанализируй следующий код на {language}:
        
        ```{language}
        {code}
        ```
        
        Предскажи, что будет выведено при выполнении этого кода, и есть ли в нем ошибки.
        """
        
        return self.think(execution_prompt)
    
    def use_tool(self, tool_name: str, input_data: str) -> str:
        """Использование инструмента"""
        if tool_name in self.tools:
            return self.tools[tool_name](input_data)
        else:
            return f"Инструмент {tool_name} не найден"

# Использование
enhanced_agent = ToolEnhancedAgent("Технический-ассистент", "Помощник с инструментами")

# Использование различных инструментов
calculation = enhanced_agent.use_tool('calculator', '2 + 2 * 2')
search_result = enhanced_agent.use_tool('web_search', 'новейшие разработки в AI')
code_analysis = enhanced_agent.use_tool('code_executor', 'print("Hello, World!")')

print("🧮 Результаты работы инструментов:")
print(f"Калькулятор: {calculation}")
print(f"Поиск: {search_result[:200]}...")
print(f"Анализ кода: {code_analysis[:200]}...")
```

## 🔄 Продвинутые паттерны взаимодействия

### 1. **Рекурсивное улучшение**

```python
class RecursiveImprovementAgent(BaseAgent):
    """Агент для рекурсивного улучшения результатов"""
    
    def recursive_improve(self, 
                         initial_input: str,
                         improvement_criteria: str,
                         max_iterations: int = 5,
                         quality_threshold: float = 0.8) -> Dict:
        """Рекурсивное улучшение результата"""
        
        current_result = self.think(initial_input)
        iterations = 0
        improvement_history = []
        
        while iterations < max_iterations:
            iterations += 1
            
            # Оценка текущего результата
            evaluation = self.evaluate_result(current_result, improvement_criteria)
            
            if evaluation['score'] >= quality_threshold:
                break
            
            # Генерация улучшений
            improvement_suggestions = self.generate_improvements(
                current_result, evaluation, improvement_criteria
            )
            
            # Применение улучшений
            improved_result = self.apply_improvements(current_result, improvement_suggestions)
            
            improvement_history.append({
                'iteration': iterations,
                'previous_score': evaluation['score'],
                'improvements': improvement_suggestions,
                'new_result': improved_result
            })
            
            current_result = improved_result
        
        final_evaluation = self.evaluate_result(current_result, improvement_criteria)
        
        return {
            'final_result': current_result,
            'final_score': final_evaluation['score'],
            'iterations': iterations,
            'improvement_history': improvement_history,
            'quality_achieved': final_evaluation['score'] >= quality_threshold
        }
    
    def evaluate_result(self, result: str, criteria: str) -> Dict:
        """Оценка результата по критериям"""
        eval_prompt = f"""
        Оцени результат по шкале 0-1 по следующим критериям:
        {criteria}
        
        Результат для оценки:
        {result}
        
        Верни оценку и краткое обоснование.
        """
        
        evaluation = self.think(eval_prompt)
        
        # Упрощенное извлечение оценки
        import re
        numbers = re.findall(r'0\.\d+|\d+\.\d+', evaluation)
        score = float(numbers[0]) if numbers else 0.5
        
        return {'score': score, 'reasoning': evaluation}
    
    def generate_improvements(self, result: str, evaluation: Dict, criteria: str) -> List[str]:
        """Генерация предложений по улучшению"""
        improve_prompt = f"""
        На основе оценки сгенерируй конкретные предложения по улучшению:
        
        Текущий результат: {result}
        Оценка: {evaluation['score']}/1
        Обоснование оценки: {evaluation['reasoning']}
        Критерии улучшения: {criteria}
        
        Предложи 3 конкретных улучшения.
        """
        
        improvements = self.think(improve_prompt)
        return [imp.strip() for imp in improvements.split('\n') if imp.strip()]
    
    def apply_improvements(self, result: str, improvements: List[str]) -> str:
        """Применение улучшений к результату"""
        apply_prompt = f"""
        Примени следующие улучшения к результату:
        
        Исходный результат: {result}
        
        Улучшения для применения:
        {chr(10).join(improvements)}
        
        Верни улучшенную версию результата.
        """
        
        return self.think(apply_prompt)
```