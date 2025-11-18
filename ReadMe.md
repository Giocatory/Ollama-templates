## 🛠️ Предварительная настройка и установка

Перед началом работы убедитесь, что Ollama установлена и запущена на вашем компьютере.

### Установка Ollama
Скачайте и установите Ollama с официального сайта или используйте команды для вашей ОС :
```bash
# На macOS (через Homebrew):
brew install ollama

# На Linux (используя официальный скрипт установки):
curl -sS https://ollama.ai/install.sh | bash
```

### Запуск Ollama
После установки запустите Ollama сервер:
```bash
ollama serve
```

### Установка Python-библиотеки
Установите официальную Python-библиотеку Ollama :
```bash
pip install ollama
```

### Проверка установки
Убедитесь, что Ollama работает, и загрузите модель :
```bash
# Проверить список доступных моделей
ollama list

# Загрузить модель (например, llama3.2)
ollama pull llama3.2:1b
```

## 📟 Основные способы работы с Ollama из Python

### Способ 1: Использование официальной Python-библиотеки

#### Простая генерация текста
```python
import ollama

# Базовая генерация
response = ollama.generate(model='llama3.2:1b', prompt='Почему небо синее?')
print(response['response'])
```

#### Чат с историей сообщений
```python
import ollama

response = ollama.chat(
    model='llama3.2:1b',
    messages=[
        {'role': 'system', 'content': 'Ты - полезный помощник.'},
        {'role': 'user', 'content': 'Напиши функцию Python для переворота строки.'}
    ]
)
print(response['message']['content'])
```

#### Потоковая передача ответов
```python
import ollama

# Потоковая генерация
stream = ollama.generate(
    model='llama3.2:1b', 
    prompt='Опиши рекурсию в одном предложении.',
    stream=True
)

for chunk in stream:
    print(chunk['response'], end='', flush=True)

# Потоковый чат
stream = ollama.chat(
    model='llama3.2:1b',
    messages=[
        {'role': 'user', 'content': 'Напиши хайку о программировании.'}
    ],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

### Способ 2: Прямое использование REST API

Если вы предпочитаете работать напрямую с HTTP-запросами или используете фреймворки, которые уже работают с HTTP :

#### Чат через API
```python
import requests
import json

url = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3.2:1b",
    "messages": [
        {"role": "system", "content": "Ты - помощник по Python."},
        {"role": "user", "content": "Напиши функцию, которая переворачивает строку."}
    ]
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        data = json.loads(line)
        content = data.get("message", {}).get("content", "")
        print(content, end="")
```

#### Генерация через API
```python
import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.2:1b",
    "prompt": "Объясни рекурсию в одном предложении."
}

response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
```

## 🔧 Продвинутые возможности и настройки

### Работа с параметрами генерации

Ollama предоставляет множество параметров для тонкой настройки генерации текста :

```python
import ollama

response = ollama.generate(
    model='llama3.2:1b',
    prompt='Расскажи о преимуществах Python для data science',
    options={
        'temperature': 0.7,      # Контроль случайности (0-1)
        'top_p': 0.9,           # Нуклеус-сэмплинг
        'top_k': 40,            # Ограничение словаря
        'num_predict': 256,     # Максимальная длина ответа
        'stop': ['\n', '###'],  # Строки остановки
        'repeat_penalty': 1.1,  # Штраф за повторения
        'seed': 42              # Seed для воспроизводимости
    }
)
```

### Поддержание контекста диалога

Для создания чат-бота с памятью о предыдущих сообщениях :

```python
import ollama

# Инициализация истории сообщений
messages = []

while True:
    user_input = input('Вы: ')
    if user_input.lower() == 'выход':
        break
        
    # Добавление пользовательского сообщения в историю
    messages.append({'role': 'user', 'content': user_input})
    
    # Получение ответа от модели
    response = ollama.chat(
        model='llama3.2:1b',
        messages=messages
    )
    
    answer = response['message']['content']
    print(f'Бот: {answer}')
    
    # Добавление ответа ассистента в историю
    messages.append({'role': 'assistant', 'content': answer})
```

### Создание собственного клиента Ollama

Для полного контроля над процессом вы можете создать собственный класс клиента :

```python
import requests
import json
from typing import Dict, Any, List, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.context = None
    
    def generate(self, 
                prompt: str, 
                model: str = "llama3.2:1b",
                temperature: float = 0.7,
                top_p: float = 0.9,
                num_predict: int = 256,
                **kwargs) -> Dict[str, Any]:
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
                **kwargs
            }
        }
        
        if self.context:
            payload["context"] = self.context
            
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Сохраняем контекст для следующих запросов
            if "context" in result:
                self.context = result["context"]
                
            return result
            
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def clear_context(self):
        """Очистить контекст"""
        self.context = None

# Использование
client = OllamaClient()
result = client.generate("Расскажи о машинном обучении")
print(result.get("response", "Ошибка"))
```

## 🧠 Продвинутые техники

### Работа со "мыслящими" моделями

Некоторые модели (например, qwen3) поддерживают вывод промежуточных размышлений :

```python
import ollama
import re

response = ollama.chat(
    model="qwen3",
    messages=[
        {"role": "user", "content": "Какая столица Австралии?"}
    ]
)

content = response['message']['content']

# Извлечение процесса мышления
thinking = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

print("🧠 Процесс мышления:\n", thinking[0].strip() if thinking else "N/A")
print("\n✅ Окончательный ответ:\n", answer.strip())
```

### Структурированный вывод

Для получения ответов в формате JSON :

```python
import ollama
import json

response = ollama.chat(
    model='llama3.2:1b',
    messages=[{
        'role': 'user', 
        'content': 'Опиши столицу США в формате JSON с полями: название, население, площадь.'
    }],
    format='json',  # Указываем формат вывода
    options={'temperature': 0}  # Для более предсказуемых результатов
)

try:
    json_response = json.loads(response['message']['content'])
    print(json.dumps(json_response, indent=2, ensure_ascii=False))
except json.JSONDecodeError:
    print("Ответ не в JSON формате:", response['message']['content'])
```

### Асинхронная работа

Для обработки нескольких запросов одновременно :

```python
import asyncio
from ollama import AsyncClient

async def async_chat_example():
    client = AsyncClient()
    
    messages = [
        {'role': 'user', 'content': 'Почему трава зеленая?'}
    ]
    
    # Обычный асинхронный запрос
    response = await client.chat(model='llama3.2:1b', messages=messages)
    print(response['message']['content'])
    
    # Асинхронный потоковый запрос
    async for part in await client.chat(model='llama3.2:1b', messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)

# Запуск асинхронной функции
asyncio.run(async_chat_example())
```

## 💡 Практические примеры использования

### Пример 1: Чат-бот с сохранением контекста

```python
import ollama

class ChatBot:
    def __init__(self, model: str = "llama3.2:1b"):
        self.model = model
        self.conversation_history = []
        
    def add_system_prompt(self, prompt: str):
        """Добавить системный промпт"""
        self.conversation_history.append({
            'role': 'system', 
            'content': prompt
        })
    
    def chat(self, user_input: str) -> str:
        """Отправить сообщение и получить ответ"""
        self.conversation_history.append({
            'role': 'user', 
            'content': user_input
        })
        
        response = ollama.chat(
            model=self.model,
            messages=self.conversation_history
        )
        
        answer = response['message']['content']
        
        self.conversation_history.append({
            'role': 'assistant', 
            'content': answer
        })
        
        return answer
    
    def clear_history(self):
        """Очистить историю разговора"""
        self.conversation_history = []

# Использование
bot = ChatBot()
bot.add_system_prompt("Ты - полезный ассистент, который отвечает кратко и по делу.")

while True:
    user_input = input("Вы: ")
    if user_input.lower() in ['выход', 'exit', 'quit']:
        break
        
    response = bot.chat(user_input)
    print(f"Бот: {response}")
```

### Пример 2: Автоматизация рабочих процессов

```python
import ollama

class WorkflowAutomation:
    def __init__(self, model: str = "llama3.2:1b"):
        self.model = model
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Суммаризировать текст"""
        prompt = f"""
        Суммаризируй следующий текст в {max_length} символов или меньше:
        
        {text}
        """
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.3, 'num_predict': max_length}
        )
        
        return response['response']
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """Генерация кода по описанию"""
        prompt = f"""
        Напиши код на {language} для следующей задачи:
        {description}
        
        Требования:
        - Добавь комментарии
        - Следуй best practices
        - Верни только код, без пояснений
        """
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.2}
        )
        
        return response['response']

# Использование
automator = WorkflowAutomation()

# Суммаризация текста
text = """
Оллома - это мощный инструмент для запуска больших языковых моделей локально...
"""
summary = automator.summarize_text(text)
print("Суммаризация:", summary)

# Генерация кода
code = automator.generate_code("функция для вычисления факториала")
print("Сгенерированный код:")
print(code)
```

## 🚀 Управление моделями

Библиотека Ollama также предоставляет функции для управления моделями :

```python
import ollama

# Список локальных моделей
models = ollama.list()
print("Доступные модели:")
for model in models['models']:
    print(f" - {model['name']}")

# Информация о модели
model_info = ollama.show('llama3.2:1b')
print("Информация о модели:", model_info)

# Создание кастомной модели
modelfile = '''
FROM llama3.2:1b
SYSTEM Ты - эксперт по Python программированию.
'''

ollama.create(model='my-python-assistant', modelfile=modelfile)

# Удаление модели
# ollama.delete('my-python-assistant')
```

## 🔍 Отладка и обработка ошибок

### Базовая обработка ошибок

```python
import ollama
from requests.exceptions import ConnectionError

def safe_ollama_chat(messages, model='llama3.2:1b', max_retries=3):
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content']
        
        except ConnectionError:
            if attempt < max_retries - 1:
                print(f"Ошибка соединения, попытка {attempt + 2} из {max_retries}...")
                continue
            else:
                return "Ошибка: не удается подключиться к Ollama серверу. Убедитесь, что Ollama запущена."
        
        except Exception as e:
            return f"Произошла ошибка: {str(e)}"

# Использование с обработкой ошибок
result = safe_ollama_chat([{'role': 'user', 'content': 'Привет!'}])
print(result)
```

### Проверка здоровья сервера

```python
import requests

def check_ollama_health():
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

if check_ollama_health():
    print("Ollama сервер работает нормально")
else:
    print("Ollama сервер недоступен. Запустите: ollama serve")
```

## 📊 Сравнение способов интеграции

| Задача | REST API | Клиент Python |
|--------|----------|---------------|
| Простая генерация текста | `/api/generate` | `ollama.generate()` |
| Диалоговый чат | `/api/chat` | `ollama.chat()` |
| Поддержка потоковой передачи | Да | Да |
| Управление моделями | Через CLI | `ollama.list()`, `ollama.pull()` и др. |
| Работа с контекстом | Ручное управление | Автоматическое через библиотеку |
| Асинхронная работа | Реализация самостоятельно | `AsyncClient` |
