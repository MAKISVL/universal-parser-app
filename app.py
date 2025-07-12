import os
import io
import json
import zipfile
import pandas as pd
import requests
import google.generativeai as genai
import time
import re
import logging

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from dotenv import load_dotenv

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24) # Генерация случайного ключа для сессий

# --- Конфигурация Gemini API ---
api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    logging.info("GOOGLE_API_KEY успешно загружен и сконфигурирован.")
else:
    logging.critical("CRITICAL: Переменная окружения GOOGLE_API_KEY не найдена! Приложение не сможет взаимодействовать с Gemini API.")

# Инициализация модели Gemini
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Вспомогательные функции ---

def extract_json_from_text(text):
    """
    Извлекает первую корректную JSON-строку из заданного текста,
    игнорируя окружающий "мусор" (например, markdown-обертки).
    """
    text = text.strip()
    
    start_brace = text.find('{')
    start_bracket = text.find('[')

    start_index = -1
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_index = start_brace
    elif start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
        start_index = start_bracket

    if start_index == -1:
        logging.warning("extract_json_from_text: Не найдено начало JSON ({ или [).")
        return None

    end_brace = text.rfind('}')
    end_bracket = text.rfind(']')

    end_index = -1
    if end_brace != -1 and (end_bracket == -1 or end_brace > end_bracket):
        end_index = end_brace
    elif end_bracket != -1 and (end_brace == -1 or end_bracket > end_brace):
        end_index = end_bracket
    
    if end_index == -1:
        logging.warning("extract_json_from_text: Не найдено окончание JSON (} или ]).")
        return None

    if start_index == -1 or end_index == -1 or start_index > end_index:
        logging.warning(f"extract_json_from_text: Неверные индексы старта/конца JSON. Start: {start_index}, End: {end_index}")
        return None

    json_candidate = text[start_index : end_index + 1]
    json_candidate = json_candidate.replace("```json", "").replace("```", "").strip()

    return json_candidate

def call_gemini_api_with_retry(prompt, retries=3, backoff_factor=1.5):
    """
    Выполняет запрос к Gemini API с повторными попытками при ошибках.
    Включает экспоненциальную задержку.
    """
    for i in range(retries):
        try:
            logging.info(f"Попытка {i+1}/{retries} запроса к Gemini API...")
            response = model.generate_content(prompt)
            
            if response and hasattr(response, 'text') and response.text:
                logging.info(f"Gemini API успешно вернул ответ на попытке {i+1}.")
                return response.text
            else:
                logging.warning(f"Попытка {i+1}: Gemini API вернул пустой или некорректный ответ.")
        except Exception as e:
            logging.error(f"Попытка {i+1}: Ошибка при вызове Gemini API: {e}")
            if "status_code=429" in str(e) or "quota" in str(e).lower():
                logging.warning("Обнаружен Too Many Requests (429) или ошибка квоты. Увеличиваем задержку.")
                time.sleep((backoff_factor ** i) * 5)
            elif i < retries - 1:
                time.sleep(backoff_factor ** i)
            else:
                logging.error(f"Все {retries} попыток запроса к Gemini API завершились неудачно.")
                raise 
    return None

def get_content_from_url(url):
    """
    Получает контент веб-страницы с помощью Jina AI Reader.
    """
    try:
        logging.info(f"Загрузка контента с Jina AI для URL: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36'}
        response = requests.get(f"https://r.jina.ai/{url}", timeout=30, headers=headers) # Здесь используется Jina AI
        response.raise_for_status() # Вызывает HTTPError для плохих ответов (4xx или 5xx)
        logging.info(f"Контент с Jina AI для URL {url} успешно загружен. Размер: {len(response.text)} символов.")
        return response.text
    except requests.exceptions.Timeout:
        logging.error(f"Таймаут (30 сек) при получении контента с Jina AI для {url}. Возможно, страница слишком долго отвечала.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при получении контента с Jina AI для {url}: {e}")
        return None

# --- Маршруты Flask ---

@app.route('/')
def index():
    history = session.get('history', [])
    last_result_exists = 'last_result' in session
    return render_template('index.html', history=history, last_result_exists=last_result_exists)

@app.route('/parse', methods=['POST'])
def parse_url():
    if not api_key:
        logging.error("API ключ Google не настроен. Отклоняем запрос /parse.")
        return jsonify({'error': 'Сервер не настроен: API ключ отсутствует. Пожалуйста, установите GOOGLE_API_KEY.'}), 503
    
    data = request.json
    url = data.get('url')
    step = data.get('step')
    
    if not url:
        logging.warning("URL не указан в запросе к /parse.")
        return jsonify({'error': 'URL не указан'}), 400

    if step == 'analyze':
        logging.info(f"Начало шага 'analyze' для URL: {url}")
        content = get_content_from_url(url)
        if content is None:
            logging.error(f"Не удалось загрузить контент страницы для анализа: {url}. Возвращаем ошибку 500.")
            return jsonify({'error': 'Не удалось загрузить контент страницы. Проверьте URL или повторите попытку.'}), 500
        
        session['page_content'] = content
        session['current_url'] = url
        
        # Обрезаем контент для промпта, чтобы уложиться в токен-лимиты Gemini.
        prompt_content = content[:8000] 
        if len(content) > 8000:
            logging.info(f"Контент страницы обрезан до 8000 символов для промпта анализа. Исходный размер: {len(content)}.")

        # Промпт для Jina AI-обработанного контента
        prompt = f"""
        Проанализируй контент веб-страницы. Твоя задача — предоставить краткую сводку и определить, какие структурированные данные можно извлечь.
        Ответь СТРОГО в формате JSON без markdown-оберток.
        Структура JSON:
        {{
          "summary": "Краткое описание содержимого страницы в 2-3 предложениях.",
          "available_data": [
            {{"type": "tables", "description": "Краткое описание таблиц (например, 'Технические характеристики' или 'Сравнительные данные')."}},
            {{"type": "images", "description": "Описание изображений (например, 'Галерея продукта' или 'Логотипы')."}},
            {{"type": "prices", "description": "Описание цен (например, 'Стоимость товаров', 'Цены на услуги' или 'Пакеты подписки')."}},
            {{"type": "contacts", "description": "Описание контактов (например, 'Адрес и телефон', 'Электронная почта' или 'Форма обратной связи')."}}
          ]
        }}
        Если тип данных отсутствует, не включай его в массив "available_data".
        Контент для анализа:
        ---
        {prompt_content}
        ---
        """
        
        try:
            logging.info(f"Отправка запроса на анализ в Gemini API для URL: {url} (контент: {len(prompt_content)} символов).")
            gemini_raw_response = call_gemini_api_with_retry(prompt)
            
            if gemini_raw_response is None:
                logging.error(f"Gemini API не вернул ответ после нескольких попыток для анализа URL: {url}.")
                return jsonify({'error': 'Не удалось получить ответ от AI после нескольких попыток. Попробуйте снова или проверьте URL.'}), 500

            logging.info(f"Получен RAW-ответ от Gemini (анализ). Начало: '{gemini_raw_response[:200]}...' Конец: '...{gemini_raw_response[-200:]}'")
            
            json_response_text = extract_json_from_text(gemini_raw_response)
            
            if json_response_text is None:
                logging.error(f"Не удалось извлечь JSON из ответа Gemini (анализ). RAW-ответ: {gemini_raw_response}")
                return jsonify({'error': 'AI вернул данные в некорректном формате для анализа. Пожалуйста, сообщите разработчику.'}), 500

            parsed_json_result = json.loads(json_response_text)
            logging.info("JSON от Gemini успешно распарсен (анализ).")

            return jsonify(parsed_json_result) 
        
        except json.JSONDecodeError as e:
            logging.error(f"Ошибка декодирования JSON после извлечения (анализ): {e}. JSON-кандидат: {json_response_text}. Полный RAW: {gemini_raw_response}")
            return jsonify({'error': f'AI вернул данные в неправильном формате (JSON-ошибка при анализе). Детали: {e}'}), 500
        except Exception as e:
            logging.exception(f'Непредвиденная ошибка API Gemini (анализ) для URL {url}:')
            return jsonify({'error': f'Произошла внутренняя ошибка при анализе страницы. Детали: {e}'}), 500

    elif step == 'extract':
        logging.info(f"Начало шага 'extract' для URL: {url}")
        categories = data.get('categories')
        content = session.get('page_content')
        
        if not categories or not content:
            logging.warning("Отсутствуют категории для извлечения или контент страницы в сессии. Перенаправляем на повторный анализ.")
            return jsonify({'error': 'Отсутствуют категории для извлечения или контент страницы. Пожалуйста, повторите анализ.'}), 400
        
        # Промпты для Jina AI-обработанного контента
        prompts = {
            'tables': "Извлеки все таблицы с веб-страницы. Верни как JSON-массив объектов. Каждый объект должен представлять строку таблицы, а ключи - заголовки столбцов.",
            'images': "Извлеки все URL изображений. Верни как JSON-массив строк.",
            'prices': "Извлеки все товары и их цены. Верни как JSON-массив объектов с ключами 'item' и 'price'.",
            'contacts': "Извлеки все контакты. Верни как JSON-объект, где ключи - типы контактов (например, 'address', 'phone', 'email'), а значения - найденные данные."
        }
        
        results = {}
        for category in categories:
            if category in prompts:
                prompt_text = f"{prompts[category]}\n\nКонтент для извлечения:\n---\n{content}\n---"
                try:
                    logging.info(f"Отправка запроса на извлечение ({category}) в Gemini API для URL: {url} (контент: {len(content)} символов).")
                    gemini_raw_response = call_gemini_api_with_retry(prompt_text)

                    if gemini_raw_response is None:
                        logging.error(f"Gemini API не вернул ответ после нескольких попыток для категории {category}.")
                        results[category] = json.dumps({'error': 'AI не смог извлечь данные после нескольких попыток.'})
                        continue

                    logging.info(f"Получен RAW-ответ от Gemini ({category}). Начало: '{gemini_raw_response[:200]}...'")
                    
                    json_response_text = extract_json_from_text(gemini_raw_response)

                    if json_response_text is None:
                        logging.error(f"Не удалось извлечь JSON из ответа Gemini (категория {category}). RAW: {gemini_raw_response}")
                        results[category] = json.dumps({'error': 'AI вернул данные в некорректном формате для этой категории.'})
                        continue

                    json.loads(json_response_text) 
                    results[category] = json_response_text 
                    logging.info(f"Данные для категории {category} успешно извлечены и валидированы.")

                except json.JSONDecodeError as e:
                    logging.error(f"Ошибка декодирования JSON после извлечения (категория {category}): {e}. JSON-кандидат: {json_response_text}. Полный RAW: {gemini_raw_response}")
                    results[category] = json.dumps({'error': f'AI вернул данные в неправильном формате для этой категории. Детали: {e}'})
                except Exception as e:
                    logging.exception(f'Непредвиденная ошибка API Gemini (извлечение {category}) для URL {url}:')
                    results[category] = json.dumps({'error': f'Произошла внутренняя ошибка при извлечении данных. Детали: {e}'})
            else:
                logging.warning(f"Запрошена неизвестная категория: {category}. Игнорируем.")
                results[category] = json.dumps({'error': 'Неизвестная категория запроса.'})
        
        session['last_result'] = results
        history = session.get('history', [])
        if url not in history:
            history.insert(0, url)
        session['history'] = history[:5]
        logging.info(f"Шаг 'extract' завершен. Данные успешно извлечены и сохранены в сессии для URL: {url}.")
        return jsonify({'success': True, 'redirect_url': url_for('show_results')})

# --- Маршруты для скачивания ---

@app.route('/results')
def show_results():
    results = session.get('last_result')
    if not results:
        logging.info("Попытка перейти на /results без данных в сессии. Перенаправление на главную страницу.")
        return redirect(url_for('index'))
    logging.info(f"Отображение результатов для URL: {session.get('current_url', 'N/A')}")
    return render_template('results.html', results=results, url=session.get('current_url', ''))

@app.route('/download_csv/<category>')
def download_csv(category):
    data_str = session.get('last_result', {}).get(category, '[]')
    if not data_str or data_str == '[]' or 'error' in data_str.lower():
        logging.warning(f"Попытка скачать CSV для {category}, но данные отсутствуют или содержат ошибку. Данные: {data_str[:100]}...")
        return "Данные для скачивания отсутствуют или некорректны.", 404
    try:
        df = pd.read_json(io.StringIO(data_str))
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8') 
        mem_file = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        mem_file.seek(0)
        logging.info(f"CSV файл для категории {category} успешно сгенерирован и отправлен.")
        return send_file(mem_file, as_attachment=True, download_name=f'{category}_data.csv', mimetype='text/csv')
    except Exception as e:
        logging.error(f"Не удалось создать CSV для категории {category}: {e}. Исходные данные: {data_str[:200]}...")
        return f"Не удалось создать CSV: {e}. Возможно, данные в неправильном формате для конвертации в CSV.", 500

@app.route('/download_images')
def download_images():
    image_urls_str = session.get('last_result', {}).get('images', '[]')
    if not image_urls_str or image_urls_str == '[]' or 'error' in image_urls_str.lower():
        logging.warning(f"Попытка скачать изображения, но URL отсутствуют или содержат ошибку. Данные: {image_urls_str[:100]}...")
        return "URL изображений для скачивания отсутствуют или некорректны.", 404
    try:
        image_urls = json.loads(image_urls_str)
        if not isinstance(image_urls, list): 
            logging.error(f"Ошибка формата данных URL изображений: {image_urls_str}. Ожидался список URL-ов.")
            return "Ошибка формата данных: ожидался список URL изображений", 500
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка декодирования JSON для URL изображений: {e}. Исходные данные: {image_urls_str[:200]}...")
        return "Ошибка декодирования JSON для URL изображений", 500
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_f:
        downloaded_filenames = set() 

        for i, url in enumerate(image_urls):
            if not url or not isinstance(url, str):
                logging.warning(f"Пропущен невалидный URL изображения: {url}. (Индекс: {i})")
                continue

            try:
                logging.info(f"Скачивание изображения {i+1}/{len(image_urls)}: {url[:100]}...")
                img_response = requests.get(url, stream=True, timeout=15)
                
                if img_response.status_code == 200:
                    filename = os.path.basename(url.split('?')[0].split('#')[0]) 
                    if not filename:
                        filename = f"image_{i+1}.jpg"
                        logging.warning(f"Не удалось извлечь имя файла из URL {url}. Использовано: {filename}")
                    
                    if '.' not in filename and img_response.headers.get('Content-Type'):
                        mime_type = img_response.headers['Content-Type']
                        if 'image/jpeg' in mime_type:
                            filename += '.jpg'
                        elif 'image/png' in mime_type:
                            filename += '.png'
                        elif 'image/gif' in mime_type:
                            filename += '.gif'

                    base_filename, extension = os.path.splitext(filename)
                    counter = 0
                    final_filename = filename
                    while final_filename in downloaded_filenames:
                        counter += 1
                        final_filename = f"{base_filename}_{counter}{extension}"
                    
                    zip_f.writestr(final_filename, img_response.content)
                    downloaded_filenames.add(final_filename)
                    logging.info(f"Изображение {url[:50]}... успешно добавлено в ZIP как {final_filename}.")
                else:
                    logging.warning(f"Не удалось скачать изображение {url[:100]}...: Статус {img_response.status_code}. Пропускаем.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Ошибка HTTP/сети при скачивании изображения {url[:100]}...: {e}. Пропускаем.")
            except Exception as e:
                logging.error(f"Непредвиденная ошибка при обработке изображения {url[:100]}...: {e}. Пропускаем.")

    zip_buffer.seek(0)
    if not downloaded_filenames:
        logging.warning("ZIP-архив с изображениями пуст, так как ничего не было скачано.")
        return "Не удалось скачать изображения. Проверьте URL или настройки доступа.", 404 
    
    logging.info(f"ZIP-архив с {len(downloaded_filenames)} изображениями успешно сгенерирован и отправлен.")
    return send_file(zip_buffer, as_attachment=True, download_name='images.zip', mimetype='application/zip')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)