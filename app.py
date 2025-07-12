import os
import io
import json
import zipfile
import pandas as pd
import requests
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("CRITICAL: Переменная окружения GOOGLE_API_KEY не найдена!")

model = genai.GenerativeModel('gemini-1.5-flash')

def get_content_from_url(url):
    try:
        response = requests.get(f"https://r.jina.ai/{url}", timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Ошибка при получении контента с {url}: {e}")
        return None

@app.route('/')
def index():
    history = session.get('history', [])
    last_result_exists = 'last_result' in session
    return render_template('index.html', history=history, last_result_exists=last_result_exists)

@app.route('/parse', methods=['POST'])
def parse_url():
    if not api_key:
        return jsonify({'error': 'Сервер не настроен: API ключ отсутствует.'}), 503
    data = request.json
    url = data.get('url')
    step = data.get('step')
    if not url:
        return jsonify({'error': 'URL не указан'}), 400
    if step == 'analyze':
        content = get_content_from_url(url)
        if not content:
            return jsonify({'error': 'Не удалось загрузить контент страницы.'}), 500
        session['page_content'] = content
        session['current_url'] = url
        prompt = f"""
        Проанализируй контент веб-страницы. Твоя задача — предоставить краткую сводку и определить, какие структурированные данные можно извлечь.
        Ответь СТРОГО в формате JSON без markdown-оберток. Структура JSON:
        {{
          "summary": "Краткое описание содержимого страницы в 2-3 предложениях.",
          "available_data": [
            {{"type": "tables", "description": "Краткое описание таблиц (например, 'Технические характеристики')."}},
            {{"type": "images", "description": "Описание изображений (например, 'Галерея продукта')."}},
            {{"type": "prices", "description": "Описание цен (например, 'Стоимость товаров')."}},
            {{"type": "contacts", "description": "Описание контактов (например, 'Адрес и телефон')."}}
          ]
        }}
        Если тип данных отсутствует, не включай его в массив "available_data".
        Контент: --- {content[:4000]} ---
        """
        try:
            response = model.generate_content(prompt)
            json_response_text = response.text.strip().replace("```json", "").replace("```", "")
            return jsonify(json_response_text)
        except Exception as e:
            return jsonify({'error': f'Ошибка API Gemini: {e}'}), 500
    elif step == 'extract':
        categories = data.get('categories')
        content = session.get('page_content')
        if not categories or not content:
            return jsonify({'error': 'Отсутствуют категории или контент.'}), 400
        prompts = {
            'tables': "Извлеки все таблицы. Верни как JSON-массив объектов.",
            'images': "Извлеки все URL изображений. Верни как JSON-массив строк.",
            'prices': "Извлеки все товары и их цены. Верни как JSON-массив объектов с ключами 'item' и 'price'.",
            'contacts': "Извлеки все контакты. Верни как JSON-объект."
        }
        results = {}
        for category in categories:
            if category in prompts:
                prompt_text = f"{prompts[category]}\n\nКонтент:\n---\n{content}\n---"
                try:
                    response = model.generate_content(prompt_text)
                    json_response_text = response.text.strip().replace("```json", "").replace("```", "")
                    results[category] = json_response_text
                except Exception as e:
                    results[category] = json.dumps({'error': str(e)})
        session['last_result'] = results
        history = session.get('history', [])
        if url not in history:
            history.insert(0, url)
        session['history'] = history[:5]
        return jsonify({'success': True, 'redirect_url': url_for('show_results')})

@app.route('/results')
def show_results():
    results = session.get('last_result')
    if not results:
        return redirect(url_for('index'))
    return render_template('results.html', results=results, url=session.get('current_url', ''))

@app.route('/download_csv/<category>')
def download_csv(category):
    data_str = session.get('last_result', {}).get(category, '[]')
    try:
        df = pd.read_json(io.StringIO(data_str))
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        mem_file = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        mem_file.seek(0)
        return send_file(mem_file, as_attachment=True, download_name=f'{category}_data.csv', mimetype='text/csv')
    except Exception as e:
        return f"Не удалось создать CSV: {e}", 500

@app.route('/download_images')
def download_images():
    image_urls_str = session.get('last_result', {}).get('images', '[]')
    try:
        image_urls = json.loads(image_urls_str)
        if not isinstance(image_urls, list): return "Ошибка формата данных", 500
    except json.JSONDecodeError:
        return "Ошибка декодирования JSON", 500
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_f:
        for i, url in enumerate(image_urls):
            try:
                img_response = requests.get(url, stream=True, timeout=10)
                if img_response.status_code == 200:
                    filename = os.path.basename(url.split('?')[0]) or f"image_{i+1}.jpg"
                    zip_f.writestr(filename, img_response.content)
            except requests.RequestException:
                pass
    zip_buffer.seek(0)
    return send_file(zip_buffer, as_attachment=True, download_name='images.zip', mimetype='application/zip')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)