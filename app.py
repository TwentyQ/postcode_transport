from flask import Flask, request, Response, render_template, jsonify
from xml.etree import ElementTree as ET
import base64
import io
import tempfile
import os
import subprocess
import uuid

from PIL import Image
import torch
from ml.model import predict, build_model, valid_tf
from id_transportation import save_result, get_result

app = Flask(__name__)

XML_HEADER = '<?xml version="1.0" encoding="UTF-8"?>'
UPLOAD_DIR = 'data/uploads'
WEIGHTS_PATH = 'ml/best_model.pth'
os.makedirs(UPLOAD_DIR, exist_ok=True)


def xml_response(root, code=200):
    body = ET.tostring(root, encoding='unicode')
    return Response(XML_HEADER + body, mimetype='application/xml', status=code)


def error_response(message, code=400):
    root = ET.Element('response')
    ET.SubElement(root, 'status').text = 'error'
    ET.SubElement(root, 'message').text = message
    return xml_response(root, code)


def predict_with_confidence(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device).eval()

    img = Image.open(image_path).convert('RGB')
    inp = valid_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = [
            model(inp).item(),
            model(torch.flip(inp, dims=[3])).item(),
            model(torch.flip(inp, dims=[2])).item(),
            model(torch.flip(inp, dims=[2, 3])).item(),
        ]

    mean_pct = sum(preds) / len(preds) * 100
    mean_pct = min(max(mean_pct, 0.0), 100.0)
    std_pct = (sum((p * 100 - mean_pct) ** 2 for p in preds) / len(preds)) ** 0.5
    confidence = max(0.0, 1.0 - std_pct / 30.0)
    return mean_pct, confidence


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        root = ET.fromstring(request.data)
    except ET.ParseError:
        return error_response('Некорректный XML')

    transport_id = root.findtext('transport_id')
    image_b64 = root.findtext('image_base64')

    if not transport_id:
        return error_response('Не указан transport_id')
    if not image_b64:
        return error_response('Не передано изображение')

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return error_response('Некорректный base64')

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return error_response('Файл повреждён или не является изображением')

    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()
        load_pct, confidence = predict_with_confidence(tmp.name)
    finally:
        os.unlink(tmp.name)

    save_result(transport_id, load_pct)

    resp = ET.Element('response')
    ET.SubElement(resp, 'status').text = 'success'
    ET.SubElement(resp, 'transport_id').text = transport_id
    ET.SubElement(resp, 'load_pct').text = f'{load_pct:.2f}'
    ET.SubElement(resp, 'confidence').text = f'{confidence:.2f}'
    ET.SubElement(resp, 'is_underloaded').text = str(load_pct < 30).lower()
    return xml_response(resp)


@app.route('/api/result/<transport_id>', methods=['GET'])
def api_result(transport_id):
    result = get_result(transport_id)
    if result is None:
        return error_response(f'Перевозка {transport_id} не найдена', code=404)

    resp = ET.Element('response')
    ET.SubElement(resp, 'status').text = 'success'
    ET.SubElement(resp, 'transport_id').text = transport_id
    ET.SubElement(resp, 'load_pct').text = f"{result['load_pct']:.2f}"
    return xml_response(resp)


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    try:
        root = ET.fromstring(request.data)
    except ET.ParseError:
        return error_response('Некорректный XML')

    csv_b64 = root.findtext('csv_base64')
    if not csv_b64:
        return error_response('Не передан CSV')

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        csv_bytes = base64.b64decode(csv_b64)
    except Exception:
        return error_response('Некорректный base64 в CSV')

    csv_path = os.path.join(job_dir, 'new_data.csv')
    with open(csv_path, 'wb') as f:
        f.write(csv_bytes)

    image_count = 0
    for img_el in root.findall('image_base64'):
        if not img_el.text:
            continue
        try:
            img_bytes = base64.b64decode(img_el.text)
        except Exception:
            continue
        image_id = img_el.get('image_id') or str(uuid.uuid4())[:8]
        with open(os.path.join(job_dir, f'{image_id}.jpg'), 'wb') as f:
            f.write(img_bytes)
        image_count += 1

    if image_count == 0:
        return error_response('Не передано ни одного изображения')

    out_weights = f'ml/best_model_{job_id}.pth'
    log_path = os.path.join(job_dir, 'log.txt')
    with open(log_path, 'w') as log:
        subprocess.Popen(
            ['python', 'ml/retrain.py',
             '--csv', csv_path,
             '--images', job_dir,
             '--base', WEIGHTS_PATH,
             '--out', out_weights],
            stdout=log, stderr=subprocess.STDOUT
        )

    resp = ET.Element('response')
    ET.SubElement(resp, 'status').text = 'started'
    ET.SubElement(resp, 'job_id').text = job_id
    ET.SubElement(resp, 'image_count').text = str(image_count)
    return xml_response(resp)


@app.route('/api/retrain_status/<job_id>', methods=['GET'])
def retrain_status(job_id):
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    log_path = os.path.join(job_dir, 'log.txt')
    if not os.path.exists(log_path):
        return error_response('Job не найден', code=404)

    with open(log_path) as f:
        log = f.read()

    out_weights = f'ml/best_model_{job_id}.pth'
    done = os.path.exists(out_weights)

    resp = ET.Element('response')
    ET.SubElement(resp, 'status').text = 'done' if done else 'running'
    ET.SubElement(resp, 'job_id').text = job_id
    if done:
        ET.SubElement(resp, 'model_path').text = out_weights
    ET.SubElement(resp, 'log').text = log[-2000:]
    return xml_response(resp)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/ui/result/<transport_id>', methods=['GET'])
def ui_result(transport_id):
    result = get_result(transport_id)
    if result is None:
        return jsonify({'status': 'error', 'message': 'not found'}), 404
    return jsonify({
        'status': 'success',
        'transport_id': transport_id,
        'load_pct': result['load_pct'],
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)