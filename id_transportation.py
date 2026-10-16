import os
import json
from datetime import datetime

ID_TRANSPORTATION_PATH = 'data/results.json'

def _load():
    if not os.path.exists(ID_TRANSPORTATION_PATH):
        return {}
    with open(ID_TRANSPORTATION_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save(data):
    os.makedirs(os.path.dirname(ID_TRANSPORTATION_PATH), exist_ok=True)
    with open(ID_TRANSPORTATION_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_result(transport_id, load_pct):
    data = _load()
    data[transport_id] = {'load_pct': load_pct}
    _save(data)
    return data[transport_id]


def get_result(transport_id):
    return _load().get(transport_id)


def save_result_full(transport_id, load_pct, confidence=None):
    data = _load()
    entry = data.get(transport_id, {})
    history = entry.get('history', [])
    record = {
        'load_pct': load_pct,
        'timestamp': datetime.utcnow().isoformat(),
    }
    if confidence is not None:
        record['confidence'] = confidence
    history.append(record)
    data[transport_id] = {
        'load_pct': load_pct,
        'confidence': confidence,
        'created_at': record['timestamp'],
        'history': history,
    }
    _save(data)
    return data[transport_id]


def get_history(transport_id):
    entry = _load().get(transport_id)
    if entry is None:
        return None
    return entry.get('history', [])


def get_all_results():
    return _load()


def get_stats():
    data = _load()
    total = len(data)
    if total == 0:
        return {
            'total_trips': 0,
            'avg_load_pct': 0.0,
            'underloaded_count': 0,
            'underloaded_percent': 0.0,
        }
    loads = [v['load_pct'] for v in data.values() if 'load_pct' in v]
    underloaded = [x for x in loads if x < 30]
    avg = sum(loads) / len(loads) if loads else 0.0
    return {
        'total_trips': total,
        'avg_load_pct': round(avg, 2),
        'underloaded_count': len(underloaded),
        'underloaded_percent': round(len(underloaded) / total * 100, 2) if total else 0.0,
    }