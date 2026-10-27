

def load_json(path):
    import json
    from pathlib import Path
    def reject(value): raise ValueError(f'Non-finite JSON constant: {value}')
    value=json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)
    if not isinstance(value,dict): raise ValueError(f'{path} must contain a JSON object')
    return value