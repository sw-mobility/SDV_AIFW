import re

def is_custom_model_id(model: str) -> bool:
    # 예시: 4자리 숫자 + 'P' + 4자리 숫자 + 'T' + 4자리 숫자
    pattern = r"^\d{4}P\d{4}T\d{4}$"
    return bool(re.match(pattern, model))