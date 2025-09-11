import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


def extract_yolo_metrics(results, model=None) -> Dict[str, Any]:
    """YOLO 결과에서 메트릭을 추출하는 공통 함수"""
    try:
        metrics = {}

        # Box 메트릭 추출
        if hasattr(results, 'box') and results.box:
            box = results.box
            metrics['mAP_0.5'] = float(getattr(box, 'map50', 0.0) or 0.0)
            metrics['mAP_0.5_0.95'] = float(getattr(box, 'map', 0.0) or 0.0)
            metrics['mean_precision'] = float(getattr(box, 'mp', 0.0) or 0.0)
            metrics['mean_recall'] = float(getattr(box, 'mr', 0.0) or 0.0)

            # 클래스별 AP 추출
            if hasattr(box, 'ap50') and box.ap50 is not None:
                ap50_values = box.ap50
                if isinstance(ap50_values, (list, np.ndarray)) and len(ap50_values) > 0:
                    # 클래스별 AP50 값을 딕셔너리로 변환
                    class_ap = {}
                    for i, ap_value in enumerate(ap50_values):
                        if not np.isnan(ap_value) and ap_value is not None:
                            class_ap[str(i)] = float(ap_value)
                    metrics['class_ap'] = class_ap

        # 기타 정보
        if hasattr(results, 'speed'):
            metrics['inference_speed'] = results.speed

        if hasattr(results, 'seen'):
            metrics['total_images'] = int(results.seen)

        # 클래스 이름
        class_names = getattr(results, 'names', None) or getattr(model, 'names', None)
        if class_names:
            if isinstance(class_names, dict):
                metrics['class_names'] = {str(k): str(v) for k, v in class_names.items()}
            elif isinstance(class_names, (list, tuple)):
                metrics['class_names'] = {str(i): str(name) for i, name in enumerate(class_names)}
            else:
                # 기타 타입은 문자열로 변환
                metrics['class_names'] = str(class_names)

        # numpy 배열 변환
        metrics = _convert_numpy_values(metrics)
        metrics['validation_completed'] = True

        return metrics

    except Exception as e:
        logger.warning(f"Error extracting metrics: {e}")
        return {
            "mAP_0.5": 0.0,
            "mAP_0.5_0.95": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "total_images": 0,
            "validation_completed": False,
            "error": str(e)
        }


def extract_essential_summary(full_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """핵심 메트릭만 추출 - API 응답용"""
    if not full_metrics:
        return {}

    # 핵심 메트릭 추출
    essential = {
        'mAP_0.5': full_metrics.get('mAP_0.5', 0.0),
        'mAP_0.5_0.95': full_metrics.get('mAP_0.5_0.95', 0.0),
        'mean_precision': full_metrics.get('mean_precision', 0.0),
        'mean_recall': full_metrics.get('mean_recall', 0.0),
        'total_images': full_metrics.get('total_images', 0),
        'validation_completed': full_metrics.get('validation_completed', False),
        'class_names': full_metrics.get('class_names', {}),
        'inference_speed': full_metrics.get('inference_speed')
    }

    # 클래스 정보
    essential['total_classes'] = len(essential['class_names'])

    # 클래스별 AP (있으면 성능순 정렬)
    class_ap = full_metrics.get('class_ap', {})
    if class_ap:
        essential['class_ap'] = dict(sorted(class_ap.items(), key=lambda x: x[1], reverse=True))
        essential['total_classes_with_ap'] = len(class_ap)
    else:
        essential['class_ap'] = {}
        essential['total_classes_with_ap'] = 0

    return essential


def _convert_numpy_values(obj):
    """numpy 값들을 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_values(item) for item in obj]
    return obj