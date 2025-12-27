import cv2
import numpy as np
import tensorflow as tf
import logging

# Deep feature extractor cho similarity checks (lazy-loaded)
MobileNetV2 = tf.keras.applications.MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

FEATURE_EXTRACTOR = None

def get_feature_extractor():
    """
    Tải và trả về mô hình MobileNetV2 đã được huấn luyện trước để trích xuất đặc trưng.
    Sử dụng cache để tránh tải lại mô hình.
    """
    global FEATURE_EXTRACTOR
    if FEATURE_EXTRACTOR is None:
        try:
            FEATURE_EXTRACTOR = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
        except Exception as e:
            logging.getLogger(__name__).exception("Error loading MobileNetV2: %s", e)
            return None
    return FEATURE_EXTRACTOR

def compute_embedding(img_bgr):
    """
    Tính toán vector embedding đã được chuẩn hóa cho một ảnh.
    Trả về một mảng numpy 1-D hoặc None nếu thất bại.
    """
    try:
        extractor = get_feature_extractor()
        if extractor is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224,224))
        x = np.expand_dims(img_resized.astype('float32'), axis=0)
        x = preprocess_input(x)
        emb = extractor.predict(x)
        norm = np.linalg.norm(emb)
        return emb.flatten() / norm if norm > 0 else emb.flatten()
    except Exception:
        logging.getLogger(__name__).exception("Error computing embedding")
        return None

def compute_hist(img_bgr):
    """
    Tính toán histogram màu HSV cho một ảnh.
    """
    s_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Tính histogram 2D cho H và S, sau đó flatten thành 1D để cv2.compareHist hoạt động ổn định
    s_hist = cv2.calcHist([s_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    # Chuẩn hóa in-place và trả về dưới dạng vector float32 1-D
    cv2.normalize(s_hist, s_hist)
    return s_hist.flatten().astype('float32')