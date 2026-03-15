import io
import sys

# Fix Windows console encoding FIRST (before any print statements)
if sys.platform == 'win32':
    import codecs
    try:
        # Reconfigure stdout and stderr to use UTF-8
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python < 3.7 fallback
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

import numpy as np
import pickle
import os
import cv2
import json
import random
import hashlib
import re
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session, make_response, jsonify
from functools import wraps
import tensorflow as tf
import shutil 
from PIL import Image
from utils import (
    compute_hist, 
    compute_embedding as _compute_embedding,
    calculate_severity_from_prediction,
    assess_disease_severity,
    check_image_quality
)
from datetime import datetime, timedelta
import threading
from uuid import uuid4
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import google.generativeai as genai
from collections import Counter, defaultdict
import time
from typing import Dict, List, Optional, Tuple, Any
import requests
from payment import VNPayPayment, MoMoPayment, get_client_ip
from notifications import (
    init_email_service, 
    send_order_confirmation_email,
    send_payment_success_email,
    send_payment_failed_email,
    send_welcome_email
)
try:
    import markdown
except ImportError:
    markdown = None  # Fallback nếu chưa cài

# Load biến môi trường từ file .env
load_dotenv()

# Cấu hình Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = None
LAST_WORKING_MODEL = None  # Cache model đã thành công

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Security: Chỉ hiển thị 8 ký tự đầu của API key
        masked_key = f"{GEMINI_API_KEY[:8]}***{GEMINI_API_KEY[-4:]}" if len(GEMINI_API_KEY) > 12 else "***"
        print(f"[OK] Gemini API key configured: {masked_key}")
        
        # Khởi tạo model - không test ngay để tránh waste quota
        # Thử các model theo thứ tự: model miễn phí ổn định nhất trước
        model_names = [
            'models/gemini-flash-latest',       # Luôn dùng phiên bản mới nhất
            'models/gemini-2.5-flash',          # Model ổn định, miễn phí
            'models/gemini-2.0-flash',          # Backup ổn định
        ]
        
        # Khởi tạo model đầu tiên, sẽ test khi user gửi câu hỏi thật
        GEMINI_MODEL = genai.GenerativeModel(model_names[0])
        LAST_WORKING_MODEL = model_names[0]
        print(f"[OK] Gemini model initialized: {model_names[0]}")
        print(f"[OK] Chatbot ready to answer questions")
        
    except Exception as e:
        print(f"[ERROR] Failed to configure Gemini API: {e}")
        GEMINI_MODEL = None
else:
    print("[WARNING] GEMINI_API_KEY not found - chatbot will not work")

# ----------------- CUSTOM EXCEPTIONS -----------------
class AppException(Exception):
    """Exception cơ sở cho các lỗi ứng dụng"""
    def __init__(self, message, details=None, user_message=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.user_message = user_message or message

class ValidationError(AppException):
    """Exception cho lỗi validation"""
    pass

class ModelError(AppException):
    """Exception cho lỗi liên quan đến model"""
    pass

class ImageProcessingError(AppException):
    """Exception cho lỗi xử lý ảnh"""
    pass

# ----------------- LOGGING CONFIGURATION -----------------
def setup_logging(app):
    """Cấu hình logging có cấu trúc với rotation file và console output"""
    # Tạo thư mục logs ở root project (web_tomato/logs) thay vì tomato/logs
    log_dir = Path(__file__).resolve().parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Đặt logging level từ biến môi trường hoặc mặc định INFO
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    app.logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Xóa các handlers mặc định
    app.logger.handlers.clear()
    
    # Console handler với output cho development
    # UTF-8 encoding đã được set ở đầu file cho Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    app.logger.addHandler(console_handler)
    
    # File handler với rotation cho tất cả logs
    file_handler = RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    app.logger.addHandler(file_handler)
    
    # File log riêng cho errors
    error_handler = RotatingFileHandler(
        log_dir / 'error.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    app.logger.addHandler(error_handler)
    
    # Log các exceptions không được bắt
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        app.logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception
    
    app.logger.info("Logging system initialized with level: %s", log_level)

# ----------------- CÀI ĐẶT -----------------
# BASE_DIR trỏ về root project (web_tomato/) thay vì tomato/
# Tránh tạo duplicate folders __pycache__, data, logs
BASE_DIR = Path(__file__).resolve().parent.parent

# Cấu hình LRU cache
MAX_LOADED_MODELS = int(os.environ.get('MAX_LOADED_MODELS', '2'))

# Các thres cáu hình thông qua env vars (có thể tune)
MIN_MODEL_CONF = float(os.environ.get('MIN_MODEL_CONF', '0.85'))
POS_SIM_THRESH = float(os.environ.get('POS_SIM_THRESH', os.environ.get('POS_SIM_THRES', '0.60')))
NEG_SIM_THRESH = float(os.environ.get('NEG_SIM_THRESH', os.environ.get('NEG_SIM_THRES', '0.75')))
MODELS_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus"
]

# Thông tin bệnh (tên hiển thị, định nghĩa ngắn, và các biện pháp phòng ngừa)
DISEASE_INFO = {
    "Tomato_Early_blight": {
        "name": "Bệnh cháy sớm (Early blight)",
        "definition": (
            "Do nấm Alternaria solani gây ra, xuất hiện các đốm màu nâu đậm trên lá và cuống, "
            "có thể làm lá vàng và rụng, giảm khả năng quang hợp và năng suất."
        ),
        "prevention": [
            "Luân canh cây trồng, tránh trồng cà chua liên tiếp trên cùng diện tích.",
            "Loại bỏ lá và tàn dư cây bệnh, tiêu hủy sạch sẽ.",
            "Sử dụng giống kháng nếu có và bón phân cân đối để cây khỏe.",
            "Phun thuốc bảo vệ thực vật (fungicide) đúng loại và đúng liều khi bệnh xuất hiện.",
            "Tránh tưới lên lá (tưới nhỏ giọt hoặc vào gốc) và giữ mật độ trồng hợp lý để thoáng khí.",
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Bệnh đốm lá Septoria (Septoria leaf spot)",
        "definition": (
            "Gây ra bởi nấm Septoria lycopersici, xuất hiện nhiều đốm nhỏ, vòng tròn có tâm màu sáng "
            "và viền đậm; nặng có thể làm lá rụng hàng loạt."
        ),
        "prevention": [
            "Loại bỏ và tiêu hủy lá, cành bị nhiễm để giảm nguồn bệnh.",
            "Tránh tưới phun lên lá, áp dụng tưới nhỏ giọt để hạn chế độ ẩm bề mặt lá.",
            "Luân canh cây trồng, không để tàn dư cây bệnh trên ruộng.",
            "Sử dụng thuốc bảo vệ thực vật theo khuyến cáo khi mật độ bệnh cao.",
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Bệnh xoăn vàng lá (Tomato yellow leaf curl virus)",
        "definition": (
            "Là bệnh do virus (TYLCV) gây ra, truyền bằng rầy tàu (whitefly). Triệu chứng gồm lá vàng, "
            "mép lá cuốn quăn, cây lùn và năng suất giảm mạnh."
        ),
        "prevention": [
            "Kiểm soát rầy mang virus: dùng bẫy dính, thuốc trừ sâu chọn lọc và biện pháp sinh học.",
            "Sử dụng giống kháng virus nếu có sẵn.",
            "Loại bỏ và tiêu hủy cây bị nhiễm để tránh nguồn lây lan.",
            "Dùng lưới, mái che hoặc biện pháp bảo vệ ban đầu để giảm mật độ rầy.",
            "Trồng vào thời vụ ít rầy hoặc kết hợp biện pháp phòng trừ tổng hợp (IPM)."
        ]
    },
    "Tomato_healthy": {
        "name": "Cây khỏe mạnh",
        "definition": "Cây cà chua không có dấu hiệu bệnh tật, lá xanh tốt, không có đốm hay biến dạng.",
        "prevention": [
            "Duy trì thói quen chăm sóc tốt: tưới nước đều, bón phân cân đối và kiểm soát sâu bệnh định kỳ.",
            "Đảm bảo cây trồng có đủ ánh sáng và thông thoáng để phát triển khỏe mạnh.",
            "Thường xuyên kiểm tra cây để phát hiện sớm các dấu hiệu bất thường.",
        ]
    },
    "Tomato_Bacterial_spot": {
        "name": "Bệnh đốm vi khuẩn (Bacterial spot)",
        "definition": (
            "Do vi khuẩn Xanthomonas gây ra, xuất hiện các đốm nhỏ màu đen hoặc nâu trên lá, thân và quả. "
            "Bệnh phát triển mạnh trong điều kiện ẩm ướt, gây giảm năng suất và chất lượng quả."
        ),
        "prevention": [
            "Sử dụng giống kháng bệnh và hạt giống không nhiễm bệnh.",
            "Tránh tưới phun lên lá, sử dụng hệ thống tưới nhỏ giọt.",
            "Loại bỏ và tiêu hủy cây bệnh để giảm nguồn lây nhiễm.",
            "Luân canh cây trồng với các loại cây không thuộc họ cà.",
            "Phun thuốc chứa đồng (copper-based) theo khuyến cáo khi bệnh xuất hiện.",
            "Đảm bảo thông thoáng và giảm độ ẩm trong vườn trồng.",
        ]
    },
    "Tomato_Late_blight": {
        "name": "Bệnh cháy muộn (Late blight)",
        "definition": (
            "Do nấm mốc Phytophthora infestans gây ra, là bệnh nguy hiểm nhất trên cà chua. "
            "Triệu chứng gồm các vết đốm màu nâu đen lan nhanh trên lá, thân và quả, có thể phá hủy toàn bộ vườn trong vài ngày."
        ),
        "prevention": [
            "Sử dụng giống kháng bệnh nếu có.",
            "Tránh trồng gần khoai tây vì cùng bị bệnh này.",
            "Đảm bảo khoảng cách trồng hợp lý để thoáng khí.",
            "Tránh tưới nước vào buổi tối, không để lá ướt qua đêm.",
            "Phun thuốc phòng ngừa (fungicide hệ thống) khi điều kiện thuận lợi cho bệnh.",
            "Loại bỏ và tiêu hủy cây bệnh ngay khi phát hiện.",
            "Theo dõi dự báo thời tiết và cảnh báo dịch bệnh trong vùng.",
        ]
    },
    "Tomato_Leaf_Mold": {
        "name": "Bệnh mốc lá (Leaf Mold)",
        "definition": (
            "Do nấm Passalora fulva (trước gọi là Cladosporium fulvum) gây ra, thường xảy ra trong nhà kính. "
            "Triệu chứng là các đốm màu vàng trên mặt trên của lá và lớp nấm mốc màu xanh lục hoặc xám trên mặt dưới."
        ),
        "prevention": [
            "Đảm bảo thông gió tốt trong nhà kính hoặc vườn trồng.",
            "Kiểm soát độ ẩm, tránh độ ẩm quá cao (trên 85%).",
            "Tưới vào gốc, không tưới phun lên lá.",
            "Giữ khoảng cách trồng hợp lý để cây được thoáng.",
            "Sử dụng giống kháng bệnh nếu có.",
            "Loại bỏ lá bị nhiễm và tiêu hủy.",
            "Phun thuốc bảo vệ thực vật khi cần thiết.",
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Nhện đỏ hai chấm (Two-spotted spider mite)",
        "definition": (
            "Nhện đỏ là loại sâu hại nhỏ bé hút dịch lá, gây ra các đốm nhỏ màu vàng trên lá. "
            "Khi nhiễm nặng, lá sẽ khô, vàng và rụng. Thường xuất hiện trong điều kiện khô hạn và nóng."
        ),
        "prevention": [
            "Tưới nước đầy đủ, duy trì độ ẩm thích hợp vì nhện đỏ thích môi trường khô.",
            "Phun nước lên mặt dưới của lá để loại bỏ nhện.",
            "Sử dụng thiên địch tự nhiên như rệp khướng (predatory mites) để kiểm soát.",
            "Tránh sử dụng thuốc trừ sâu phổ rộng có thể giết thiên địch.",
            "Loại bỏ lá bị nhiễm nặng.",
            "Sử dụng xà phòng diệt côn trùng hoặc dầu neem khi cần.",
            "Theo dõi thường xuyên, đặc biệt trong mùa khô.",
        ]
    },
    "Tomato__Target_Spot": {
        "name": "Bệnh đốm bia (Target Spot)",
        "definition": (
            "Do nấm Corynespora cassiicola gây ra, tạo các vết đốm hình tròn đồng tâm giống bia bắn. "
            "Bệnh ảnh hưởng đến lá, thân và quả, làm giảm năng suất và chất lượng."
        ),
        "prevention": [
            "Sử dụng giống có khả năng chống chịu tốt.",
            "Luân canh cây trồng để giảm nguồn bệnh trong đất.",
            "Loại bỏ tàn dư cây sau thu hoạch.",
            "Đảm bảo thoát nước tốt và tránh úng nước.",
            "Tưới vào gốc cây, tránh làm ướt lá.",
            "Giữ khoảng cách trồng hợp lý để thông thoáng.",
            "Phun thuốc bảo vệ thực vật khi phát hiện bệnh.",
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Bệnh vi rút khảm lá (Tomato mosaic virus)",
        "definition": (
            "Do virus ToMV gây ra, lây lan qua tiếp xúc cơ học, dụng cụ, tay người làm vườn. "
            "Triệu chứng gồm lá có vệt khảm màu vàng xanh, lá biến dạng, quả có vệt và phát triển không đều."
        ),
        "prevention": [
            "Sử dụng giống kháng virus nếu có.",
            "Rửa tay và khử trùng dụng cụ trước khi làm việc với cây.",
            "Tránh hút thuốc gần cây cà chua vì thuốc lá có thể mang virus.",
            "Loại bỏ và tiêu hủy cây bị nhiễm ngay lập tức.",
            "Kiểm soát sâu hút dịch có thể truyền bệnh.",
            "Sử dụng hạt giống sạch bệnh hoặc xử lý nhiệt hạt giống.",
            "Tránh trồng cà chua gần các loại cây cùng họ đã bị nhiễm.",
        ]
    }
}
# Danh sách các kiến trúc model
ARCHITECTURES = [
    'VGG19', 'MobileNetV2', 'ResNet50', 'CNN', 'InceptionV3', 'DenseNet', 'Xception', 'VGG16'
]

# Constants for model selection
MODELS = ARCHITECTURES  # Alias for compatibility
DEFAULT_MODEL = 'VGG19'
DEFAULT_PIPELINE = 'average_hsv'

def discover_models(base_dir: Path, architectures: list, logger=None) -> dict:
    """
    Tự động quét thư mục `base_dir` để tìm các file model (hỗ trợ cả `.h5` và `.keras`) và xây dựng map.
    Quy ước tên file: {pipeline_key}_{arch_name}_best.(h5|keras)
    Ưu tiên file _repaired.keras nếu có.
    """
    model_map = {}
    # Tìm các file kết thúc bằng _best.h5 hoặc _best.keras
    model_files = list(base_dir.rglob('*_best.*'))
    # Cũng tìm các file đã sửa chữa
    repaired_files = list(base_dir.rglob('*_best_repaired.*'))

    for model_path in model_files:
        if model_path.suffix.lower() not in ('.h5', '.keras'):
            continue
        # Bỏ qua các file chứa '_repaired' trong tên, chúng ta sẽ xử lý riêng
        if '_repaired' in model_path.stem:
            continue
            
        filename = model_path.name
        lowername = filename.lower()
        # Tìm kiến trúc model trong tên file (không phân biệt hoa/thường)
        found_arch = None
        matched_marker = None
        for arch in architectures:
            marker = f"_{arch.lower()}_best"
            if marker in lowername:
                found_arch = arch
                matched_marker = marker
                break

        if found_arch and matched_marker:
            # Trích xuất pipeline_key bằng cách lấy phần trước marker (chữ thường để khớp PIPELINES)
            idx = lowername.rfind(matched_marker)
            pipeline_key = lowername[:idx]
            key = (found_arch, pipeline_key)
            model_map[key] = model_path

    # Ưu tiên load file repaired nếu có (ghi đè lên file gốc)
    for model_path in repaired_files:
        if model_path.suffix.lower() not in ('.h5', '.keras'):
            continue
        filename = model_path.name
        lowername = filename.lower()
        # Tìm kiến trúc model trong tên file (không phân biệt hoa/thường)
        found_arch = None
        matched_marker = None
        for arch in architectures:
            marker = f"_{arch.lower()}_best_repaired"
            if marker in lowername:
                found_arch = arch
                matched_marker = marker
                break

        if found_arch and matched_marker:
            # Trích xuất pipeline_key bằng cách lấy phần trước marker (chữ thường để khớp PIPELINES)
            idx = lowername.rfind(matched_marker)
            pipeline_key = lowername[:idx]
            key = (found_arch, pipeline_key)
            # Ghi đè lên file gốc nếu có
            model_map[key] = model_path
            if logger:
                logger.info("Using repaired model for %s + %s: %s", found_arch, pipeline_key, model_path)

    return model_map

# ----------------- PIPELINES -----------------
# Chuyển đổi mô hình màu
# RGB -> CMYK
def rgb_to_cmyk(arr_uint8):
    rgb = arr_uint8.astype('float32') / 255.0
    K = 1 - np.max(rgb, axis=2)
    # Add a small epsilon to avoid division by zero
    den = 1 - K + 1e-9
    C = (1 - rgb[...,0] - K) / den
    M = (1 - rgb[...,1] - K) / den
    Y = (1 - rgb[...,2] - K) / den
    cmy = np.stack([C, M, Y], axis=2)
    # Giữ định dạng float32 trong khoảng [0, 1]
    return np.clip(cmy, 0, 1)
# RGB -> HSI
def rgb_to_hsi(arr_uint8):
    img = arr_uint8.astype('float32') / 255.0
    R, G, B = img[...,0], img[...,1], img[...,2]
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-9
    theta = np.arccos(np.clip(num/den, -1, 1))
    H = np.where(B <= G, theta, 2*np.pi - theta) / (2*np.pi)
    S = 1 - 3*np.minimum(np.minimum(R, G), B) / (R+G+B+1e-9)
    I = (R + G + B) / 3.0
    hsi = np.stack([H, S, I], axis=2)
    # Giữ định dạng float32 trong khoảng [0, 1]
    return np.clip(hsi, 0, 1)
# RGB -> HSV
def rgb_to_hsv(arr_uint8):
    """
    Chuyển đổi RGB sang không gian màu HSV.
    OpenCV HSV_FULL trả về giá trị trong khoảng [0, 255] cho tất cả các kênh.
    Chúng ta chuẩn hóa về [0, 1] để tương thích với input của model.
    """
    # cv2.cvtColor mong đợi input uint8 và trả về output uint8 với HSV_FULL
    hsv = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2HSV_FULL)
    # Chuẩn hóa về [0, 1] cho model
    return hsv.astype('float32') / 255.0

# Các phương pháp tiền xử lý
# Gaussian Noise
def add_gaussian_noise_uint8(img_uint8, var=5.0):
    sigma = var**0.5
    gauss = np.random.normal(0, sigma, img_uint8.shape).astype('float32')
    noisy = img_uint8.astype('float32') + gauss
    return np.clip(noisy, 0, 255).astype('uint8')
# Gaussian Noise + Gaussian Blur và CMYK
def pipeline_gb_noise_cmyk(img_arr):
    img = cv2.GaussianBlur(img_arr, (3,3), 0)
    img = add_gaussian_noise_uint8(img, var=5.0) # add_gaussian_noise_uint8 still returns uint8
    return rgb_to_cmyk(img) # rgb_to_cmyk now returns float32 [0, 1]
# Gaussian Noise + Gaussian Blur và HSI
def pipeline_gb_noise_hsi(img_arr):
    img = cv2.GaussianBlur(img_arr, (3,3), 0)
    img = add_gaussian_noise_uint8(img, var=5.0) # add_gaussian_noise_uint8 still returns uint8
    return rgb_to_hsi(img) # rgb_to_hsi now returns float32 [0, 1]
# Median Filter và CMYK
def pipeline_median_cmyk(img_arr):
    img = cv2.medianBlur(img_arr, 3)
    return rgb_to_cmyk(img) # trả về float32 [0, 1]
# Median Filter và HSI
def pipeline_median_hsi(img_arr):
    img = cv2.medianBlur(img_arr, 3)
    return rgb_to_hsi(img) # trả về float32 [0, 1]
# Average Filter và HSV
def pipeline_average_hsv(img_arr):
    img = cv2.blur(img_arr, (3, 3)) # Bộ lọc trung bình
    return rgb_to_hsv(img)
PIPELINES = {
    'gb_noise_cmyk': (pipeline_gb_noise_cmyk, 'GaussianBlur + Noise -> CMYK'),
    'gb_noise_hsi' : (pipeline_gb_noise_hsi, 'GaussianBlur + Noise -> HSI'),
    'median_cmyk' : (pipeline_median_cmyk, 'Median -> CMYK'),
    'median_hsi' : (pipeline_median_hsi, 'Median -> HSI'), 
    'average_hsv': (pipeline_average_hsv, 'Average Blur -> HSV')
}

# ----------------- Flask app -----------------
app = Flask(__name__, 
            static_folder=str(BASE_DIR / "static"), 
            template_folder=str(BASE_DIR / "templates"))
app.config['START_TIME'] = datetime.utcnow()
# Sử dụng biến môi trường cho secret_key, với một giá trị mặc định an toàn cho môi trường dev
app.secret_key = os.environ.get('SECRET_KEY', 'a-strong-default-secret-key-for-development-only')

# ----------------- Payment Gateway Configuration -----------------
# VNPay Configuration
VNPAY_TMN_CODE = os.environ.get('VNPAY_TMN_CODE', '')
VNPAY_SECRET_KEY = os.environ.get('VNPAY_SECRET_KEY', '')
VNPAY_PAYMENT_URL = os.environ.get('VNPAY_PAYMENT_URL', 'https://sandbox.vnpayment.vn/paymentv2/vpcpay.html')
VNPAY_RETURN_URL = os.environ.get('VNPAY_RETURN_URL', 'http://localhost:5000/payment/vnpay/callback')

# MoMo Configuration
MOMO_PARTNER_CODE = os.environ.get('MOMO_PARTNER_CODE', '')
MOMO_ACCESS_KEY = os.environ.get('MOMO_ACCESS_KEY', '')
MOMO_SECRET_KEY = os.environ.get('MOMO_SECRET_KEY', '')
MOMO_PAYMENT_URL = os.environ.get('MOMO_PAYMENT_URL', 'https://test-payment.momo.vn/v2/gateway/api/create')
MOMO_RETURN_URL = os.environ.get('MOMO_RETURN_URL', 'http://localhost:5000/payment/momo/callback')
MOMO_NOTIFY_URL = os.environ.get('MOMO_NOTIFY_URL', 'http://localhost:5000/payment/momo/ipn')

# Initialize payment gateways
vnpay_payment = None
momo_payment = None

if VNPAY_TMN_CODE and VNPAY_SECRET_KEY:
    vnpay_payment = VNPayPayment(
        tmn_code=VNPAY_TMN_CODE,
        secret_key=VNPAY_SECRET_KEY,
        payment_url=VNPAY_PAYMENT_URL,
        return_url=VNPAY_RETURN_URL
    )
    print(f"[OK] VNPay payment gateway configured")
else:
    print("[WARNING] VNPay not configured - payment with VNPay will not work")

if MOMO_PARTNER_CODE and MOMO_ACCESS_KEY and MOMO_SECRET_KEY:
    momo_payment = MoMoPayment(
        partner_code=MOMO_PARTNER_CODE,
        access_key=MOMO_ACCESS_KEY,
        secret_key=MOMO_SECRET_KEY,
        payment_url=MOMO_PAYMENT_URL,
        return_url=MOMO_RETURN_URL,
        notify_url=MOMO_NOTIFY_URL
    )
    print(f"[OK] MoMo payment gateway configured")
else:
    print("[WARNING] MoMo not configured - payment with MoMo will not work")

# ----------------- Email Notification Configuration -----------------
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
SMTP_FROM_EMAIL = os.environ.get('SMTP_FROM_EMAIL', SMTP_USER)
SMTP_FROM_NAME = os.environ.get('SMTP_FROM_NAME', 'Tomato AI')

# Initialize email service
if SMTP_USER and SMTP_PASSWORD:
    init_email_service(
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        smtp_user=SMTP_USER,
        smtp_password=SMTP_PASSWORD,
        from_email=SMTP_FROM_EMAIL,
        from_name=SMTP_FROM_NAME
    )
    print(f"[OK] Email notification service configured: {SMTP_FROM_EMAIL}")
else:
    print("[WARNING] Email service not configured - email notifications will not work")

# ----------------- Weather API Configuration -----------------
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
WEATHER_CACHE = {}  # Simple in-memory cache: {cache_key: (data, timestamp)}
WEATHER_CACHE_TTL = 600  # 10 minutes

if OPENWEATHER_API_KEY:
    masked_key = f"{OPENWEATHER_API_KEY[:8]}***{OPENWEATHER_API_KEY[-4:]}" if len(OPENWEATHER_API_KEY) > 12 else "***"
    print(f"[OK] OpenWeatherMap API configured: {masked_key}")
else:
    print("[WARNING] OpenWeatherMap API key not configured - weather features will not work")

# ----------------- Jinja2 Custom Filters -----------------
@app.template_filter('format_number')
def format_number_filter(value):
    """Format number with thousand separators for Vietnamese."""
    try:
        return f"{int(value):,}".replace(',', '.')
    except (ValueError, TypeError):
        return value

# Giới hạn kích thước file tải lên để tránh chấp nhận file quá lớn (16 MB)

# ----------------- ADMIN AUTHENTICATION -----------------
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')  # ĐỔI MẬT KHẨU NÀY!

# Khởi tạo FULL_MODEL_MAP là dict rỗng (sẽ được điền sau khi setup logging)
FULL_MODEL_MAP = {}

def check_auth(username, password):
    """Check if username/password combination is valid."""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def authenticate():
    """Send 401 response that enables basic auth."""
    response = make_response(
        'Yêu cầu xác thực.\n'
        'Vui lòng đăng nhập với tài khoản quản trị viên.', 401
    )
    response.headers['WWW-Authenticate'] = 'Basic realm="Admin Access Required"'
    return response

def requires_admin_auth(f):
    """Decorator to require admin authentication (session or HTTP Basic Auth)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Kiểm tra session trước - nếu đã đăng nhập qua form và là admin
        if session.get('is_admin'):
            return f(*args, **kwargs)
        
        # Nếu chưa có session admin, thử HTTP Basic Auth
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            app.logger.warning(
                "Unauthorized admin access attempt from %s to %s",
                request.remote_addr, request.path
            )
            # Chuyển hướng về trang login thay vì hiện HTTP Basic Auth dialog
            flash('Vui lòng đăng nhập với tài khoản quản trị viên', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Kích thước tối đa cho phép của ảnh (cạnh lớn nhất). Có thể ghi đè bằng env var.
app.config['MAX_IMAGE_DIM'] = int(os.environ.get('MAX_IMAGE_DIM', '3000'))

# Thiết lập hệ thống logging
setup_logging(app)

# Warn if using default secret key
if app.secret_key == 'a-strong-default-secret-key-for-development-only':
    app.logger.warning(
        'Using default SECRET_KEY. Set SECRET_KEY env var for production to secure sessions.'
    )

# Log configuration on startup
app.logger.info('Application configuration:')
app.logger.info('  MAX_CONTENT_LENGTH: %s MB', app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024))
app.logger.info('  MAX_IMAGE_DIM: %s', app.config['MAX_IMAGE_DIM'])
app.logger.info('  MIN_MODEL_CONF: %.2f', MIN_MODEL_CONF)
app.logger.info('  POS_SIM_THRESH: %.2f', POS_SIM_THRESH)
app.logger.info('  NEG_SIM_THRESH: %.2f', NEG_SIM_THRESH)
app.logger.info('  MAX_LOADED_MODELS: %s', MAX_LOADED_MODELS)

# Phát hiện và tải cấu hình model
FULL_MODEL_MAP = discover_models(MODELS_DIR, ARCHITECTURES, app.logger)
app.logger.info('Discovered %d model configurations', len(FULL_MODEL_MAP))

# Model LRU Cache với tự động dọn dẹp
class ModelLRUCache:
    """LRU cache an toàn thread cho các ML model với quản lý bộ nhớ tự động."""
    
    def __init__(self, max_size=2):
        self.max_size = max_size
        self.cache = {}  # Python 3.7+ dict maintains insertion order
        self.lock = threading.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size': 0
        }
    
    def get(self, key):
        """Lấy model từ cache, chuyển nó về cuối (mới sử dụng gần đây nhất)."""
        with self.lock:
            if key in self.cache:
                # Pop and re-insert to move to end (mark as recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats['hits'] += 1
                app.logger.debug("Cache HIT for %s", key)
                return value
            else:
                self.stats['misses'] += 1
                app.logger.debug("Cache MISS for %s", key)
                return None
    
    def put(self, key, value):
        """Thêm model vào cache, loại bỏ LRU nếu cần thiết."""
        with self.lock:
            if key in self.cache:
                # Xóa và thêm lại để chuyển về cuối
                self.cache.pop(key)
                self.cache[key] = value
                app.logger.debug("Cache UPDATE for %s", key)
            else:
                # Thêm mục mới
                if len(self.cache) >= self.max_size:
                    # Loại bỏ mục ít sử dụng gần đây nhất (mục đầu tiên trong dict)
                    evicted_key = next(iter(self.cache))
                    evicted_value = self.cache.pop(evicted_key)
                    self._cleanup_model(evicted_key, evicted_value)
                    self.stats['evictions'] += 1
                    app.logger.info(
                        "Cache EVICTION: %s (cache full: %d/%d)",
                        evicted_key, len(self.cache), self.max_size
                    )
                
                self.cache[key] = value
                self.stats['current_size'] = len(self.cache)
                app.logger.debug(
                    "Cache ADD: %s (size: %d/%d)",
                    key, len(self.cache), self.max_size
                )
    
    def _cleanup_model(self, key, value):
        """Dọn dẹp tài nguyên model (bộ nhớ, TensorFlow sessions, v.v.)."""
        try:
            model, class_names = value
            # Xóa Keras/TensorFlow backend session
            if hasattr(model, '_keras_api_names'):
                try:
                    # Đối với các model TensorFlow/Keras
                    import gc
                    del model
                    gc.collect()
                    # Xóa TensorFlow session nếu khả dụng
                    try:
                        import tensorflow as tf
                        tf.keras.backend.clear_session()
                    except Exception:
                        pass
                except Exception as e:
                    app.logger.warning("Error during model cleanup: %s", str(e))
            
            app.logger.info("Model cleaned up: %s", key)
        except Exception as e:
            app.logger.error("Failed to cleanup model %s: %s", key, str(e))
    
    def clear(self):
        """Xóa tất cả các model đã cache."""
        with self.lock:
            for key, value in list(self.cache.items()):
                self._cleanup_model(key, value)
            self.cache.clear()
            self.stats['current_size'] = 0
            app.logger.info("Cache cleared")
    
    def get_stats(self):
        """Lấy thống kê cache."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'max_size': self.max_size,
                'keys': list(self.cache.keys())
            }
    
    def __contains__(self, key):
        """Kiểm tra key có tồn tại trong cache không."""
        with self.lock:
            return key in self.cache
    
    def __len__(self):
        """Lấy kích thước cache."""
        with self.lock:
            return len(self.cache)

# ============= CHATBOT CONFIGURATION =============
# Constants
CHAT_MAX_QUESTION_LENGTH = 500
CHAT_MIN_ANSWER_LENGTH = 20
CHAT_MAX_TOKENS = 800
CHAT_API_TIMEOUT = 30  # seconds
CHAT_RATE_LIMIT_PER_MINUTE = 10
CHAT_RETRY_ATTEMPTS = 3
CHAT_RETRY_DELAY = 2  # seconds

# Rate limiting storage
CHAT_RATE_LIMITER = defaultdict(list)  # IP -> [timestamps]

# FAQ Cache - Câu hỏi thường gặp (fallback khi API fail)
FAQ_RESPONSES = {
    "cà chua là gì": "Cà chua (Solanum lycopersicum) là loại cây trồng thuộc họ Cà (Solanaceae), có nguồn gốc từ Nam Mỹ. Quả cà chua giàu vitamin C, lycopene và các chất chống oxi hóa, rất tốt cho sức khỏe. 🍅",
    "bệnh cháy sớm": "Bệnh cháy sớm (Early blight) do nấm Alternaria solani gây ra, xuất hiện các đốm màu nâu đậm trên lá, có thể làm lá vàng và rụng, giảm năng suất. Phòng ngừa bằng cách luân canh, loại bỏ lá bệnh và phun thuốc fungicide đúng liều.",
    "bệnh cháy muộn": "Bệnh cháy muộn (Late blight) do nấm Phytophthora infestans, là bệnh nguy hiểm nhất. Triệu chứng: đốm nâu đen lan nhanh trên lá, thân và quả. Phòng ngừa: dùng giống kháng, tránh trồng gần khoai tây, phun thuốc đúng lúc.",
    "bệnh đốm lá septoria": "Bệnh đốm lá Septoria do nấm Septoria lycopersici, xuất hiện nhiều đốm nhỏ tròn có tâm sáng và viền đậm. Phòng ngừa: loại bỏ lá bệnh, tưới nhỏ giọt, luân canh và dùng thuốc bảo vệ thực vật.",
    "bệnh virus xoăn vàng lá": "Bệnh xoăn vàng lá (TYLCV) do virus, truyền qua rầy tàu. Triệu chứng: lá vàng, cuộn quăn, cây lùn. Phòng ngừa: kiểm soát rầy bằng bẫy dính, thuốc trừ sâu, dùng giống kháng và lưới che.",
    "bệnh đốm vi khuẩn": "Bệnh đốm vi khuẩn do Xanthomonas, xuất hiện đốm đen/nâu trên lá, thân, quả. Phòng ngừa: dùng giống kháng, tránh tưới phun lá, loại bỏ cây bệnh, luân canh và phun thuốc chứa đồng.",
    "triệu chứng": "Các triệu chứng bệnh cà chua phổ biến: đốm lá (nâu, đen, vàng), lá xoăn, héo, vàng rụng, quả thối, đốm trên thân. Mỗi bệnh có triệu chứng đặc trưng riêng. Bạn muốn hỏi về bệnh nào cụ thể?",
    "phòng ngừa": "Các biện pháp phòng bệnh: (1) Luân canh cây trồng, (2) Dùng giống kháng bệnh, (3) Tưới nhỏ giọt thay vì phun lên lá, (4) Loại bỏ và tiêu hủy cây bệnh, (5) Bón phân cân đối, (6) Phun thuốc đúng loại đúng lúc, (7) Giữ vườn sạch và thông thoáng.",
    "cách chăm sóc": "Chăm sóc cà chua: (1) Tưới đều, tránh úng, (2) Bón phân NPK cân đối, (3) Tỉa cành, cắt lá già, (4) Kiểm tra sâu bệnh thường xuyên, (5) Đảm bảo ánh sáng đủ 6-8 giờ/ngày, (6) Giữ độ ẩm đất 60-70%, (7) Che phủ gốc để giữ ẩm.",
    "khi nào thu hoạch": "Thu hoạch cà chua khi quả chín từ 70-90% (màu đỏ/vàng tùy giống), còn hơi cứng. Quả quá chín dễ hỏng khi vận chuyển. Thu hoạch buổi sáng sớm hoặc chiều mát để quả tươi lâu hơn. 🍅"
}

# Response cache (hash-based)
RESPONSE_CACHE = {}  # question_hash -> (answer, timestamp)
CACHE_TTL = 3600  # 1 hour

# Khởi tạo model cache
LOADED_MODELS = ModelLRUCache(max_size=MAX_LOADED_MODELS)
MODEL_LOAD_LOCK = threading.Lock()

def check_faq_response(question: str) -> Optional[str]:
    """Kiểm tra xem câu hỏi có match với FAQ không.
    
    Args:
        question: Câu hỏi người dùng
        
    Returns:
        Câu trả lời từ FAQ hoặc None nếu không match
        
    Example:
        >>> check_faq_response("cà chua là gì")
        "Cà chua (Solanum lycopersicum) là loại cây trồng..."
    """
    q_lower = question.lower().strip()
    
    # Exact match hoặc contains
    for key, answer in FAQ_RESPONSES.items():
        if key in q_lower or q_lower in key:
            return answer
    return None


def get_cached_response(question: str) -> Optional[str]:
    """Lấy response từ cache nếu có và chưa expired.
    
    Args:
        question: Câu hỏi người dùng
        
    Returns:
        Cached answer hoặc None nếu không có hoặc đã expired
    """
    q_hash = hashlib.md5(question.lower().encode()).hexdigest()
    
    if q_hash in RESPONSE_CACHE:
        cached_answer, timestamp = RESPONSE_CACHE[q_hash]
        # Kiểm tra TTL
        if time.time() - timestamp < CACHE_TTL:
            return cached_answer
        else:
            # Xóa cache cũ
            del RESPONSE_CACHE[q_hash]
    return None


def cache_response(question: str, answer: str) -> None:
    """Lưu response vào cache với timestamp hiện tại.
    
    Args:
        question: Câu hỏi người dùng
        answer: Câu trả lời cần cache
    """
    q_hash = hashlib.md5(question.lower().encode()).hexdigest()
    RESPONSE_CACHE[q_hash] = (answer, time.time())


def estimate_tokens(text: str) -> int:
    """Ước tính số token trong text (rough estimate: 1 token ≈ 4 chars).
    
    Args:
        text: Text cần ước tính
        
    Returns:
        Số token ước tính
    """
    return len(text) // 4


def call_gemini_with_retry(
    model: Any, 
    prompt: str, 
    config: Any, 
    max_attempts: int = 3
) -> Any:
    """Gọi Gemini API với retry logic (exponential backoff).
    
    Args:
        model: Gemini model instance
        prompt: Prompt string để gửi tới API
        config: GenerationConfig object
        max_attempts: Số lần thử tối đa (default: 3)
        
    Returns:
        Response object từ Gemini API
        
    Raises:
        Exception: Khi tất cả attempts đều fail
    """
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(
                prompt,
                generation_config=config,
                request_options={'timeout': CHAT_API_TIMEOUT}
            )
            return response
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # Không retry cho auth errors
            if "401" in error_msg or "403" in error_msg or "API key" in error_msg:
                raise
            
            # Retry với exponential backoff
            if attempt < max_attempts - 1:
                wait_time = CHAT_RETRY_DELAY * (2 ** attempt)  # 2, 4, 8 seconds
                app.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {error_msg[:100]}")
                time.sleep(wait_time)
            else:
                app.logger.error(f"All {max_attempts} attempts failed")
    
    raise last_error


def get_gemini_response(user_question: str) -> str:
    """Gọi API Gemini để trả lời câu hỏi về cà chua.
    
    Args:
        user_question: Câu hỏi của người dùng
        
    Returns:
        Câu trả lời từ Gemini hoặc thông báo lỗi
    """
    global LAST_WORKING_MODEL
    
    if not GEMINI_API_KEY:
        return "[!] Hệ thống chatbot chưa được cấu hình. Vui lòng liên hệ quản trị viên."
    
    if not GEMINI_MODEL:
        return "[!] Không thể khởi tạo model AI. Vui lòng liên hệ quản trị viên."
    
    # Check FAQ first (nhanh và không tốn quota)
    faq_answer = check_faq_response(user_question)
    if faq_answer:
        app.logger.info("[OK] Answer from FAQ")
        return faq_answer
    
    # Check cache
    cached = get_cached_response(user_question)
    if cached:
        app.logger.info("[OK] Answer from cache")
        return cached
    
    # Danh sách model để thử (ưu tiên model đã thành công)
    model_names = [
        'models/gemini-flash-latest',
        'models/gemini-2.5-flash',
        'models/gemini-2.0-flash',
    ]
    
    # Đưa last working model lên đầu
    if LAST_WORKING_MODEL and LAST_WORKING_MODEL in model_names:
        model_names.remove(LAST_WORKING_MODEL)
        model_names.insert(0, LAST_WORKING_MODEL)
    
    # Tạo prompt
    system_prompt = """Bạn là chuyên gia cà chua. Trả lời HOÀN CHỈNH, TỰ NHIÊN bằng tiếng Việt.

Nếu hỏi về cà chua: Giải thích rõ ràng, cụ thể, đầy đủ (3-5 câu).
Nếu KHÔNG về cà chua: "Xin lỗi, tôi chỉ trả lời về cà chua."

QUAN TRỌNG: 
- Trả lời ĐẦY ĐỦ, KHÔNG bỏ dở giữa chừng
- Dùng ngôn ngữ đời thường, dễ hiểu
- Đi thẳng vào nội dung
- Kết thúc câu trả lời một cách hoàn chỉnh"""
    
    full_prompt = f"{system_prompt}\n\nCâu hỏi: {user_question}\n\nTrả lời:"
    
    # Thử từng model với retry logic
    last_error = None
    for model_name in model_names:
        try:
            app.logger.info(f"Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # Gọi API với retry
            response = call_gemini_with_retry(
                model,
                full_prompt,
                genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=CHAT_MAX_TOKENS,
                    top_p=0.9,
                    candidate_count=1,
                ),
                max_attempts=CHAT_RETRY_ATTEMPTS
            )
            
            if response and response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                
                # Lấy text từ response
                answer = None
                if hasattr(response, 'text') and response.text:
                    answer = response.text.strip()
                elif candidate.content and candidate.content.parts:
                    answer = ''.join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
                
                if answer and len(answer) >= CHAT_MIN_ANSWER_LENGTH:
                    app.logger.info(f"[OK] Success with {model_name} (finish: {finish_reason})")
                    LAST_WORKING_MODEL = model_name
                    # Cache response
                    cache_response(user_question, answer)
                    return answer
                else:
                    app.logger.warning(f"Response too short from {model_name}: {len(answer) if answer else 0} chars")
            
        except Exception as e:
            error_msg = str(e)
            app.logger.warning(f"{model_name} failed: {error_msg[:100]}")
            last_error = error_msg
            
            # Auth errors - dừng ngay
            if "401" in error_msg or "403" in error_msg or "API key" in error_msg:
                return "[AUTH ERROR] API key không hợp lệ. Vui lòng liên hệ quản trị viên."
            
            # Quota errors - thử model khác
            if "429" in error_msg or "quota" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg:
                continue
            
            # Timeout errors
            if "timeout" in error_msg.lower():
                continue
            
            # Các lỗi khác - thử model tiếp theo
            continue
    
    # Tất cả model đều thất bại - fallback FAQ
    app.logger.error(f"All models failed. Last error: {last_error}")
    
    # Thử tìm FAQ gần đúng
    faq_fuzzy = check_faq_response(user_question)
    if faq_fuzzy:
        return faq_fuzzy + "\n\n(Lưu ý: Hệ thống AI tạm thời không khả dụng, câu trả lời trên từ FAQ)"
    
    # Error messages
    if last_error:
        if "429" in last_error or "quota" in last_error.lower() or "RESOURCE_EXHAUSTED" in last_error:
            return ("[!] Hệ thống chatbot tạm thời quá tải (đã hết quota miễn phí). "
                   "Vui lòng thử lại sau hoặc liên hệ quản trị viên.")
        elif "timeout" in last_error.lower():
            return "[!] Hệ thống phản hồi quá chậm. Vui lòng thử lại với câu hỏi ngắn gọn hơn."
    
    return "[!] Không thể kết nối với chatbot. Vui lòng thử lại sau hoặc tham khảo mục Giới thiệu."

# Hàm giả để xử lý lỗi "Could not locate function '_input_preprocess_layer'"
# Lỗi này xảy ra khi model được lưu có chứa custom object (ví dụ: Lambda layer)
# mà không được cung cấp khi tải lại. Chúng ta chỉ cần một placeholder có cùng tên.
def _input_preprocess_layer(x):
    """Hàm giả, không làm gì cả."""
    return x


def _read_labels_from_file(path: Path):
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            return None
        if path.suffix.lower() == '.json':
            import json
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                # try to order by numeric keys
                try:
                    items = sorted(data.items(), key=lambda kv: int(kv[0]))
                    return [v for k, v in items]
                except Exception:
                    return list(data.values())
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                return lines
    except Exception:
        app.logger.exception("Error reading label file %s", path)
        return None
    return None

def _find_class_names_for_model(model_path: Path):
    # Tìm các file label phổ biến bên cạnh model
    candidates = [
        model_path.with_name('class_names.txt'),
        model_path.with_name('classes.txt'),
        model_path.with_name('labels.txt'),
        model_path.with_name('labels.json'),
        model_path.with_name('class_map.json'),
        model_path.with_suffix('.labels.txt'),
        model_path.with_suffix('.classes.txt'),
        model_path.with_suffix('.json'),
    ]
    for c in candidates:
        names = _read_labels_from_file(c)
        if names:
            app.logger.info("Loaded class names from %s", c)
            return names
    # Thử thư mục cha
    parent = model_path.parent
    for fname in ('class_names.txt', 'classes.txt', 'labels.txt', 'labels.json'):
        names = _read_labels_from_file(parent / fname)
        if names:
            app.logger.info("Loaded class names from %s", parent / fname)
            return names
    app.logger.warning("No label file found for model %s; falling back to global CLASS_NAMES", model_path)
    return CLASS_NAMES

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def load_model_by_name(arch_name, pipeline_key):
    """Tải model theo tên kiến trúc và khóa pipeline với xử lý lỗi đúng cách."""
    key = (arch_name, pipeline_key)
    
    try:
        # Kiểm tra LRU cache trước
        cached = LOADED_MODELS.get(key)
        if cached is not None:
            app.logger.debug("Model loaded from LRU cache: %s + %s", arch_name, pipeline_key)
            return cached
        
        # Xác thực model tồn tại trong bản đồ
        if key not in FULL_MODEL_MAP:
            available_keys = list(FULL_MODEL_MAP.keys())
            app.logger.error(
                "Model not found in map. Requested: %s + %s. Available: %s",
                arch_name, pipeline_key, available_keys
            )
            raise ModelError(
                f"Model configuration not found: {arch_name} + {pipeline_key}",
                details={'arch_name': arch_name, 'pipeline_key': pipeline_key, 'available': available_keys},
                user_message=f"Không tìm thấy model {arch_name} với pipeline {pipeline_key}"
            )
        
        model_path = FULL_MODEL_MAP[key]
        
        # Kiểm tra file có tồn tại không
        if not model_path.exists():
            app.logger.error("Model file not found at path: %s", model_path)
            raise ModelError(
                f"Model file not found: {model_path}",
                details={'path': str(model_path)},
                user_message="File model không tồn tại"
            )
        
        app.logger.info("Loading model: %s (size: %.2f MB)", model_path, model_path.stat().st_size / (1024*1024))
        
        # Bảo vệ quá trình tải bằng lock
        with MODEL_LOAD_LOCK:
            # Kiểm tra lại cache sau khi lấy lock
            cached = LOADED_MODELS.get(key)
            if cached is not None:
                app.logger.debug("Model was loaded by another thread")
                return cached
            
            def _try_load(path):
                return tf.keras.models.load_model(
                    str(path),
                    custom_objects={'_input_preprocess_layer': _input_preprocess_layer},
                    compile=False
                )
            
            # Thử tải model
            model = None
            try:
                model = _try_load(model_path)
                app.logger.info("Model loaded successfully: %s", model_path.name)
            except Exception as first_exc:
                app.logger.warning(
                    "Initial model load failed for %s: %s", 
                    model_path.name, str(first_exc)
                )
                
                # Thử sửa chữa cho các file .keras
                try:
                    repaired_path = _attempt_repair_keras(model_path)
                    if repaired_path:
                        app.logger.info("Attempting to load repaired model: %s", repaired_path.name)
                        model = _try_load(repaired_path)
                        app.logger.info("Repaired model loaded successfully")
                    else:
                        raise first_exc
                except Exception as second_exc:
                    app.logger.error(
                        "Failed to load model after repair attempt. Original error: %s, Repair error: %s",
                        str(first_exc), str(second_exc),
                        exc_info=True
                    )
                    raise ModelError(
                        f"Cannot load model: {model_path.name}",
                        details={
                            'path': str(model_path),
                            'original_error': str(first_exc),
                            'repair_error': str(second_exc)
                        },
                        user_message="Không thể tải model. Vui lòng kiểm tra file model."
                    ) from first_exc
            
            # Tải danh sách tên lớp
            try:
                class_names = _find_class_names_for_model(model_path)
                app.logger.info("Class names loaded: %s classes", len(class_names))
            except Exception as e:
                app.logger.error("Failed to load class names: %s", str(e), exc_info=True)
                raise ModelError(
                    "Failed to load class names",
                    details={'path': str(model_path), 'error': str(e)},
                    user_message="Không thể tải danh sách labels"
                ) from e
            
            # Cache the model (LRU will auto-evict if needed)
            LOADED_MODELS.put(key, (model, class_names))
            cache_stats = LOADED_MODELS.get_stats()
            app.logger.info(
                "Model cached. Cache stats - size: %d/%d, hit_rate: %.1f%%, evictions: %d",
                cache_stats['size'], cache_stats['max_size'],
                cache_stats['hit_rate'], cache_stats['evictions']
            )
            
            return (model, class_names)
    
    except ModelError:
        raise
    except Exception as e:
        app.logger.exception("Unexpected error in load_model_by_name")
        raise ModelError(
            "Unexpected error loading model",
            details={'arch_name': arch_name, 'pipeline_key': pipeline_key, 'error': str(e)},
            user_message="Lỗi không xác định khi tải model"
        ) from e


def _attempt_repair_keras(model_path: Path):
    """Attempt a non-destructive repair of a .keras (zip) file by adjusting
    any 4-D shape entries whose channel dimension is 1 to 3 inside `config.json`.
    Writes a new file next to the original with suffix `_repaired.keras` and
    returns its Path on success, or None if repair not possible.

    This is a conservative heuristic to fix common mismatches where saved
    config accidentally contains single-channel shapes. It does not alter
    weights and never overwrites the original file.
    """
    try:
        if not model_path.exists():
            return None
        suffix = model_path.suffix.lower()
        if suffix != '.keras':
            return None
        import zipfile, tempfile
        repaired_file = model_path.with_name(model_path.stem + '_repaired' + model_path.suffix)
        with zipfile.ZipFile(model_path, 'r') as zin:
            names = zin.namelist()
            if 'config.json' not in names:
                return None
            cfg_bytes = zin.read('config.json')
            cfg_text = cfg_bytes.decode('utf-8')
            cfg = json.loads(cfg_text)

            modified = False

            def walk_and_fix(o):
                nonlocal modified
                if isinstance(o, dict):
                    for k, v in list(o.items()):
                        # If value is a list like [None, H, W, C] and C == 1 -> set to 3
                        if isinstance(v, list) and len(v) >= 4:
                            try:
                                ch = v[3]
                                if ch == 1:
                                    o[k] = [v[0], v[1], v[2], 3]
                                    modified = True
                            except Exception:
                                pass
                        # Nếu kích thước không gian là 225 nhưng IMG_SIZE của chúng ta khác, chỉnh sửa chúng
                        try:
                            if isinstance(v, list) and len(v) >= 4 and isinstance(v[1], int) and isinstance(v[2], int):
                                if (v[1] == 225 and v[2] == 225) and (v[1] != IMG_SIZE[0] or v[2] != IMG_SIZE[1]):
                                    o[k] = [v[0], IMG_SIZE[0], IMG_SIZE[1], v[3]]
                                    modified = True
                        except Exception:
                            pass
                        walk_and_fix(v)
                elif isinstance(o, list):
                    for i, item in enumerate(o):
                        if isinstance(item, list) and len(item) >= 4:
                            try:
                                if item[3] == 1:
                                    o[i] = [item[0], item[1], item[2], 3]
                                    modified = True
                            except Exception:
                                pass
                        try:
                            if isinstance(item, list) and len(item) >= 4 and isinstance(item[1], int) and isinstance(item[2], int):
                                if (item[1] == 225 and item[2] == 225) and (item[1] != IMG_SIZE[0] or item[2] != IMG_SIZE[1]):
                                    o[i] = [item[0], IMG_SIZE[0], IMG_SIZE[1], item[3]]
                                    modified = True
                        except Exception:
                            pass
                        walk_and_fix(item)

            walk_and_fix(cfg)

            if not modified:
                return None

            # Write repaired archive
            with zipfile.ZipFile(repaired_file, 'w') as zout:
                # Sao chép tất cả các file trừ config.json
                for name in names:
                    if name == 'config.json':
                        continue
                    zout.writestr(name, zin.read(name))
                # Ghi config.json đã chỉnh sửa
                zout.writestr('config.json', json.dumps(cfg).encode('utf-8'))

        return repaired_file
    except Exception:
        app.logger.exception('Error while attempting to repair .keras file: %s', model_path)
        return None

def preprocess_image_for_model(image_bgr, pipeline_key):
    """
    image_bgr: numpy array read by OpenCV (BGR)
    pipeline_key: one of PIPELINES keys
    Returns: numpy array shape (1, H, W, 3) float32 (same scaling as training pipelines)
    """
    # Convert BGR (cv2) -> RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Resize to target IMG_SIZE (height, width)
    # cv2.resize expects (width, height)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
    # Xác thực pipeline_key và áp dụng pipeline
    if pipeline_key not in PIPELINES:
        raise ValueError(f"Unknown pipeline key: {pipeline_key}")
    fn, _ = PIPELINES[pipeline_key]
    proc = fn(img_resized)  # returns float32 in [0,1] (we kept same as training)
    # Đảm bảo hình dạng (H,W,3)
    if proc is None:
        raise ValueError(f"Pipeline '{pipeline_key}' returned None for the input image")
    if isinstance(proc, np.ndarray):
        # force float32
        proc = proc.astype('float32')
        if proc.ndim == 2: # Nếu là ảnh xám (H, W)
            proc = np.stack([proc]*3, axis=-1) # Chuyển thành (H, W, 3)
        elif proc.ndim == 3: # Nếu là (H, W, C)
            ch = proc.shape[-1]
            if ch == 1:
                # duplicate single channel to RGB
                proc = np.concatenate([proc, proc, proc], axis=2)
            elif ch > 3:
                # Giữ 3 kênh đầu tiên
                proc = proc[..., :3]
        else:
            raise ValueError(f"Pipeline '{pipeline_key}' returned array with unsupported ndim={proc.ndim}")
    else:
        raise TypeError(f"Pipeline '{pipeline_key}' returned non-array type: {type(proc)}")

    out = proc # Bắt đầu với kết quả từ pipeline
    # If pipeline produced values in 0-255, normalize to [0,1]
    if out.max() > 2.0:
        out = out.astype('float32') / 255.0

    # Ensure final spatial size matches IMG_SIZE exactly; if not, resize
    if out.shape[0] != IMG_SIZE[0] or out.shape[1] != IMG_SIZE[1]:
        try:
            # cv2.resize expects uint8 or float; scale back to 0-255 uint8 for resizing stability
            tmp = (np.clip(out, 0.0, 1.0) * 255.0).astype('uint8')
            tmp_resized = cv2.resize(tmp, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
            out = (tmp_resized.astype('float32') / 255.0)
        except Exception:
            # fallback: use numpy-based resize by simple cropping/pad if cv2 fails
            h, w = out.shape[:2]
            target_h, target_w = IMG_SIZE
            if h < target_h or w < target_w:
                # pad
                pad_h = max(0, target_h - h)
                pad_w = max(0, target_w - w)
                out = np.pad(out, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')[:target_h, :target_w, :]
            else:
                out = out[:target_h, :target_w, :]

    # Kiểm tra lại số kênh
    if out.ndim == 3 and out.shape[-1] == 1:
        out = np.concatenate([out, out, out], axis=2)

    # Kiểm tra cuối cùng
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"Preprocessed image has wrong shape {out.shape}; expected (H,W,3)")

    # Diagnostic logs
    try:
        app.logger.info("Preprocess: pipeline=%s, proc.shape=%s, dtype=%s, min=%.5f, max=%.5f", pipeline_key, proc.shape, proc.dtype, float(proc.min()), float(proc.max()))
        app.logger.info("Preprocess: final out.shape=%s, dtype=%s, min=%.5f, max=%.5f", out.shape, out.dtype, float(out.min()), float(out.max()))
    except Exception:
        # ignore logging errors
        pass

    return np.expand_dims(out.astype('float32'), axis=0)

# --- Tối ưu hóa: Tải trước cache đặc trưng của ảnh mẫu ---
CACHED_SAMPLE_FEATURES = None

def _load_sample_features_cache():
    """Tải các đặc trưng đã được tính toán trước của ảnh mẫu từ file pickle."""
    global CACHED_SAMPLE_FEATURES
    if CACHED_SAMPLE_FEATURES is not None:
        return CACHED_SAMPLE_FEATURES
    
    cache_file = DATA_DIR / 'sample_features.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            CACHED_SAMPLE_FEATURES = pickle.load(f)
        num_pos = len(CACHED_SAMPLE_FEATURES.get('positive', []))
        num_neg = len(CACHED_SAMPLE_FEATURES.get('negative', []))
        app.logger.info(f"Đã tải cache đặc trưng ảnh mẫu thành công: {num_pos} mẫu tích cực, {num_neg} mẫu tiêu cực")
    return CACHED_SAMPLE_FEATURES

def is_leaf_like(image_bgr, green_ratio_threshold=0.05):
    """
    Heuristic check whether the image contains a leaf-like amount of green.
    Returns True if fraction of 'green' pixels exceeds green_ratio_threshold.

    This is a lightweight filter (not a classifier). It converts to HSV and
    counts pixels in a green hue range with sufficient saturation/value.
    """
    try:
        # Resize to speed up calculation while keeping ratio
        small = cv2.resize(image_bgr, (224, 224))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # OpenCV H range is [0,179]. Green hue approx between 25 and 95.
        lower_h, upper_h = 25, 95
        # Require moderate saturation and brightness to avoid very pale/wet backgrounds
        sat_thresh = 40
        val_thresh = 40
        mask_h = (h >= lower_h) & (h <= upper_h)
        mask_sv = (s >= sat_thresh) & (v >= val_thresh)
        mask = mask_h & mask_sv
        green_count = int(np.count_nonzero(mask))
        total = mask.size
        ratio = green_count / float(total)
        return ratio >= green_ratio_threshold
    except Exception:
        # If any error occurs, log and be conservative (allow the image)
        app.logger.exception("Error in is_leaf_like heuristic")
        return True

def compute_sample_similarity(img_bgr):
    """
    Compute similarity metrics against positive (tomato) and negative (not_tomato) samples.
    Returns a dict with keys like 'positive_sim', 'negative_sim', 'combined_score'.
    """
    try:
        # Tải cache đặc trưng của các ảnh mẫu
        sample_features = _load_sample_features_cache()
        if not sample_features or (not sample_features['positive'] and not sample_features['negative']):
            return {'has_samples': False}

        # Tính toán đặc trưng cho ảnh đầu vào
        inp_hist = compute_hist(img_bgr)
        try:
            inp_emb = _compute_embedding(img_bgr)
        except Exception:
            app.logger.exception("Error computing input embedding")
            inp_emb = None

        # Helper: normalize vector to unit length safely
        def _norm_vec(v):
            try:
                a = np.asarray(v, dtype=np.float32)
                n = np.linalg.norm(a)
                if n <= 0:
                    return None
                return a / (n + 1e-9)
            except Exception:
                return None

        if inp_emb is not None:
            inp_emb = _norm_vec(inp_emb)

        def get_max_sim_for_category(category_features):
            max_hist_sim, max_deep_sim = 0.0, 0.0
            
            for features in category_features:
                # So sánh Histogram
                try:
                    sim_hist = cv2.compareHist(inp_hist, features['hist'], cv2.HISTCMP_CORREL)
                except Exception:
                    sim_hist = 0.0
                if sim_hist > max_hist_sim:
                    max_hist_sim = sim_hist

                # So sánh Deep Embedding (chuẩn hóa trước khi dot)
                try:
                    feat_emb = features.get('embedding')
                    if inp_emb is not None and feat_emb is not None:
                        n_feat = _norm_vec(feat_emb)
                        if n_feat is not None:
                            sim_deep = float(np.dot(inp_emb, n_feat))
                            if sim_deep > max_deep_sim:
                                max_deep_sim = sim_deep
                except Exception:
                    app.logger.exception('Error comparing deep embeddings')
                    # ignore and continue
            return max_hist_sim, max_deep_sim

        pos_hist, pos_deep = get_max_sim_for_category(sample_features.get('positive', []))
        neg_hist, neg_deep = get_max_sim_for_category(sample_features.get('negative', []))

        def combine_scores(hist, deep):
            hist_norm = float(max(0.0, min(1.0, hist)))
            deep_norm = float((deep + 1.0) / 2.0) # scale from [-1,1] to [0,1]
            w_deep, w_hist = 0.5, 0.5
            return w_deep * deep_norm + w_hist * hist_norm

        positive_sim = combine_scores(pos_hist, pos_deep)
        negative_sim = combine_scores(neg_hist, neg_deep)
        
        app.logger.info(f"Similarity computed: pos_sim={positive_sim:.3f} (hist={pos_hist:.3f}, deep={pos_deep:.3f}), "
                       f"neg_sim={negative_sim:.3f} (hist={neg_hist:.3f}, deep={neg_deep:.3f})")

        return {
            'has_samples': True,
            'positive_sim': positive_sim,
            'negative_sim': negative_sim,
            'combined_score': positive_sim - negative_sim,
            'details': {
                'pos': {'hist': pos_hist, 'deep': pos_deep},
                'neg': {'hist': neg_hist, 'deep': neg_deep}
            }
        }
    except Exception as e:
        app.logger.error(f"Error in compute_sample_similarity: {e}")
        return {'has_samples': False}

def revert_for_display(array_3ch, pipeline_key):
    """
    array_3ch: numpy array HxWx3 either in float [0,1] (pipelines produce this)
    Return uint8 RGB image for showing in browser
    """
    img = (array_3ch * 255.0).astype(np.uint8)
    if pipeline_key in ['gb_noise_cmyk', 'median_cmyk']:
        # Convert CMY + K approx -> RGB (approx inverse)
        C = img[...,0].astype(np.int32)
        M = img[...,1].astype(np.int32)
        Y = img[...,2].astype(np.int32)
        K = 255 - np.maximum.reduce([C, M, Y])
        R = 255 - np.clip(C + K, 0, 255)
        G = 255 - np.clip(M + K, 0, 255)
        B = 255 - np.clip(Y + K, 0, 255)
        out = np.stack([R, G, B], axis=2).astype(np.uint8)
    elif pipeline_key in ['gb_noise_hsi', 'median_hsi']:
        # Không gian màu HSI tự định nghĩa rất khó để đảo ngược chính xác.
        # Cách tiếp cận thực tế là coi 3 kênh H, S, I như là H, S, V và chuyển đổi
        # bằng OpenCV để có hình ảnh đại diện.
        # Đầu vào `img` là uint8, các kênh H, S, I đã được scale về [0, 255].
        # Để dùng cvtColor, cần scale lại H về [0, 179].
        try:
            hsv_like = img.copy()
            hsv_like[..., 0] = (hsv_like[..., 0] / 255.0 * 179).astype(np.uint8)
            out = cv2.cvtColor(hsv_like, cv2.COLOR_HSV2RGB)
        except Exception:
            out = img
    elif pipeline_key in ['bilateral_hsv', 'average_hsv']:
        # Chuyển từ HSV sang BGR, rồi BGR sang RGB
        out_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    # elif pipeline_key in ['bilateral_lab', 'average_lab']:
    #     # Chuyển từ LAB sang BGR, rồi BGR sang RGB
    #     out_bgr = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    #     out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    else:
        out = img
    return out

@app.route('/', methods=['GET', 'POST'])
def index():
    # Sử dụng ARCHITECTURES mới
    last_model = session.get('last_model')
    last_pipeline = session.get('last_pipeline')
    models_list = ARCHITECTURES
    pipelines_list = list(PIPELINES.keys())
    return render_template('index.html',
                           models=models_list,
                           pipelines=pipelines_list,
                           last_model=last_model,
                           last_pipeline=last_pipeline)

@app.route('/about')
def about_page():
    """Render about page"""
    return render_template('about.html')

def _read_history_file():
    """Helper function to read and parse history file"""
    history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
    history_list = []
    
    if not history_file.exists():
        return history_list
    
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Parse timestamp
                try:
                    ts = datetime.fromisoformat(entry.get('timestamp', ''))
                    entry['formatted_time'] = ts.strftime('%d/%m/%Y %H:%M:%S')
                    entry['timestamp_obj'] = ts
                except:
                    entry['formatted_time'] = entry.get('timestamp', 'N/A')
                    entry['timestamp_obj'] = None
                
                # Get disease info
                label = entry.get('predicted_label', '')
                if label in DISEASE_INFO:
                    entry['disease_name'] = DISEASE_INFO[label]['name']
                else:
                    entry['disease_name'] = label
                
                # Thêm confidence field từ probability nếu chưa có
                if 'confidence' not in entry and 'probability' in entry:
                    entry['confidence'] = entry['probability']
                elif 'confidence' not in entry:
                    entry['confidence'] = 0.0
                
                history_list.append(entry)
            except json.JSONDecodeError:
                continue
    
    return history_list

@app.route('/history')
def history():
    """Hiển thị lịch sử dự đoán"""
    try:
        history_list = _read_history_file()
        
        # Nếu không phải admin, chỉ hiện lịch sử của user hiện tại
        if not session.get('is_admin'):
            current_user_id = session.get('user_id')
            current_user_email = session.get('user_email')
            
            # Filter lịch sử - CHỈ hiện entry của user hiện tại
            # Không hiện các entry cũ không có user_id (để tránh nhầm lẫn)
            history_list = [
                entry for entry in history_list
                if (entry.get('user_id') == current_user_id or 
                    entry.get('user_email') == current_user_email) and
                   (entry.get('user_id') or entry.get('user_email'))  # Chỉ lấy entry có user info
            ]
        
        # Sắp xếp theo thời gian mới nhất trước
        history_list.reverse()
        return render_template('history.html', history=history_list)
    except Exception as e:
        app.logger.exception('Error loading history')
        flash('Không thể tải lịch sử dự đoán')
        return redirect(url_for('index'))


# ============= GAME / ENTERTAINMENT ROUTES =============

# Quiz Questions Database
QUIZ_QUESTIONS = [
    # Easy Questions (1-3)
    {
        'id': 1,
        'difficulty': 'easy',
        'question': 'Cà chua thuộc họ thực vật nào?',
        'options': ['Họ Cà (Solanaceae)', 'Họ Đậu (Fabaceae)', 'Họ Bầu bí (Cucurbitaceae)', 'Họ Hoa hồng (Rosaceae)'],
        'correct': 0,
        'hint': 'Cùng họ với khoai tây, ớt và cà tím'
    },
    {
        'id': 2,
        'difficulty': 'easy',
        'question': 'Bệnh nào sau đây KHÔNG phải là bệnh phổ biến trên cà chua?',
        'options': ['Late Blight', 'Early Blight', 'Bệnh đốm nhỏ', 'Bệnh khảm lá'],
        'correct': 2,
        'hint': 'Bệnh đốm nhỏ thường xuất hiện ở lúa'
    },
    {
        'id': 3,
        'difficulty': 'easy',
        'question': 'Triệu chứng nào là dấu hiệu của cà chua khỏe mạnh?',
        'options': ['Lá vàng úa', 'Lá xanh đậm, không có đốm', 'Lá cuộn và héo', 'Lá có nhiều vết đen'],
        'correct': 1,
        'hint': 'Khỏe mạnh = màu xanh tươi'
    },
    # Medium Questions (4-6)
    {
        'id': 4,
        'difficulty': 'medium',
        'question': 'Late Blight (bệnh mốc sương) do tác nhân nào gây ra?',
        'options': ['Vi khuẩn', 'Nấm (Phytophthora infestans)', 'Virus', 'Côn trùng'],
        'correct': 1,
        'hint': 'Tên khoa học: Phytophthora infestans'
    },
    {
        'id': 5,
        'difficulty': 'medium',
        'question': 'Điều kiện nào thuận lợi cho sự phát triển của Late Blight?',
        'options': ['Khô hanh, nắng gắt', 'Ẩm ướt, nhiệt độ 15-25°C', 'Rất lạnh dưới 5°C', 'Gió mạnh, khô ráo'],
        'correct': 1,
        'hint': 'Bệnh mốc thích môi trường ẩm'
    },
    {
        'id': 6,
        'difficulty': 'medium',
        'question': 'Early Blight thường bắt đầu xuất hiện ở đâu trên cây cà chua?',
        'options': ['Đỉnh cây', 'Lá già ở phần dưới cây', 'Hoa', 'Rễ'],
        'correct': 1,
        'hint': 'Bệnh tiến triển từ dưới lên trên'
    },
    # Hard Questions (7-10)
    {
        'id': 7,
        'difficulty': 'hard',
        'question': 'Septoria Leaf Spot có đặc điểm gì để phân biệt?',
        'options': ['Đốm lớn màu nâu', 'Đốm nhỏ tròn với viền đen và tâm trắng xám', 'Lá cuộn lại', 'Vết vàng lan rộng'],
        'correct': 1,
        'hint': 'Có hình dạng đặc trưng: tâm sáng, viền tối'
    },
    {
        'id': 8,
        'difficulty': 'hard',
        'question': 'Virus TYLCV (Tomato Yellow Leaf Curl Virus) lây lan qua con đường nào?',
        'options': ['Gió và mưa', 'Ruồi trắng (Whitefly - Bemisia tabaci)', 'Đất nhiễm bệnh', 'Hạt giống'],
        'correct': 1,
        'hint': 'Vector truyền bệnh là một loài côn trùng nhỏ màu trắng'
    },
    {
        'id': 9,
        'difficulty': 'hard',
        'question': 'Biện pháp nào HIỆU QUẢ NHẤT để phòng ngừa Late Blight?',
        'options': ['Chỉ tưới nước vào buổi sáng, tránh ẩm ướt kéo dài', 'Bón nhiều đạm', 'Trồng dày đặc', 'Tưới nước buổi tối'],
        'correct': 0,
        'hint': 'Kiểm soát độ ẩm là chìa khóa'
    },
    {
        'id': 10,
        'difficulty': 'hard',
        'question': 'Target Spot (Corynespora cassiicola) có đặc điểm nào sau đây?',
        'options': ['Chỉ tấn công rễ', 'Đốm có dạng vòng tròn đồng tâm giống bia bắn', 'Lá chuyển màu tím', 'Chỉ xuất hiện vào mùa đông'],
        'correct': 1,
        'hint': 'Tên gọi "Target" gợi ý hình dạng như mục tiêu bắn'
    }
]

def _get_high_scores():
    """Load high scores from file"""
    try:
        scores_file = BASE_DIR / 'data' / 'quiz_scores.jsonl'
        if not scores_file.exists():
            return []
        
        scores = []
        with open(scores_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    scores.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        # Sort by score desc, time asc
        scores.sort(key=lambda x: (-x.get('score', 0), x.get('timestamp', '')))
        return scores[:10]  # Top 10
    except Exception:
        app.logger.exception('Error loading quiz scores')
        return []

def _save_quiz_score(player_name, score, correct_answers, total_questions, time_taken):
    """Save quiz score to file"""
    try:
        scores_dir = BASE_DIR / 'data'
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_file = scores_dir / 'quiz_scores.jsonl'
        
        entry = {
            'id': str(uuid4())[:8],
            'player_name': player_name,
            'score': score,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'timestamp': datetime.utcnow().isoformat(),
            'time_taken': time_taken
        }
        
        with open(scores_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return True
    except Exception:
        app.logger.exception('Error saving quiz score')
        return False

def _generate_voucher(player_name):
    """Generate discount voucher"""
    voucher_code = f"TOMATO{str(uuid4())[:8].upper()}"
    voucher_data = {
        'code': voucher_code,
        'player_name': player_name,
        'discount': '15%',
        'expires': (datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d'),
        'generated': datetime.utcnow().isoformat(),
        'description': 'Giảm giá 15% khi mua phân bón và thuốc trừ sâu cho cà chua'
    }
    
    try:
        vouchers_dir = BASE_DIR / 'data'
        vouchers_dir.mkdir(parents=True, exist_ok=True)
        vouchers_file = vouchers_dir / 'vouchers.jsonl'
        
        with open(vouchers_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(voucher_data, ensure_ascii=False) + '\n')
        
        return voucher_data
    except Exception:
        app.logger.exception('Error generating voucher')
        return None

@app.route('/game')
def game_menu():
    """Game menu page"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    high_scores = _get_high_scores()
    memory_high_scores = _get_memory_high_scores()
    return render_template('game_menu.html', high_scores=high_scores, memory_high_scores=memory_high_scores)

@app.route('/game/quiz')
def quiz_game():
    """Tomato quiz game"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    return render_template('quiz.html', questions=QUIZ_QUESTIONS)

@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz():
    """Submit quiz answers and get results"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', 'Anonymous').strip()[:50]
        answers = data.get('answers', {})
        time_taken = data.get('time_taken', 0)
        
        if not player_name:
            player_name = 'Anonymous'
        
        # Calculate score
        correct_count = 0
        total_questions = len(QUIZ_QUESTIONS)
        
        for question in QUIZ_QUESTIONS:
            qid = str(question['id'])
            if qid in answers and answers[qid] == question['correct']:
                correct_count += 1
        
        # Calculate score (100 points per correct answer)
        score = correct_count * 100
        
        # Check for perfect score or high score (>70%)
        perfect_score = (correct_count == total_questions)
        high_score = (correct_count >= total_questions * 0.7)  # 70% trở lên
        voucher = None
        
        # Award voucher for high score
        if high_score:
            base_value = 50000 if perfect_score else 30000  # 50k for perfect, 30k for high score
            
            # Generate voucher code
            voucher_code = f"QUIZ{str(uuid4())[:8].upper()}"
            
            # If user is logged in, add voucher to account with membership boost
            if 'user_id' in session and session.get('user_id') != 'admin':
                user_id = session.get('user_id')
                user = get_user_by_id(user_id)
                
                if user:
                    # Apply membership voucher boost
                    tier = user.get('membership_tier', 'bronze')
                    benefits = get_membership_benefits(tier)
                    voucher_value = int(base_value * benefits.get('voucher_boost', 1.0))
                    
                    # Add voucher to user account
                    add_voucher_to_user(user_id, voucher_code, voucher_value, 'quiz')
                    
                    # Update user statistics
                    quiz_completed = user.get('quiz_completed', 0) + 1
                    update_user(user_id, {'quiz_completed': quiz_completed})
                    
                    # Award points for completing quiz
                    add_points(user_id, 50, 'quiz_completed', 'Hoàn thành quiz kiến thức')
                    session['user_points'] = user.get('points', 0) + 50
                    
                    # Update quest progress
                    update_quest_progress(user_id, 'game', increment=1)
                    
                    # Check achievements
                    if perfect_score:
                        check_and_award_achievements(user_id, 'quiz', {'perfect': True})
                    
                    voucher = {
                        'code': voucher_code,
                        'value': voucher_value,
                        'message': f'Mã đã được lưu vào tài khoản! Giảm {voucher_value:,}đ',
                        'auto_added': True
                    }
            else:
                # User not logged in - just show voucher code to copy
                voucher_value = base_value
                voucher = {
                    'code': voucher_code,
                    'value': voucher_value,
                    'message': f'Đăng nhập và nhập mã để nhận giảm giá {voucher_value:,}đ!',
                    'auto_added': False
                }
        
        # Save score
        _save_quiz_score(player_name, score, correct_count, total_questions, time_taken)
        
        # Get updated high scores
        high_scores = _get_high_scores()
        
        return {
            'ok': True,
            'score': score,
            'correct_answers': correct_count,
            'total_questions': total_questions,
            'perfect_score': perfect_score,
            'high_score': high_score,
            'voucher': voucher,
            'high_scores': high_scores
        }
        
    except Exception as e:
        app.logger.exception('Error submitting quiz')
        return {'ok': False, 'error': str(e)}, 500

# ==================== Memory Game ====================

def _get_memory_high_scores():
    """Load memory game high scores from file"""
    try:
        scores_file = BASE_DIR / 'data' / 'memory_scores.jsonl'
        if not scores_file.exists():
            return []
        
        scores = []
        with open(scores_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    scores.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        # Sort by score desc, time asc
        scores.sort(key=lambda x: (-x.get('score', 0), x.get('time_taken', 999)))
        return scores[:10]  # Top 10
    except Exception:
        app.logger.exception('Error loading memory scores')
        return []

def _save_memory_score(player_name, score, moves, time_taken, difficulty, pairs):
    """Save memory game score to file"""
    try:
        scores_dir = BASE_DIR / 'data'
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_file = scores_dir / 'memory_scores.jsonl'
        
        entry = {
            'id': str(uuid4())[:8],
            'player_name': player_name,
            'score': score,
            'moves': moves,
            'time_taken': time_taken,
            'difficulty': difficulty,
            'pairs': pairs,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(scores_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        app.logger.info(f'Saved memory score: {player_name} - {score}')
        return True
    except Exception:
        app.logger.exception('Error saving memory score')
        return False

@app.route('/game/memory')
def memory_game():
    """Memory matching game"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    # Get random tomato images
    images_dir = Path(app.static_folder) / 'images' / 'tomato_samples'
    tomato_images = []
    
    if images_dir.exists():
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            tomato_images.extend([
                url_for('static', filename=f'images/tomato_samples/{img.name}')
                for img in images_dir.glob(ext)
            ])
    
    # Shuffle and limit to 20 images max
    random.shuffle(tomato_images)
    tomato_images = tomato_images[:20]
    
    return render_template('memory_game.html', tomato_images=tomato_images)

@app.route('/api/memory/submit', methods=['POST'])
def submit_memory():
    """Submit memory game score"""
    try:
        data = request.get_json()
        player_name = data.get('player_name', 'Anonymous').strip()[:50]
        score = data.get('score', 0)
        moves = data.get('moves', 0)
        time_taken = data.get('time_taken', 0)
        difficulty = data.get('difficulty', 'medium')
        pairs = data.get('pairs', 8)
        
        if not player_name:
            player_name = 'Anonymous'
        
        # Award voucher for high score (based on difficulty)
        voucher = None
        min_score_for_voucher = {
            'easy': 700,
            'medium': 900,
            'hard': 1100
        }
        
        if score >= min_score_for_voucher.get(difficulty, 900):
            # Base voucher value increases with difficulty
            base_values = {
                'easy': 15000,    # 15k for easy
                'medium': 25000,  # 25k for medium
                'hard': 40000     # 40k for hard
            }
            base_value = base_values.get(difficulty, 25000)
            
            # Generate voucher code
            voucher_code = f"MEM{str(uuid4())[:8].upper()}"
            
            # If user is logged in, add voucher to account with membership boost
            if 'user_id' in session and session.get('user_id') != 'admin':
                user_id = session.get('user_id')
                user = get_user_by_id(user_id)
                
                if user:
                    # Apply membership voucher boost
                    tier = user.get('membership_tier', 'bronze')
                    benefits = get_membership_benefits(tier)
                    voucher_value = int(base_value * benefits.get('voucher_boost', 1.0))
                    
                    # Add voucher to user account
                    add_voucher_to_user(user_id, voucher_code, voucher_value, 'memory_game')
                    
                    # Update user statistics
                    memory_completed = user.get('memory_completed', 0) + 1
                    update_user(user_id, {'memory_completed': memory_completed})
                    
                    # Award points for completing memory game
                    add_points(user_id, 30, 'memory_game', 'Hoàn thành trò chơi trí nhớ')
                    session['user_points'] = user.get('points', 0) + 30
                    
                    # Update quest progress
                    update_quest_progress(user_id, 'game', increment=1)
                    
                    # Check achievements (fast completion)
                    if time_taken < 20:
                        check_and_award_achievements(user_id, 'memory', {'fast': True, 'time': time_taken})
                    
                    voucher = {
                        'code': voucher_code,
                        'value': voucher_value,
                        'message': f'Mã đã được lưu vào tài khoản! Giảm {voucher_value:,}đ',
                        'auto_added': True
                    }
            else:
                # User not logged in - just show voucher code to copy
                voucher_value = base_value
                voucher = {
                    'code': voucher_code,
                    'value': voucher_value,
                    'message': f'Đăng nhập và nhập mã để nhận giảm giá {voucher_value:,}đ!',
                    'auto_added': False
                }
        
        # Save score
        _save_memory_score(player_name, score, moves, time_taken, difficulty, pairs)
        
        return {
            'ok': True,
            'voucher': voucher
        }
        
    except Exception as e:
        app.logger.exception('Error submitting memory game score')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/memory/scores', methods=['GET'])
def get_memory_scores():
    """Get memory game high scores"""
    try:
        scores = _get_memory_high_scores()
        return {'ok': True, 'scores': scores}
    except Exception as e:
        app.logger.exception('Error getting memory scores')
        return {'ok': False, 'error': str(e)}, 500

# ==================== FARM GAME ====================

# Farm game constants
FARM_DISEASES = [
    {'id': 'healthy', 'name': 'Khỏe mạnh', 'icon': '🌱', 'treatment': None, 'damage': 0},
    {'id': 'Tomato___Late_blight', 'name': 'Mốc sương', 'icon': '🍄', 'treatment': 'Thuốc trừ nấm', 'damage': 30},
    {'id': 'Tomato___Early_blight', 'name': 'Bệnh sớm', 'icon': '🦠', 'treatment': 'Thuốc trừ nấm', 'damage': 25},
    {'id': 'Tomato___Septoria_leaf_spot', 'name': 'Đốm lá Septoria', 'icon': '⚫', 'treatment': 'Thuốc trừ nấm', 'damage': 20},
    {'id': 'Tomato___Bacterial_spot', 'name': 'Đốm vi khuẩn', 'icon': '🔴', 'treatment': 'Thuốc kháng sinh', 'damage': 25},
    {'id': 'Tomato___Target_Spot', 'name': 'Đốm mục tiêu', 'icon': '🎯', 'treatment': 'Thuốc trừ nấm', 'damage': 20},
    {'id': 'Tomato___Leaf_Mold', 'name': 'Mốc lá', 'icon': '🍂', 'treatment': 'Thuốc trừ nấm', 'damage': 15},
    {'id': 'Tomato___Spider_mites', 'name': 'Nhện đỏ', 'icon': '🕷️', 'treatment': 'Thuốc trừ sâu', 'damage': 20},
    {'id': 'Tomato___Yellow_Leaf_Curl_Virus', 'name': 'Virus khảm lá', 'icon': '🦠', 'treatment': 'Thuốc trừ ruồi trắng', 'damage': 35},
    {'id': 'Tomato___Mosaic_virus', 'name': 'Virus khảm', 'icon': '🌈', 'treatment': 'Loại bỏ cây', 'damage': 40}
]

FARM_ITEMS = {
    'water': {'name': 'Nước', 'icon': '💧', 'price': 0, 'effect': 'Giữ cây sống'},
    'fertilizer': {'name': 'Phân bón', 'icon': '🌾', 'price': 50, 'effect': 'Tăng tốc độ lớn 20%'},
    'fungicide': {'name': 'Thuốc trừ nấm', 'icon': '💊', 'price': 100, 'effect': 'Chữa bệnh nấm'},
    'pesticide': {'name': 'Thuốc trừ sâu', 'icon': '🧪', 'effect': 'Chữa sâu/nhện', 'price': 100},
    'antibiotic': {'name': 'Thuốc kháng sinh', 'icon': '💉', 'price': 150, 'effect': 'Chữa vi khuẩn'},
    'premium_fertilizer': {'name': 'Phân cao cấp', 'icon': '✨', 'price': 200, 'effect': 'Tăng tốc 50%'}
}

def _get_farm_progress(user_id):
    """Load farm progress for a user"""
    try:
        progress_file = BASE_DIR / 'data' / 'farm_progress.jsonl'
        if not progress_file.exists():
            return None
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('user_id') == user_id:
                        return data
                except json.JSONDecodeError:
                    continue
        return None
    except Exception:
        app.logger.exception('Error loading farm progress')
        return None

def _save_farm_progress(user_id, progress_data):
    """Save farm progress for a user"""
    try:
        progress_file = BASE_DIR / 'data' / 'farm_progress.jsonl'
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        all_data = []
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('user_id') != user_id:
                            all_data.append(data)
                    except json.JSONDecodeError:
                        continue
        
        # Add new data
        progress_data['user_id'] = user_id
        progress_data['updated_at'] = datetime.utcnow().isoformat()
        all_data.append(progress_data)
        
        # Write all data
        with open(progress_file, 'w', encoding='utf-8') as f:
            for data in all_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        return True
    except Exception:
        app.logger.exception('Error saving farm progress')
        return False

@app.route('/game/farm')
def farm_game():
    """Tomato RPG game - control character to farm"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    if 'user_id' not in session:
        flash('Vui lòng đăng nhập để chơi game')
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    progress = _get_farm_progress(user_id)
    
    # Initialize new farm if no progress
    if not progress:
        progress = {
            'coins': 100,
            'plants': [None] * 6,  # 6 slots, None = empty
            'inventory': {
                'water': 10,
                'fertilizer': 5,
                'pesticide': 2,
                'medicine': 1
            },
            'total_harvested': 0,
            'total_earned': 0,
            'level': 1
        }
        _save_farm_progress(user_id, progress)
    
    # Ensure plants is a list with proper structure
    if not isinstance(progress['plants'], list):
        progress['plants'] = [None] * 6
    
    return render_template('farm_rpg.html', progress=progress)

@app.route('/api/farm/plant', methods=['POST'])
def farm_plant():
    """Plant a new tomato in a slot"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        slot = data.get('slot', 0)
        
        progress = _get_farm_progress(user_id)
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check if user has enough coins
        plant_cost = 20
        if progress['coins'] < plant_cost:
            return {'ok': False, 'error': 'Không đủ xu!'}, 400
        
        # Check if slot is valid and empty
        if slot < 0 or slot >= len(progress['plants']):
            return {'ok': False, 'error': 'Slot không hợp lệ'}, 400
        
        if progress['plants'][slot] is not None:
            return {'ok': False, 'error': 'Ô này đã có cây rồi!'}, 400
        
        # Create new plant
        plant = {
            'id': str(uuid4())[:8],
            'planted_at': datetime.utcnow().isoformat(),
            'stage': 'seed',
            'health': 100,
            'disease': None,
            'last_watered': datetime.utcnow().isoformat(),
            'last_fertilized': None
        }
        
        progress['plants'][slot] = plant
        progress['coins'] -= plant_cost
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'progress': progress}
        
    except Exception as e:
        app.logger.exception('Error planting')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/water', methods=['POST'])
def farm_water_new():
    """Water a plant in slot"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        slot = data.get('slot', 0)
        
        progress = _get_farm_progress(user_id)
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check water in inventory
        if progress['inventory'].get('water', 0) <= 0:
            return {'ok': False, 'error': 'Hết nước! Mua thêm ở cửa hàng.'}, 400
        
        # Check if slot is valid and has plant
        if slot < 0 or slot >= len(progress['plants']) or progress['plants'][slot] is None:
            return {'ok': False, 'error': 'Không có cây ở đây!'}, 400
        
        plant = progress['plants'][slot]
        
        # Water the plant - boost health slightly
        plant['health'] = min(100, plant['health'] + 5)
        plant['last_watered'] = datetime.utcnow().isoformat()
        progress['inventory']['water'] -= 1
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'progress': progress}
        
    except Exception as e:
        app.logger.exception('Error watering')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/fertilize', methods=['POST'])
def farm_fertilize_new():
    """Fertilize a plant in slot"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        slot = data.get('slot', 0)
        
        progress = _get_farm_progress(user_id)
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check fertilizer in inventory
        if progress['inventory'].get('fertilizer', 0) <= 0:
            return {'ok': False, 'error': 'Hết phân bón!'}, 400
        
        # Check if slot is valid and has plant
        if slot < 0 or slot >= len(progress['plants']) or progress['plants'][slot] is None:
            return {'ok': False, 'error': 'Không có cây ở đây!'}, 400
        
        plant = progress['plants'][slot]
        
        # Fertilize - speed up growth
        plant['health'] = min(100, plant['health'] + 10)
        plant['last_fertilized'] = datetime.utcnow().isoformat()
        progress['inventory']['fertilizer'] -= 1
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'progress': progress}
        
    except Exception as e:
        app.logger.exception('Error fertilizing')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/treat', methods=['POST'])
def farm_treat_new():
    """Treat a diseased plant"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        slot = data.get('slot', 0)
        medicine = data.get('medicine', 'medicine')
        
        progress = _get_farm_progress(user_id)
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check if slot is valid and has plant
        if slot < 0 or slot >= len(progress['plants']) or progress['plants'][slot] is None:
            return {'ok': False, 'error': 'Không có cây ở đây!'}, 400
        
        plant = progress['plants'][slot]
        
        if not plant.get('disease'):
            return {'ok': False, 'error': 'Cây không bị bệnh!'}, 400
        
        # Check medicine in inventory
        if progress['inventory'].get(medicine, 0) <= 0:
            return {'ok': False, 'error': f'Hết {medicine}!'}, 400
        
        # Cure the disease
        plant['disease'] = None
        plant['health'] = min(100, plant['health'] + 20)
        progress['inventory'][medicine] -= 1
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'progress': progress}
        
    except Exception as e:
        app.logger.exception('Error treating')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/harvest', methods=['POST'])
def farm_harvest_new():
    """Harvest a mature plant"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        slot = data.get('slot', 0)
        
        progress = _get_farm_progress(user_id)
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check if slot is valid and has plant
        if slot < 0 or slot >= len(progress['plants']) or progress['plants'][slot] is None:
            return {'ok': False, 'error': 'Không có cây ở đây!'}, 400
        
        plant = progress['plants'][slot]
        
        if plant['stage'] != 'fruiting':
            return {'ok': False, 'error': 'Cây chưa ra quả!'}, 400
        
        # Calculate reward based on health
        base_reward = 50
        health_bonus = int(base_reward * (plant['health'] / 100))
        total_reward = base_reward + health_bonus
        
        # Give reward
        progress['coins'] += total_reward
        progress['total_harvested'] += 1
        progress['total_earned'] += total_reward
        
        # Check level up
        harvests_for_next_level = progress['level'] * 10
        if progress['total_harvested'] >= harvests_for_next_level:
            progress['level'] += 1
        
        # Remove plant from slot
        progress['plants'][slot] = None
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'progress': progress, 'reward': total_reward}
        
    except Exception as e:
        app.logger.exception('Error harvesting')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/buy', methods=['POST'])
def farm_buy_new():
    """Buy an item from shop"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        item = data.get('item')
        
        progress = _get_farm_progress(user_id)
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Item prices
        prices = {
            'seed': 20,
            'water': 5,
            'fertilizer': 15,
            'pesticide': 20,
            'medicine': 30
        }
        
        if item not in prices:
            return {'ok': False, 'error': 'Item không tồn tại!'}, 400
        
        price = prices[item]
        
        # Check if user has enough coins
        if progress['coins'] < price:
            return {'ok': False, 'error': 'Không đủ xu!'}, 400
        
        # Buy item (seed is special - doesn't go to inventory)
        if item != 'seed':
            progress['inventory'][item] = progress['inventory'].get(item, 0) + 1
        
        progress['coins'] -= price
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'progress': progress}
        
    except Exception as e:
        app.logger.exception('Error buying')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/water/<plant_id>', methods=['POST'])
def farm_water(plant_id):
    """Water a plant"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        progress = _get_farm_progress(user_id)
        
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check water in inventory
        if progress['inventory'].get('water', 0) <= 0:
            return {'ok': False, 'error': 'Hết nước! Mua thêm trong cửa hàng.'}, 400
        
        # Find plant
        plant = next((p for p in progress['plants'] if p['id'] == plant_id), None)
        if not plant:
            return {'ok': False, 'error': 'Plant not found'}, 404
        
        # Water the plant
        plant['water_level'] = min(100, plant['water_level'] + 50)
        progress['inventory']['water'] -= 1
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'plant': plant, 'inventory': progress['inventory']}
        
    except Exception as e:
        app.logger.exception('Error watering')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/fertilize/<plant_id>', methods=['POST'])
def farm_fertilize(plant_id):
    """Fertilize a plant"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        data = request.get_json() or {}
        fertilizer_type = data.get('type', 'fertilizer')
        
        user_id = session.get('user_id')
        progress = _get_farm_progress(user_id)
        
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check fertilizer in inventory
        if progress['inventory'].get(fertilizer_type, 0) <= 0:
            return {'ok': False, 'error': f'Hết {FARM_ITEMS[fertilizer_type]["name"]}!'}, 400
        
        # Find plant
        plant = next((p for p in progress['plants'] if p['id'] == plant_id), None)
        if not plant:
            return {'ok': False, 'error': 'Plant not found'}, 404
        
        # Apply fertilizer
        plant['fertilized'] = fertilizer_type
        progress['inventory'][fertilizer_type] -= 1
        
        _save_farm_progress(user_id, progress)
        
        return {'ok': True, 'plant': plant, 'inventory': progress['inventory']}
        
    except Exception as e:
        app.logger.exception('Error fertilizing')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/treat/<plant_id>', methods=['POST'])
def farm_treat(plant_id):
    """Treat a diseased plant"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        data = request.get_json() or {}
        treatment = data.get('treatment')
        
        user_id = session.get('user_id')
        progress = _get_farm_progress(user_id)
        
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Find plant
        plant = next((p for p in progress['plants'] if p['id'] == plant_id), None)
        if not plant or not plant.get('disease'):
            return {'ok': False, 'error': 'Cây không bệnh!'}, 400
        
        # Get disease info
        disease = next((d for d in FARM_DISEASES if d['id'] == plant['disease']), None)
        if not disease:
            return {'ok': False, 'error': 'Unknown disease'}, 400
        
        # Check if treatment is correct
        treatment_map = {
            'Thuốc trừ nấm': 'fungicide',
            'Thuốc trừ sâu': 'pesticide',
            'Thuốc kháng sinh': 'antibiotic',
            'Loại bỏ cây': 'remove'
        }
        
        required_item = treatment_map.get(disease['treatment'])
        
        if treatment == 'remove':
            # Remove the plant
            progress['plants'] = [p for p in progress['plants'] if p['id'] != plant_id]
            _save_farm_progress(user_id, progress)
            return {'ok': True, 'removed': True, 'message': 'Đã loại bỏ cây bệnh'}
        
        # Check if user has the right treatment
        if not required_item or progress['inventory'].get(required_item, 0) <= 0:
            return {'ok': False, 'error': f'Cần {disease["treatment"]} để chữa bệnh này!'}, 400
        
        # Apply treatment
        if treatment == required_item:
            plant['disease'] = None
            plant['health'] = min(100, plant['health'] + 20)
            progress['inventory'][required_item] -= 1
            
            _save_farm_progress(user_id, progress)
            
            return {'ok': True, 'plant': plant, 'inventory': progress['inventory']}
        else:
            return {'ok': False, 'error': 'Sai loại thuốc!'}, 400
        
    except Exception as e:
        app.logger.exception('Error treating')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/harvest/<plant_id>', methods=['POST'])
def farm_harvest(plant_id):
    """Harvest a mature plant"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        progress = _get_farm_progress(user_id)
        
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Find plant
        plant = next((p for p in progress['plants'] if p['id'] == plant_id), None)
        if not plant:
            return {'ok': False, 'error': 'Plant not found'}, 404
        
        # Check if plant is ready to harvest
        if plant['stage'] != 'fruiting' or plant.get('disease'):
            return {'ok': False, 'error': 'Cây chưa sẵn sàng thu hoạch!'}, 400
        
        # Calculate yield based on health
        base_yield = 50
        health_multiplier = plant['health'] / 100
        yield_coins = int(base_yield * health_multiplier)
        
        # Bonus for perfect health
        if plant['health'] >= 95:
            yield_coins = int(yield_coins * 1.5)
        
        # Remove plant and add coins
        progress['plants'] = [p for p in progress['plants'] if p['id'] != plant_id]
        progress['coins'] += yield_coins
        progress['total_harvested'] = progress.get('total_harvested', 0) + 1
        progress['total_earned'] = progress.get('total_earned', 0) + yield_coins
        
        # Level up check
        harvests_needed = progress.get('level', 1) * 5
        if progress['total_harvested'] >= harvests_needed:
            progress['level'] = progress.get('level', 1) + 1
            leveled_up = True
        else:
            leveled_up = False
        
        _save_farm_progress(user_id, progress)
        
        return {
            'ok': True,
            'yield': yield_coins,
            'coins': progress['coins'],
            'total_harvested': progress['total_harvested'],
            'leveled_up': leveled_up,
            'level': progress.get('level', 1)
        }
        
    except Exception as e:
        app.logger.exception('Error harvesting')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/buy', methods=['POST'])
def farm_buy():
    """Buy items from shop"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        data = request.get_json() or {}
        item = data.get('item')
        quantity = data.get('quantity', 1)
        
        user_id = session.get('user_id')
        progress = _get_farm_progress(user_id)
        
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        # Check if item exists
        if item not in FARM_ITEMS:
            return {'ok': False, 'error': 'Item not found'}, 404
        
        # Calculate cost
        item_data = FARM_ITEMS[item]
        total_cost = item_data['price'] * quantity
        
        # Check if user has enough coins
        if progress['coins'] < total_cost:
            return {'ok': False, 'error': 'Không đủ xu!'}, 400
        
        # Add item to inventory
        progress['inventory'][item] = progress['inventory'].get(item, 0) + quantity
        progress['coins'] -= total_cost
        
        _save_farm_progress(user_id, progress)
        
        return {
            'ok': True,
            'inventory': progress['inventory'],
            'coins': progress['coins']
        }
        
    except Exception as e:
        app.logger.exception('Error buying item')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/farm/update', methods=['POST'])
def farm_update():
    """Update farm state (called periodically by client)"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    try:
        user_id = session.get('user_id')
        progress = _get_farm_progress(user_id)
        
        if not progress:
            return {'ok': False, 'error': 'No farm data'}, 400
        
        current_time = datetime.utcnow()
        
        # Update each plant in slots
        for i, plant in enumerate(progress['plants']):
            if plant is None:  # Skip empty slots
                continue
                
            planted_time = datetime.fromisoformat(plant['planted_at'])
            age_seconds = (current_time - planted_time).total_seconds()
            
            # Growth stages (in seconds)
            stage_durations = {
                'seed': 30,      # 30s
                'sprout': 60,    # 1 min
                'growing': 120,  # 2 min
                'mature': 120,   # 2 min
                'fruiting': 0    # ready to harvest
            }
            
            # Apply fertilizer speed boost
            if plant.get('last_fertilized'):
                fertilized_time = datetime.fromisoformat(plant['last_fertilized'])
                if (current_time - fertilized_time).total_seconds() < 300:  # Boost for 5 minutes
                    age_seconds *= 1.3
            
            # Determine stage
            total = 0
            for stage, duration in stage_durations.items():
                total += duration
                if age_seconds < total:
                    plant['stage'] = stage
                    break
            else:
                plant['stage'] = 'fruiting'
            
            # Health decreases slowly over time if not watered
            if plant.get('last_watered'):
                watered_time = datetime.fromisoformat(plant['last_watered'])
                minutes_since_water = (current_time - watered_time).total_seconds() / 60
                if minutes_since_water > 5:  # Start losing health after 5 min without water
                    plant['health'] = max(0, plant['health'] - int((minutes_since_water - 5) * 0.5))
            
            # Random disease (3% chance if health < 70)
            if plant['stage'] in ['growing', 'mature', 'fruiting'] and plant['health'] < 70 and not plant.get('disease'):
                if random.random() < 0.03:  # 3% chance
                    diseases = ['leaf_spot', 'pests', 'thirsty']
                    plant['disease'] = random.choice(diseases)
            
            # Disease damage
            if plant.get('disease'):
                disease_damage = {'leaf_spot': 0.5, 'pests': 0.8, 'thirsty': 0.3}
                damage = disease_damage.get(plant['disease'], 0.5)
                plant['health'] = max(0, plant['health'] - damage)
            
            # Plant dies if health reaches 0
            if plant['health'] <= 0:
                progress['plants'][i] = None  # Remove dead plant
        
        _save_farm_progress(user_id, progress)
        
        return {
            'ok': True,
            'progress': progress
        }
        
    except Exception as e:
        app.logger.exception('Error updating farm')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/redeem_voucher', methods=['POST'])
def redeem_voucher():
    """Redeem a voucher code and add it to user account"""
    app.logger.info('Redeem voucher request received')
    
    try:
        # Check if user is logged in
        if 'user_id' not in session:
            app.logger.warning('Redeem voucher: User not logged in')
            return {'ok': False, 'error': 'Vui lòng đăng nhập để nhập mã giảm giá'}, 401
        
        user_id = session.get('user_id')
        app.logger.info(f'Redeem voucher: User ID = {user_id}')
        
        if user_id == 'admin':
            app.logger.warning('Redeem voucher: Admin tried to redeem')
            return {'ok': False, 'error': 'Admin không thể sử dụng mã giảm giá'}, 403
        
        data = request.get_json()
        voucher_code = data.get('code', '').strip().upper()
        app.logger.info(f'Redeem voucher: Code = {voucher_code}')
        
        if not voucher_code:
            app.logger.warning('Redeem voucher: Empty code')
            return {'ok': False, 'error': 'Vui lòng nhập mã giảm giá'}, 400
        
        # Validate voucher code format (must start with MEM or QUIZ and be at least 8 characters)
        if not (voucher_code.startswith('MEM') or voucher_code.startswith('QUIZ')):
            app.logger.warning(f'Redeem voucher: Invalid prefix - {voucher_code}')
            return {'ok': False, 'error': 'Mã giảm giá không hợp lệ. Mã phải bắt đầu bằng MEM hoặc QUIZ (từ trò chơi)'}, 400
        
        if len(voucher_code) < 8:
            app.logger.warning(f'Redeem voucher: Too short - {voucher_code}')
            return {'ok': False, 'error': 'Mã giảm giá không hợp lệ. Mã phải có ít nhất 8 ký tự'}, 400
        
        user = get_user_by_id(user_id)
        if not user:
            return {'ok': False, 'error': 'Không tìm thấy thông tin người dùng'}, 404
        
        # Check if voucher already exists in user's account
        existing_vouchers = user.get('vouchers', [])
        for v in existing_vouchers:
            if v.get('code') == voucher_code:
                app.logger.warning(f'Redeem voucher: Already exists - {voucher_code}')
                return {'ok': False, 'error': 'Mã giảm giá này đã có trong tài khoản của bạn rồi!'}, 400
        
        # Determine voucher value based on code prefix and difficulty
        # MEM codes from memory game
        if voucher_code.startswith('MEM'):
            # Try to determine value from code pattern (default to medium)
            voucher_value = 25000
            source = 'memory_game'
        elif voucher_code.startswith('QUIZ'):
            voucher_value = 30000
            source = 'quiz'
        else:
            voucher_value = 20000
            source = 'game'
        
        # Apply membership boost
        tier = user.get('membership_tier', 'bronze')
        benefits = get_membership_benefits(tier)
        voucher_value = int(voucher_value * benefits.get('voucher_boost', 1.0))
        
        # Add voucher to user
        success = add_voucher_to_user(user_id, voucher_code, voucher_value, source)
        
        if success:
            app.logger.info(f'User {user_id} redeemed voucher: {voucher_code} ({voucher_value:,}đ)')
            return {
                'ok': True,
                'message': f'Đã thêm mã giảm giá {voucher_value:,}đ vào tài khoản!',
                'voucher': {
                    'code': voucher_code,
                    'value': voucher_value
                }
            }
        else:
            return {'ok': False, 'error': 'Không thể thêm mã giảm giá. Vui lòng thử lại'}, 500
            
    except Exception as e:
        app.logger.exception('Error redeeming voucher')
        return {'ok': False, 'error': 'Lỗi hệ thống. Vui lòng thử lại sau'}, 500

# ==================== Shop ====================

SUBSCRIPTION_BOX_PLANS = [
    {
        'id': 'starter',
        'name': 'Gói Khởi Đầu',
        'price_per_delivery': 99000,
        'description': 'Phù hợp người mới trồng, bộ sản phẩm cơ bản cho 1-2 luống cà chua.',
        'items': [
            'Phân bón hữu cơ mini',
            'Dung dịch phòng nấm lá',
            'Combo bẫy côn trùng',
            'Hướng dẫn chăm sóc theo tuần'
        ]
    },
    {
        'id': 'pro',
        'name': 'Gói Chuyên Sâu',
        'price_per_delivery': 179000,
        'description': 'Bổ sung dinh dưỡng + phòng trị chuyên sâu cho vườn cà chua quy mô vừa.',
        'items': [
            'Phân bón NPK cân bằng',
            'Chế phẩm vi sinh đất',
            'Thuốc trừ nấm phổ rộng',
            'Bộ test pH đất',
            'Checklist theo dõi sâu bệnh'
        ]
    },
    {
        'id': 'premium',
        'name': 'Gói Premium Farm',
        'price_per_delivery': 299000,
        'description': 'Giải pháp đầy đủ cho nhà vườn cần chăm sóc bài bản và tối ưu năng suất.',
        'items': [
            'Phân bón cao cấp đa vi lượng',
            'Bộ phòng bệnh tổng hợp 4 tuần',
            'Thiết bị đo độ ẩm mini',
            'Combo xử lý nhện đỏ',
            'Lịch chăm sóc cá nhân hóa'
        ]
    }
]

SUBSCRIPTION_FREQUENCIES = {
    'weekly': {'label': 'Hàng tuần', 'monthly_deliveries': 4},
    'biweekly': {'label': '2 tuần/lần', 'monthly_deliveries': 2},
    'monthly': {'label': 'Hàng tháng', 'monthly_deliveries': 1},
}


def get_subscription_box_plans():
    return SUBSCRIPTION_BOX_PLANS


def get_subscription_box_file():
    return BASE_DIR / 'data' / 'subscription_boxes.jsonl'


def load_subscription_boxes():
    """Load all subscription boxes from JSONL."""
    try:
        file_path = get_subscription_box_file()
        if not file_path.exists():
            return []

        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    except Exception as e:
        app.logger.error(f'Error loading subscription boxes: {e}')
        return []


def append_subscription_box(subscription):
    """Append a new subscription record."""
    try:
        file_path = get_subscription_box_file()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(subscription, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        app.logger.error(f'Error appending subscription box: {e}')
        return False


def get_user_active_subscription(user_id):
    """Return the newest active subscription for user."""
    subscriptions = load_subscription_boxes()
    active = [
        s for s in subscriptions
        if s.get('user_id') == user_id and s.get('status') == 'active'
    ]
    active.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return active[0] if active else None


def cancel_subscription_box(subscription_id, user_id):
    """Cancel a subscription if it belongs to user and is active."""
    try:
        file_path = get_subscription_box_file()
        if not file_path.exists():
            return False, 'Không tìm thấy dữ liệu gói đăng ký'

        updated_rows = []
        cancelled_item = None

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if row.get('id') == subscription_id and row.get('user_id') == user_id:
                    if row.get('status') != 'active':
                        return False, 'Gói này đã được hủy trước đó'
                    row['status'] = 'cancelled'
                    row['cancelled_at'] = datetime.utcnow().isoformat()
                    row['updated_at'] = datetime.utcnow().isoformat()
                    cancelled_item = row

                updated_rows.append(row)

        if not cancelled_item:
            return False, 'Không tìm thấy gói cần hủy'

        with open(file_path, 'w', encoding='utf-8') as f:
            for row in updated_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        return True, cancelled_item
    except Exception as e:
        app.logger.error(f'Error canceling subscription box: {e}')
        return False, 'Lỗi hệ thống khi hủy gói'


@app.route('/subscription-box')
def subscription_box_page():
    """Subscription Box landing page."""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    plans = get_subscription_box_plans()
    active_subscription = None

    user_id = session.get('user_id')
    if user_id:
        active_subscription = get_user_active_subscription(user_id)

    return render_template(
        'subscription_box.html',
        plans=plans,
        frequencies=SUBSCRIPTION_FREQUENCIES,
        active_subscription=active_subscription
    )


@app.route('/api/subscription-box/my', methods=['GET'])
def api_my_subscription_box():
    """Return current user's active subscription box."""
    user_id = session.get('user_id')
    if not user_id:
        return {'ok': False, 'error': 'Vui lòng đăng nhập'}, 401
    if session.get('is_admin'):
        return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403

    subscription = get_user_active_subscription(user_id)
    return {'ok': True, 'subscription': subscription}


@app.route('/api/subscription-box/subscribe', methods=['POST'])
def api_subscribe_box():
    """Create a new subscription box for current user."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return {'ok': False, 'error': 'Vui lòng đăng nhập để đăng ký gói'}, 401
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403

        current_active = get_user_active_subscription(user_id)
        if current_active:
            return {
                'ok': False,
                'error': 'Bạn đã có gói đang hoạt động. Vui lòng hủy gói hiện tại trước.'
            }, 400

        data = request.get_json() or {}
        plan_id = (data.get('plan_id') or '').strip()
        frequency = (data.get('frequency') or 'biweekly').strip()
        address = (data.get('address') or '').strip()[:500]
        note = (data.get('note') or '').strip()[:500]
        start_date_raw = (data.get('start_date') or '').strip()

        if not plan_id:
            return {'ok': False, 'error': 'Vui lòng chọn gói đăng ký'}, 400
        if frequency not in SUBSCRIPTION_FREQUENCIES:
            return {'ok': False, 'error': 'Tần suất giao hàng không hợp lệ'}, 400
        if not address:
            return {'ok': False, 'error': 'Vui lòng nhập địa chỉ nhận hàng'}, 400

        plan = next((p for p in SUBSCRIPTION_BOX_PLANS if p['id'] == plan_id), None)
        if not plan:
            return {'ok': False, 'error': 'Gói đăng ký không tồn tại'}, 404

        today = datetime.utcnow().date()
        try:
            start_date = datetime.fromisoformat(start_date_raw).date() if start_date_raw else today
        except ValueError:
            start_date = today
        if start_date < today:
            start_date = today

        freq_cfg = SUBSCRIPTION_FREQUENCIES[frequency]
        price_per_delivery = int(plan['price_per_delivery'])
        monthly_estimate = price_per_delivery * int(freq_cfg['monthly_deliveries'])

        subscription = {
            'id': str(uuid4())[:8],
            'user_id': user_id,
            'user_email': session.get('user_email', ''),
            'plan_id': plan['id'],
            'plan_name': plan['name'],
            'plan_items': plan['items'],
            'delivery_frequency': frequency,
            'delivery_label': freq_cfg['label'],
            'price_per_delivery': price_per_delivery,
            'monthly_estimate': monthly_estimate,
            'next_delivery_date': start_date.isoformat(),
            'address': address,
            'note': note,
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
        }

        if not append_subscription_box(subscription):
            return {'ok': False, 'error': 'Không thể lưu đăng ký. Vui lòng thử lại.'}, 500

        try:
            create_notification(
                user_id=user_id,
                title='📦 Đăng ký Subscription Box thành công',
                message=(
                    f'Bạn đã đăng ký {subscription["plan_name"]} - {subscription["delivery_label"]}. '
                    f'Kỳ giao đầu tiên: {subscription["next_delivery_date"]}'
                ),
                notification_type='success',
                link='/subscription-box'
            )
        except Exception as notify_error:
            app.logger.warning(f'Cannot create subscription notification: {notify_error}')

        app.logger.info(f'Subscription created: {subscription["id"]} for user {user_id}')
        return {'ok': True, 'message': 'Đăng ký gói thành công', 'subscription': subscription}

    except Exception as e:
        app.logger.exception('Error subscribing subscription box')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/subscription-box/cancel/<subscription_id>', methods=['POST'])
def api_cancel_subscription_box(subscription_id):
    """Cancel current user's subscription box."""
    user_id = session.get('user_id')
    if not user_id:
        return {'ok': False, 'error': 'Vui lòng đăng nhập'}, 401
    if session.get('is_admin'):
        return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403

    ok, result = cancel_subscription_box(subscription_id, user_id)
    if not ok:
        return {'ok': False, 'error': result}, 400

    try:
        create_notification(
            user_id=user_id,
            title='🛑 Đã hủy Subscription Box',
            message=f'Bạn đã hủy gói {result.get("plan_name", "Subscription Box")}',
            notification_type='info',
            link='/subscription-box'
        )
    except Exception as notify_error:
        app.logger.warning(f'Cannot create cancel-subscription notification: {notify_error}')

    return {'ok': True, 'message': 'Đã hủy gói đăng ký', 'subscription': result}


@app.route('/shop')
def shop():
    """Shop page for pesticides and fertilizers"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    try:
        products = load_products()
        # Only show active products
        products = [p for p in products if p.get('status', 'active') == 'active']
        return render_template('shop.html', products=products)
    except Exception as e:
        app.logger.error(f'Error loading shop: {e}')
        return render_template('shop.html', products=[])


BLIND_BOX_PRICE = 99000
BLIND_BOX_REWARD_POOL = [
    # Harder pool: mostly low vouchers, lower cash odds, includes no-reward chance.
    {'type': 'voucher', 'value': 5000, 'probability': 35.0, 'label': 'Voucher 5,000đ'},
    {'type': 'voucher', 'value': 10000, 'probability': 25.0, 'label': 'Voucher 10,000đ'},
    {'type': 'voucher', 'value': 20000, 'probability': 15.0, 'label': 'Voucher 20,000đ'},
    {'type': 'voucher', 'value': 30000, 'probability': 7.0, 'label': 'Voucher 30,000đ'},
    {'type': 'voucher', 'value': 50000, 'probability': 3.0, 'label': 'Voucher 50,000đ'},
    {'type': 'cash', 'value': 100000, 'probability': 7.0, 'label': '100,000đ tiền mặt ví'},
    {'type': 'cash', 'value': 200000, 'probability': 3.0, 'label': '200,000đ tiền mặt ví'},
    {'type': 'cash', 'value': 500000, 'probability': 1.5, 'label': '500,000đ tiền mặt ví'},
    {'type': 'cash', 'value': 1000000, 'probability': 0.3, 'label': '1,000,000đ tiền mặt ví'},
    {'type': 'cash', 'value': 2000000, 'probability': 0.1, 'label': '2,000,000đ tiền mặt ví'},
    {'type': 'cash', 'value': 5000000, 'probability': 0.05, 'label': '5,000,000đ tiền mặt ví'},
    {'type': 'none', 'value': 0, 'probability': 3.0, 'label': 'Chúc bạn may mắn lần sau'},
    # Special jackpot kept at 0.05%
    {'type': 'cash', 'value': 10000000, 'probability': 0.05, 'label': '🎉 JACKPOT 10,000,000đ'},
]


def _pick_blind_box_reward():
    """Pick a blind box reward based on configured weighted probabilities."""
    roll = random.random() * 100
    cumulative = 0.0
    for reward in BLIND_BOX_REWARD_POOL:
        cumulative += reward['probability']
        if roll <= cumulative:
            return reward
    return BLIND_BOX_REWARD_POOL[-1]


def _has_successful_topup(user_id):
    """Require at least one successful top-up before allowing blind box opening."""
    transactions = load_wallet_transactions(user_id)
    for tx in transactions:
        if tx.get('transaction_type') == 'topup' and float(tx.get('amount', 0)) > 0:
            return True
    return False


def _append_blind_box_opening(record):
    """Store blind box opening history for auditing and analytics."""
    try:
        file_path = BASE_DIR / 'data' / 'blind_box_openings.jsonl'
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        app.logger.exception('Error saving blind box opening record')


@app.route('/api/blind-box/open', methods=['POST'])
def api_open_blind_box():
    """Open blind box: deduct 99k from wallet and grant weighted reward."""
    try:
        if 'user_id' not in session:
            return {'ok': False, 'error': 'Vui lòng đăng nhập'}, 401

        user_id = session.get('user_id')
        if not user_id or user_id == 'admin':
            return {'ok': False, 'error': 'Tài khoản không hợp lệ để mở hộp mù'}, 403

        if not _has_successful_topup(user_id):
            return {
                'ok': False,
                'error': 'Bạn cần nạp tiền vào ví ít nhất 1 lần trước khi mở hộp mù'
            }, 400

        wallet_before = get_user_wallet(user_id)
        balance_before = float(wallet_before.get('balance', 0))

        if balance_before < BLIND_BOX_PRICE:
            return {
                'ok': False,
                'error': f'Số dư ví không đủ. Cần tối thiểu {BLIND_BOX_PRICE:,}đ',
                'wallet_balance': balance_before,
                'required': BLIND_BOX_PRICE
            }, 400

        debit_tx = add_wallet_transaction(
            user_id=user_id,
            amount=-BLIND_BOX_PRICE,
            transaction_type='blind_box_open',
            description=f'Mở hộp mù ({BLIND_BOX_PRICE:,}đ)'
        )

        reward = _pick_blind_box_reward()
        reward_payload = {
            'type': reward['type'],
            'value': reward['value'],
            'label': reward['label'],
            'probability': reward['probability']
        }

        credit_tx_id = None
        voucher_code = None

        if reward['type'] == 'voucher':
            voucher_code = f"BBX{str(uuid4())[:8].upper()}"
            added = add_voucher_to_user(user_id, voucher_code, int(reward['value']), 'blind_box')
            if not added:
                add_wallet_transaction(
                    user_id=user_id,
                    amount=BLIND_BOX_PRICE,
                    transaction_type='blind_box_refund',
                    description='Hoàn tiền do lỗi tạo voucher hộp mù'
                )
                return {'ok': False, 'error': 'Không thể tạo voucher, hệ thống đã hoàn tiền'}, 500
            reward_payload['voucher_code'] = voucher_code
        elif reward['type'] == 'cash' and float(reward.get('value', 0)) > 0:
            credit_tx = add_wallet_transaction(
                user_id=user_id,
                amount=float(reward['value']),
                transaction_type='blind_box_reward_cash',
                description=f"Thưởng hộp mù: {reward['label']}",
                reference_id=debit_tx.get('id')
            )
            credit_tx_id = credit_tx.get('id')

        wallet_after = get_user_wallet(user_id)
        balance_after = float(wallet_after.get('balance', 0))

        _append_blind_box_opening({
            'id': str(uuid4()),
            'user_id': user_id,
            'price': BLIND_BOX_PRICE,
            'reward': reward_payload,
            'debit_transaction_id': debit_tx.get('id'),
            'credit_transaction_id': credit_tx_id,
            'balance_before': balance_before,
            'balance_after': balance_after,
            'opened_at': datetime.utcnow().isoformat()
        })

        try:
            message = f'Bạn nhận được {reward_payload["label"]}!'
            if voucher_code:
                message += f' Mã voucher: {voucher_code}'
            create_notification(
                user_id=user_id,
                title='🎁 Mở hộp mù thành công',
                message=message,
                notification_type='success',
                link='/shop'
            )
        except Exception as notify_error:
            app.logger.warning(f'Cannot create blind box notification: {notify_error}')

        return {
            'ok': True,
            'message': 'Mở hộp mù thành công!',
            'price': BLIND_BOX_PRICE,
            'reward': reward_payload,
            'wallet_balance': balance_after,
            'net_change': balance_after - balance_before
        }

    except Exception:
        app.logger.exception('Error opening blind box')
        return {'ok': False, 'error': 'Có lỗi xảy ra khi mở hộp mù'}, 500

@app.route('/api/products')
def api_products():
    """API endpoint to get products"""
    try:
        products = load_products()
        status_filter = request.args.get('status', 'active')
        
        # Filter by status
        if status_filter:
            products = [p for p in products if p.get('status', 'active') == status_filter]
        
        return jsonify({'ok': True, 'products': products})
    except Exception as e:
        app.logger.error(f'Error loading products API: {e}')
        return jsonify({'ok': False, 'error': str(e)})

# ==================== PRODUCT ROUTES ====================

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    """Product detail page"""
    product = get_product_by_id(product_id)
    if not product:
        flash('Sản phẩm không tồn tại', 'error')
        return redirect(url_for('shop'))
    
    # Only show active products to non-admin users
    if product.get('status') != 'active' and not session.get('is_admin'):
        flash('Sản phẩm không tồn tại', 'error')
        return redirect(url_for('shop'))
    
    return render_template('product_detail.html', product=product)

@app.route('/cart')
def cart():
    """Shopping cart page"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    user = None
    addresses = []
    
    if 'user_id' in session:
        user = get_user_by_email(session.get('user_email'))
        addresses = get_user_addresses(session.get('user_id'))
    
    return render_template('cart.html', user=user, addresses=addresses)

@app.route('/api/order/submit', methods=['POST'])
def submit_order():
    """Submit order"""
    try:
        data = request.get_json()
        
        # Validate stock availability before creating order
        items = data.get('items', [])
        stock_errors = []
        
        for item in items:
            product = get_product_by_id(item['id'])
            if not product:
                stock_errors.append(f"{item['name']}: Sản phẩm không tồn tại")
                continue
            
            current_stock = product.get('stock', 0)
            if current_stock < item['quantity']:
                stock_errors.append(f"{item['name']}: Chỉ còn {current_stock} sản phẩm")
        
        if stock_errors:
            return {
                'ok': False, 
                'error': 'Một số sản phẩm không đủ hàng',
                'details': stock_errors
            }, 400
        
        order = {
            'id': str(uuid4())[:8],
            'user_id': session.get('user_id', ''),
            'customer_name': data.get('customer_name', '').strip()[:100],
            'customer_phone': data.get('customer_phone', '').strip()[:20],
            'customer_address': data.get('customer_address', '').strip()[:500],
            'order_note': data.get('order_note', '').strip()[:500],
            'items': items,
            'subtotal': data.get('subtotal', 0),
            'shipping': data.get('shipping', 0),
            'discount': data.get('discount', 0),
            'coupon_code': data.get('coupon_code', ''),
            'total': data.get('total', 0),
            'payment_method': data.get('payment_method', 'cod'),
            'status': 'pending',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Decrement stock for each item
        for item in items:
            update_product_stock(item['id'], -item['quantity'])
            
            # Check for low stock and alert admin
            product = get_product_by_id(item['id'])
            if product and product.get('stock', 0) < 10:
                send_low_stock_alert(product)
        
        # Save order
        orders_dir = BASE_DIR / 'data'
        orders_dir.mkdir(parents=True, exist_ok=True)
        orders_file = orders_dir / 'orders.jsonl'
        
        with open(orders_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(order, ensure_ascii=False) + '\n')
        
        app.logger.info(f'Order submitted: {order["id"]} - {order["customer_name"]}')
        
        # Send order confirmation email
        user_email = session.get('user_email') if 'user_id' in session else None
        if user_email:
            try:
                send_order_confirmation_email(user_email, order)
                app.logger.info(f'Order confirmation email sent to {user_email}')
            except Exception as e:
                app.logger.error(f'Failed to send order confirmation email: {e}')
        
        # Create in-app notification
        if 'user_id' in session and session.get('user_id') != 'admin':
            try:
                create_notification(
                    user_id=session['user_id'],
                    title='📦 Đơn hàng mới',
                    message=f'Đơn hàng #{order["id"]} đã được tạo. Tổng tiền: {order["total"]:,.0f} đ',
                    notification_type='info',
                    link='/profile'
                )
            except Exception as e:
                app.logger.error(f'Failed to create order notification: {e}')
        
        # Update user statistics if logged in
        if 'user_id' in session and session.get('user_id') != 'admin':
            user_id = session.get('user_id')
            user = get_user_by_id(user_id)
            
            if user:
                # Update spending and order count
                total_spent = user.get('total_spent', 0) + order['total']
                total_orders = user.get('total_orders', 0) + 1
                
                # Calculate new membership tier
                new_tier, _, _ = calculate_membership_tier(total_spent)
                
                # Update user
                updates = {
                    'total_spent': total_spent,
                    'total_orders': total_orders,
                    'membership_tier': new_tier
                }
                update_user(user_id, updates)
                
                app.logger.info(f'Updated user {user_id}: spent={total_spent}, tier={new_tier}')
        
        return {'ok': True, 'order_id': order['id']}
        
    except Exception as e:
        app.logger.exception('Error submitting order')
        return {'ok': False, 'error': str(e)}, 500

# ==================== Payment Processing ====================

@app.route('/api/payment/create', methods=['POST'])
def create_payment():
    """Create payment transaction"""
    try:
        data = request.get_json()
        
        payment_method = data.get('payment_method', 'cod')
        order_id = data.get('order_id', '')
        amount = int(data.get('amount', 0))
        order_info = data.get('order_info', f'Thanh toán đơn hàng #{order_id}')
        
        if not order_id or amount <= 0:
            return {'ok': False, 'error': 'Invalid order information'}, 400
        
        # COD - no payment processing needed
        if payment_method == 'cod':
            return {'ok': True, 'payment_method': 'cod', 'order_id': order_id}
        
        # VNPay payment
        elif payment_method == 'vnpay':
            if not vnpay_payment:
                return {'ok': False, 'error': 'VNPay not configured'}, 500
            
            try:
                client_ip = get_client_ip(request)
                payment_url = vnpay_payment.create_payment_url(
                    order_id=order_id,
                    amount=amount,
                    order_desc=order_info,
                    ip_addr=client_ip
                )
                
                app.logger.info(f'Created VNPay payment for order {order_id}: {amount} VND')
                return {
                    'ok': True, 
                    'payment_method': 'vnpay',
                    'payment_url': payment_url,
                    'order_id': order_id
                }
            except Exception as e:
                app.logger.error(f'Error creating VNPay payment: {e}')
                return {'ok': False, 'error': str(e)}, 500
        
        # MoMo payment
        elif payment_method == 'momo':
            if not momo_payment:
                return {'ok': False, 'error': 'MoMo not configured'}, 500
            
            try:
                success, result, response_data = momo_payment.create_payment_url(
                    order_id=order_id,
                    amount=amount,
                    order_info=order_info
                )
                
                if success:
                    app.logger.info(f'Created MoMo payment for order {order_id}: {amount} VND')
                    return {
                        'ok': True,
                        'payment_method': 'momo',
                        'payment_url': result,
                        'order_id': order_id
                    }
                else:
                    app.logger.error(f'MoMo payment creation failed: {result}')
                    return {'ok': False, 'error': result}, 500
                    
            except Exception as e:
                app.logger.error(f'Error creating MoMo payment: {e}')
                return {'ok': False, 'error': str(e)}, 500
        
        else:
            return {'ok': False, 'error': 'Invalid payment method'}, 400
            
    except Exception as e:
        app.logger.exception('Error creating payment')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/payment/vnpay/callback')
def vnpay_callback():
    """VNPay payment callback handler"""
    try:
        if not vnpay_payment:
            flash('Hệ thống thanh toán VNPay chưa được cấu hình', 'error')
            return redirect(url_for('cart'))
        
        # Get all query parameters
        params = dict(request.args)
        
        # Verify payment
        is_valid, message, payment_data = vnpay_payment.verify_payment(params)
        
        if is_valid:
            order_id = payment_data.get('order_id')
            amount = payment_data.get('amount')
            
            # Update order status
            update_order_status(order_id, 'paid', {
                'payment_method': 'vnpay',
                'transaction_id': payment_data.get('transaction_no'),
                'bank_code': payment_data.get('bank_code'),
                'paid_at': payment_data.get('pay_date')
            })
            
            # Send payment success email
            if 'user_email' in session:
                try:
                    send_payment_success_email(session['user_email'], {
                        'order_id': order_id,
                        'amount': amount,
                        'payment_method': 'VNPay',
                        'transaction_id': payment_data.get('transaction_no')
                    })
                except Exception as e:
                    app.logger.error(f'Failed to send payment success email: {e}')
            
            # Create in-app notification
            if 'user_id' in session and session.get('user_id') != 'admin':
                try:
                    create_notification(
                        user_id=session['user_id'],
                        title='✅ Thanh toán thành công',
                        message=f'Thanh toán VNPay cho đơn hàng #{order_id} thành công. Số tiền: {amount:,.0f} đ',
                        notification_type='success',
                        link='/profile'
                    )
                    
                    # Award points for purchase (10 points per 10,000 VND)
                    points_earned = max(1, int(amount / 10000) * 10)
                    add_points(session['user_id'], points_earned, 'order', 
                             f'Mua hàng #{order_id} - {amount:,.0f}đ')
                    user = get_user_by_id(session['user_id'])
                    if user:
                        session['user_points'] = user.get('points', 0)
                    
                except Exception as e:
                    app.logger.error(f'Failed to create payment notification: {e}')
            
            app.logger.info(f'VNPay payment successful: order={order_id}, amount={amount}')
            flash(f'Thanh toán thành công! Đơn hàng #{order_id} đã được xác nhận.', 'success')
            return render_template('payment_success.html', 
                                 order_id=order_id, 
                                 amount=amount,
                                 payment_method='VNPay',
                                 datetime=datetime)
        else:
            app.logger.warning(f'VNPay payment failed: {message}')
            
            # Send payment failed email
            if 'user_email' in session:
                try:
                    order_id = params.get('vnp_TxnRef', 'N/A')
                    send_payment_failed_email(session['user_email'], order_id, message)
                except Exception as e:
                    app.logger.error(f'Failed to send payment failed email: {e}')
            
            flash(f'Thanh toán thất bại: {message}', 'error')
            return render_template('payment_failed.html', 
                                 error=message,
                                 payment_method='VNPay')
            
    except Exception as e:
        app.logger.exception('Error processing VNPay callback')
        flash('Có lỗi xảy ra khi xử lý thanh toán', 'error')
        return redirect(url_for('cart'))


@app.route('/payment/momo/callback')
def momo_callback():
    """MoMo payment callback handler (return URL)"""
    try:
        if not momo_payment:
            flash('Hệ thống thanh toán MoMo chưa được cấu hình', 'error')
            return redirect(url_for('cart'))
        
        # Get all query parameters
        params = dict(request.args)
        
        # Verify payment
        is_valid, message, payment_data = momo_payment.verify_payment(params)
        
        if is_valid:
            order_id = payment_data.get('order_id')
            amount = payment_data.get('amount')
            
            # Update order status
            update_order_status(order_id, 'paid', {
                'payment_method': 'momo',
                'transaction_id': payment_data.get('transaction_id'),
                'pay_type': payment_data.get('pay_type'),
                'paid_at': payment_data.get('response_time')
            })
            
            # Send payment success email
            if 'user_email' in session:
                try:
                    send_payment_success_email(session['user_email'], {
                        'order_id': order_id,
                        'amount': amount,
                        'payment_method': 'MoMo',
                        'transaction_id': payment_data.get('transaction_id')
                    })
                except Exception as e:
                    app.logger.error(f'Failed to send payment success email: {e}')
            
            # Create in-app notification
            if 'user_id' in session and session.get('user_id') != 'admin':
                try:
                    create_notification(
                        user_id=session['user_id'],
                        title='✅ Thanh toán thành công',
                        message=f'Thanh toán MoMo cho đơn hàng #{order_id} thành công. Số tiền: {amount:,.0f} đ',
                        notification_type='success',
                        link='/profile'
                    )
                    
                    # Award points for purchase (10 points per 10,000 VND)
                    points_earned = max(1, int(amount / 10000) * 10)
                    add_points(session['user_id'], points_earned, 'order', 
                             f'Mua hàng #{order_id} - {amount:,.0f}đ')
                    user = get_user_by_id(session['user_id'])
                    if user:
                        session['user_points'] = user.get('points', 0)
                    
                except Exception as e:
                    app.logger.error(f'Failed to create payment notification: {e}')
            
            app.logger.info(f'MoMo payment successful: order={order_id}, amount={amount}')
            flash(f'Thanh toán thành công! Đơn hàng #{order_id} đã được xác nhận.', 'success')
            return render_template('payment_success.html',
                                 order_id=order_id,
                                 amount=amount,
                                 payment_method='MoMo',
                                 datetime=datetime)
        else:
            app.logger.warning(f'MoMo payment failed: {message}')
            
            # Send payment failed email
            if 'user_email' in session:
                try:
                    order_id = params.get('orderId', 'N/A')
                    send_payment_failed_email(session['user_email'], order_id, message)
                except Exception as e:
                    app.logger.error(f'Failed to send payment failed email: {e}')
            
            flash(f'Thanh toán thất bại: {message}', 'error')
            return render_template('payment_failed.html',
                                 error=message,
                                 payment_method='MoMo')
            
    except Exception as e:
        app.logger.exception('Error processing MoMo callback')
        flash('Có lỗi xảy ra khi xử lý thanh toán', 'error')
        return redirect(url_for('cart'))


@app.route('/payment/momo/ipn', methods=['POST'])
def momo_ipn():
    """MoMo IPN (Instant Payment Notification) handler"""
    try:
        if not momo_payment:
            return {'ok': False, 'error': 'MoMo not configured'}, 500
        
        # Get POST data
        params = request.get_json() or {}
        
        # Verify payment
        is_valid, message, payment_data = momo_payment.verify_payment(params)
        
        if is_valid:
            order_id = payment_data.get('order_id')
            
            # Update order status
            update_order_status(order_id, 'paid', {
                'payment_method': 'momo',
                'transaction_id': payment_data.get('transaction_id'),
                'pay_type': payment_data.get('pay_type'),
                'paid_at': payment_data.get('response_time')
            })
            
            app.logger.info(f'MoMo IPN processed: order={order_id}')
            return {'ok': True, 'message': 'Success'}, 200
        else:
            app.logger.warning(f'Invalid MoMo IPN: {message}')
            return {'ok': False, 'error': message}, 400
            
    except Exception as e:
        app.logger.exception('Error processing MoMo IPN')
        return {'ok': False, 'error': str(e)}, 500


def update_order_status(order_id: str, status: str, payment_info: Dict = None):
    """Update order status and payment information"""
    try:
        orders_file = BASE_DIR / 'data' / 'orders.jsonl'
        if not orders_file.exists():
            return False
        
        # Read all orders
        orders = []
        with open(orders_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    order = json.loads(line.strip())
                    if order.get('id') == order_id:
                        order['status'] = status
                        if payment_info:
                            order['payment_info'] = payment_info
                            order['paid_at'] = datetime.utcnow().isoformat()
                    orders.append(order)
                except json.JSONDecodeError:
                    continue
        
        # Write back all orders
        with open(orders_file, 'w', encoding='utf-8') as f:
            for order in orders:
                f.write(json.dumps(order, ensure_ascii=False) + '\n')
        
        app.logger.info(f'Updated order {order_id} status to {status}')
        return True
        
    except Exception as e:
        app.logger.error(f'Error updating order status: {e}')
        return False

# ==================== User Authentication ====================

def hash_password(password):
    """Hash password with SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_by_email(email):
    """Get user by email"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        if not users_file.exists():
            return None
        
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    user = json.loads(line.strip())
                    if user.get('email') == email:
                        return user
                except json.JSONDecodeError:
                    continue
        return None
    except Exception:
        app.logger.exception('Error getting user')
        return None

def create_user(email, password, full_name):
    """Create new user"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        users_file.parent.mkdir(parents=True, exist_ok=True)
        
        user = {
            'id': str(uuid4())[:8],
            'email': email,
            'password': hash_password(password),
            'full_name': full_name,
            'phone': '',
            'created_at': datetime.utcnow().isoformat(),
            # Membership & Wallet
            'membership_tier': 'bronze',  # bronze, silver, gold, platinum, diamond
            'wallet_balance': 0,  # Số dư ví
            'total_spent': 0,  # Tổng chi tiêu
            'points': 0,  # Điểm tích lũy
            # Gamification
            'streak_days': 0,  # Số ngày liên tục
            'last_checkin': None,  # Lần check-in cuối
            'vouchers': [],  # Danh sách mã giảm giá từ games
            # Statistics
            'total_orders': 0,
            'total_predictions': 0,
            'quiz_completed': 0,
            'memory_completed': 0
        }
        
        with open(users_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(user, ensure_ascii=False) + '\n')
        
        return user
    except Exception:
        app.logger.exception('Error creating user')
        return None

def update_user_field(user_id, field, value):
    """Update a specific field in user data"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        if not users_file.exists():
            return False
        
        users = []
        updated = False
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    user = json.loads(line.strip())
                    if user.get('id') == user_id:
                        user[field] = value
                        updated = True
                    users.append(user)
                except json.JSONDecodeError:
                    continue
        
        if updated:
            with open(users_file, 'w', encoding='utf-8') as f:
                for user in users:
                    f.write(json.dumps(user, ensure_ascii=False) + '\n')
        
        return updated
    except Exception:
        app.logger.exception('Error updating user field')
        return False

def update_user(user_id, updates):
    """Update multiple fields in user data"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        if not users_file.exists():
            return False
        
        users = []
        updated = False
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    user = json.loads(line.strip())
                    if user.get('id') == user_id:
                        user.update(updates)
                        updated = True
                    users.append(user)
                except json.JSONDecodeError:
                    continue
        
        if updated:
            with open(users_file, 'w', encoding='utf-8') as f:
                for user in users:
                    f.write(json.dumps(user, ensure_ascii=False) + '\n')
        
        return updated
    except Exception:
        app.logger.exception('Error updating user')
        return False

def calculate_membership_tier(total_spent):
    """Calculate membership tier based on total spending"""
    if total_spent >= 50000000:  # 50 triệu
        return 'diamond', '💎 Kim Cương', '#b9f2ff'
    elif total_spent >= 20000000:  # 20 triệu
        return 'platinum', '🏆 Bạch Kim', '#e5e5e5'
    elif total_spent >= 10000000:  # 10 triệu
        return 'gold', '🥇 Vàng', '#ffd700'
    elif total_spent >= 5000000:  # 5 triệu
        return 'silver', '🥈 Bạc', '#c0c0c0'
    else:
        return 'bronze', '🥉 Đồng', '#cd7f32'

def get_membership_benefits(tier):
    """Get benefits for each membership tier"""
    benefits = {
        'bronze': {
            'discount': 0,
            'free_shipping': False,
            'voucher_boost': 1.0,
            'next_tier': 'silver',
            'next_amount': 5000000
        },
        'silver': {
            'discount': 5,
            'free_shipping': False,
            'voucher_boost': 1.2,
            'next_tier': 'gold',
            'next_amount': 10000000
        },
        'gold': {
            'discount': 10,
            'free_shipping': True,
            'voucher_boost': 1.5,
            'next_tier': 'platinum',
            'next_amount': 20000000
        },
        'platinum': {
            'discount': 15,
            'free_shipping': True,
            'voucher_boost': 2.0,
            'next_tier': 'diamond',
            'next_amount': 50000000
        },
        'diamond': {
            'discount': 20,
            'free_shipping': True,
            'voucher_boost': 3.0,
            'next_tier': None,
            'next_amount': None
        }
    }
    return benefits.get(tier, benefits['bronze'])

def add_voucher_to_user(user_id, voucher_code, voucher_value, source):
    """Add voucher from game to user"""
    try:
        user = get_user_by_id(user_id)
        if not user:
            return False
        
        vouchers = user.get('vouchers', [])
        voucher = {
            'code': voucher_code,
            'value': voucher_value,
            'source': source,  # 'quiz', 'memory_game', etc.
            'created_at': datetime.utcnow().isoformat(),
            'used': False,
            'expires_at': (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        vouchers.append(voucher)
        
        return update_user_field(user_id, 'vouchers', vouchers)
    except Exception:
        app.logger.exception('Error adding voucher to user')
        return False

def get_user_by_id(user_id):
    """Get user by ID"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        if not users_file.exists():
            return None
        
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    user = json.loads(line.strip())
                    if user.get('id') == user_id:
                        return user
                except json.JSONDecodeError:
                    continue
        return None
    except Exception:
        app.logger.exception('Error getting user by ID')
        return None

def get_user_statistics(user_id):
    """Get comprehensive user statistics"""
    try:
        user = get_user_by_id(user_id)
        if not user:
            return None
        
        # Ensure all fields exist
        user.setdefault('membership_tier', 'bronze')
        user.setdefault('wallet_balance', 0)
        user.setdefault('total_spent', 0)
        user.setdefault('points', 0)
        user.setdefault('vouchers', [])
        user.setdefault('total_orders', 0)
        user.setdefault('total_predictions', 0)
        user.setdefault('quiz_completed', 0)
        user.setdefault('memory_completed', 0)
        
        # Get actual wallet data
        wallet = get_user_wallet(user_id)
        
        # Get referral info
        referral_info = get_user_referral_info(user_id)
        
        # Calculate membership info
        tier, tier_name, tier_color = calculate_membership_tier(user['total_spent'])
        benefits = get_membership_benefits(tier)
        
        # Count active vouchers
        active_vouchers = [v for v in user['vouchers'] if not v.get('used', False) and 
                          datetime.fromisoformat(v.get('expires_at', '2000-01-01')) > datetime.utcnow()]
        
        # Calculate progress to next tier
        progress_pct = 0
        remaining = 0
        if benefits['next_tier']:
            progress_pct = min(100, (user['total_spent'] / benefits['next_amount']) * 100)
            remaining = max(0, benefits['next_amount'] - user['total_spent'])
        
        stats = {
            'user': user,
            'membership': {
                'tier': tier,
                'tier_name': tier_name,
                'tier_color': tier_color,
                'progress_pct': progress_pct,
                'remaining': remaining,
                'benefits': benefits
            },
            'wallet': {
                'balance': wallet.get('balance', 0),
                'total_spent': user['total_spent'],
                'points': user['points']
            },
            'vouchers': {
                'active': active_vouchers,
                'total_count': len(active_vouchers),
                'total_value': sum(v.get('value', 0) for v in active_vouchers)
            },
            'activity': {
                'total_orders': user['total_orders'],
                'total_predictions': user['total_predictions'],
                'quiz_completed': user['quiz_completed'],
                'memory_completed': user['memory_completed']
            },
            'referral': {
                'code': referral_info.get('code', ''),
                'referral_count': referral_info.get('referral_count', 0),
                'total_earned': referral_info.get('total_earned', 0)
            }
        }
        
        return stats
    except Exception:
        app.logger.exception('Error getting user statistics')
        return None

def get_user_addresses(user_id):
    """Get all addresses for a user"""
    try:
        addresses_file = BASE_DIR / 'data' / 'addresses.jsonl'
        if not addresses_file.exists():
            return []
        
        addresses = []
        with open(addresses_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    addr = json.loads(line.strip())
                    if addr.get('user_id') == user_id:
                        addresses.append(addr)
                except json.JSONDecodeError:
                    continue
        return addresses
    except Exception:
        app.logger.exception('Error getting addresses')
        return []

def save_address(user_id, address_data):
    """Save address for user"""
    try:
        addresses_file = BASE_DIR / 'data' / 'addresses.jsonl'
        addresses_file.parent.mkdir(parents=True, exist_ok=True)
        
        address = {
            'id': str(uuid4())[:8],
            'user_id': user_id,
            'full_name': address_data.get('full_name', ''),
            'phone': address_data.get('phone', ''),
            'address': address_data.get('address', ''),
            'city': address_data.get('city', ''),
            'district': address_data.get('district', ''),
            'is_default': address_data.get('is_default', False),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # If this is default, unset other defaults
        if address['is_default']:
            addresses = []
            if addresses_file.exists():
                with open(addresses_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            addr = json.loads(line.strip())
                            if addr.get('user_id') == user_id and addr.get('is_default'):
                                addr['is_default'] = False
                            addresses.append(addr)
                        except json.JSONDecodeError:
                            continue
                
                # Rewrite file
                with open(addresses_file, 'w', encoding='utf-8') as f:
                    for addr in addresses:
                        f.write(json.dumps(addr, ensure_ascii=False) + '\n')
        
        with open(addresses_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(address, ensure_ascii=False) + '\n')
        
        return address
    except Exception:
        app.logger.exception('Error saving address')
        return None

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập để tiếp tục', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        referral_code = request.form.get('referral_code', '').strip().upper()
        
        # Validation
        if not email or not password or not full_name:
            flash('Vui lòng điền đầy đủ thông tin', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Mật khẩu xác nhận không khớp', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Mật khẩu phải có ít nhất 6 ký tự', 'error')
            return redirect(url_for('register'))
        
        # Check if email exists
        if get_user_by_email(email):
            flash('Email đã được sử dụng', 'error')
            return redirect(url_for('register'))
        
        # Create user
        user = create_user(email, password, full_name)
        if user:
            user_id = user['id']
            
            # Apply referral code if provided
            if referral_code:
                try:
                    apply_referral_code(user_id, referral_code)
                except Exception as e:
                    app.logger.error(f'Error applying referral code: {e}')
            
            # Initialize user's referral info
            try:
                get_user_referral_info(user_id)
            except Exception as e:
                app.logger.error(f'Error initializing referral info: {e}')
            
            # Send welcome email
            try:
                send_welcome_email(email, full_name)
                app.logger.info(f'Welcome email sent to {email}')
            except Exception as e:
                app.logger.error(f'Failed to send welcome email: {e}')
            
            flash('Đăng ký thành công! Vui lòng đăng nhập để tiếp tục.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Có lỗi xảy ra. Vui lòng thử lại!', 'error')
            return redirect(url_for('register'))
    
    # Get referral code from URL if present
    referral_code_param = request.args.get('ref', '')
    return render_template('register.html', referral_code=referral_code_param)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Vui lòng điền đầy đủ thông tin', 'error')
            return redirect(url_for('login'))
        
        # Check if admin login (case-insensitive)
        if email.lower() == ADMIN_USERNAME.lower() and password == ADMIN_PASSWORD:
            session['user_id'] = 'admin'
            session['user_email'] = ADMIN_USERNAME
            session['user_name'] = 'Quản trị viên'
            session['is_admin'] = True
            flash('Đăng nhập quản trị viên thành công!', 'success')
            return redirect(url_for('admin_feedback'))
        
        # Check regular user (convert email to lowercase for DB lookup)
        user = get_user_by_email(email.lower())
        if user and user['password'] == hash_password(password):
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_name'] = user['full_name']
            session['user_points'] = user.get('points', 0)
            session['is_admin'] = False
            flash('Đăng nhập thành công!', 'success')
            
            # Redirect to next page or profile
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('profile'))
        else:
            flash('Email hoặc mật khẩu không đúng', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Đã đăng xuất', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    # Check if admin
    if session.get('is_admin'):
        user = {
            'id': 'admin',
            'email': ADMIN_USERNAME,
            'full_name': 'Quản trị viên',
            'created_at': ''
        }
        addresses = []
        orders = []
        stats = None
        return render_template('profile.html', user=user, addresses=addresses, orders=orders, stats=stats)
    
    user_id = session.get('user_id')
    user = get_user_by_email(session.get('user_email'))
    
    # Get comprehensive user statistics
    stats = get_user_statistics(user_id)
    
    # Get addresses
    addresses = get_user_addresses(user_id)
    
    # Get user orders
    orders = []
    try:
        orders_file = BASE_DIR / 'data' / 'orders.jsonl'
        if orders_file.exists():
            with open(orders_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        order = json.loads(line.strip())
                        if order.get('user_id') == user_id:
                            orders.append(order)
                    except json.JSONDecodeError:
                        continue
            # Sort by timestamp desc
            orders.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    except Exception:
        app.logger.exception('Error loading orders')
    
    return render_template('profile.html', user=user, addresses=addresses, orders=orders, stats=stats)

@app.route('/api/address/add', methods=['POST'])
@login_required
def add_address():
    """Add new address"""
    try:
        data = request.get_json()
        address = save_address(session.get('user_id'), data)
        
        if address:
            return {'ok': True, 'address': address}
        else:
            return {'ok': False, 'error': 'Không thể lưu địa chỉ'}, 500
    except Exception as e:
        app.logger.exception('Error adding address')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/address/delete/<address_id>', methods=['POST'])
@login_required
def delete_address(address_id):
    """Delete address"""
    try:
        addresses_file = BASE_DIR / 'data' / 'addresses.jsonl'
        if not addresses_file.exists():
            return {'ok': False, 'error': 'File not found'}, 404
        
        addresses = []
        with open(addresses_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    addr = json.loads(line.strip())
                    # Keep if not matching or not user's address
                    if addr.get('id') != address_id or addr.get('user_id') != session.get('user_id'):
                        addresses.append(addr)
                except json.JSONDecodeError:
                    continue
        
        # Rewrite file
        with open(addresses_file, 'w', encoding='utf-8') as f:
            for addr in addresses:
                f.write(json.dumps(addr, ensure_ascii=False) + '\n')
        
        return {'ok': True}
    except Exception as e:
        app.logger.exception('Error deleting address')
        return {'ok': False, 'error': str(e)}, 500

# ==================== In-App Notifications ====================

def create_notification(user_id: str, title: str, message: str, 
                       notification_type: str = 'info', link: str = None):
    """Create in-app notification for user"""
    try:
        notifications_file = BASE_DIR / 'data' / 'notifications.jsonl'
        notifications_file.parent.mkdir(parents=True, exist_ok=True)
        
        notification = {
            'id': str(uuid4())[:8],
            'user_id': user_id,
            'title': title,
            'message': message,
            'type': notification_type,  # info, success, warning, error
            'link': link,
            'read': False,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(notifications_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(notification, ensure_ascii=False) + '\n')
        
        app.logger.info(f'Notification created for user {user_id}: {title}')
        return notification
    except Exception as e:
        app.logger.error(f'Error creating notification: {e}')
        return None


def get_user_notifications(user_id: str, unread_only: bool = False, limit: int = 50):
    """Get notifications for user"""
    try:
        notifications_file = BASE_DIR / 'data' / 'notifications.jsonl'
        if not notifications_file.exists():
            return []
        
        notifications = []
        with open(notifications_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    notif = json.loads(line.strip())
                    if notif.get('user_id') == user_id:
                        if unread_only and notif.get('read'):
                            continue
                        notifications.append(notif)
                except json.JSONDecodeError:
                    continue
        
        # Sort by timestamp descending
        notifications.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Limit results
        return notifications[:limit]
    except Exception as e:
        app.logger.error(f'Error getting notifications: {e}')
        return []


def mark_notification_read(notification_id: str, user_id: str):
    """Mark notification as read"""
    try:
        notifications_file = BASE_DIR / 'data' / 'notifications.jsonl'
        if not notifications_file.exists():
            return False
        
        notifications = []
        updated = False
        
        with open(notifications_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    notif = json.loads(line.strip())
                    if notif.get('id') == notification_id and notif.get('user_id') == user_id:
                        notif['read'] = True
                        updated = True
                    notifications.append(notif)
                except json.JSONDecodeError:
                    continue
        
        if updated:
            with open(notifications_file, 'w', encoding='utf-8') as f:
                for notif in notifications:
                    f.write(json.dumps(notif, ensure_ascii=False) + '\n')
        
        return updated
    except Exception as e:
        app.logger.error(f'Error marking notification read: {e}')
        return False


def mark_all_notifications_read(user_id: str):
    """Mark all notifications as read for user"""
    try:
        notifications_file = BASE_DIR / 'data' / 'notifications.jsonl'
        if not notifications_file.exists():
            return False
        
        notifications = []
        updated = False
        
        with open(notifications_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    notif = json.loads(line.strip())
                    if notif.get('user_id') == user_id and not notif.get('read'):
                        notif['read'] = True
                        updated = True
                    notifications.append(notif)
                except json.JSONDecodeError:
                    continue
        
        if updated:
            with open(notifications_file, 'w', encoding='utf-8') as f:
                for notif in notifications:
                    f.write(json.dumps(notif, ensure_ascii=False) + '\n')
        
        return updated
    except Exception as e:
        app.logger.error(f'Error marking all notifications read: {e}')
        return False


@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Get user notifications"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    user_id = session['user_id']
    unread_only = request.args.get('unread_only', 'false').lower() == 'true'
    
    notifications = get_user_notifications(user_id, unread_only=unread_only)
    unread_count = len([n for n in notifications if not n.get('read')])
    
    return {
        'ok': True,
        'notifications': notifications,
        'unread_count': unread_count
    }


@app.route('/api/notifications/<notification_id>/read', methods=['POST'])
def mark_notification_as_read(notification_id):
    """Mark notification as read"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    user_id = session['user_id']
    success = mark_notification_read(notification_id, user_id)
    
    return {'ok': success}


@app.route('/api/notifications/read_all', methods=['POST'])
def mark_all_as_read():
    """Mark all notifications as read"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Not logged in'}, 401
    
    user_id = session['user_id']
    success = mark_all_notifications_read(user_id)
    
    return {'ok': success}


# ==================== Coupon System ====================

@app.route('/api/coupon/validate', methods=['POST'])
def validate_coupon():
    """Validate coupon code"""
    try:
        data = request.get_json()
        coupon_code = data.get('code', '').strip().upper()
        order_total = data.get('order_total', 0)
        
        if not coupon_code:
            return {'ok': False, 'error': 'Vui lòng nhập mã giảm giá'}, 400
        
        # Load coupons
        coupons_file = BASE_DIR / 'data' / 'coupons.jsonl'
        if not coupons_file.exists():
            return {'ok': False, 'error': 'Mã giảm giá không tồn tại'}, 404
        
        coupon = None
        with open(coupons_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    if c['code'].upper() == coupon_code:
                        coupon = c
                        break
        
        if not coupon:
            return {'ok': False, 'error': 'Mã giảm giá không hợp lệ'}, 404
        
        # Check expiry
        from datetime import datetime
        if 'expires' in coupon:
            expires = datetime.fromisoformat(coupon['expires'])
            if datetime.now() > expires:
                return {'ok': False, 'error': 'Mã giảm giá đã hết hạn'}, 400
        
        # Check usage limit
        if coupon.get('used_count', 0) >= coupon.get('usage_limit', 9999):
            return {'ok': False, 'error': 'Mã giảm giá đã hết lượt sử dụng'}, 400
        
        # Check minimum order
        if order_total < coupon.get('min_order', 0):
            return {
                'ok': False, 
                'error': f'Đơn hàng tối thiểu {coupon["min_order"]:,.0f}đ để sử dụng mã này'
            }, 400
        
        # Calculate discount
        discount_amount = 0
        if coupon['discount_type'] == 'percent':
            discount_amount = order_total * coupon['discount_value'] / 100
            discount_amount = min(discount_amount, coupon.get('max_discount', discount_amount))
        elif coupon['discount_type'] == 'fixed':
            discount_amount = coupon['discount_value']
        elif coupon['discount_type'] == 'shipping':
            # This will be handled separately for shipping fee
            discount_amount = coupon['discount_value']
        
        return {
            'ok': True,
            'coupon': {
                'code': coupon['code'],
                'discount_type': coupon['discount_type'],
                'discount_value': coupon['discount_value'],
                'discount_amount': discount_amount,
                'max_discount': coupon.get('max_discount', 0),
                'description': coupon.get('description', '')
            }
        }
        
    except Exception as e:
        app.logger.error(f'Error validating coupon: {e}')
        return {'ok': False, 'error': 'Lỗi khi kiểm tra mã giảm giá'}, 500


# ==================== Product Management ====================

def load_products():
    """Load all products from database"""
    products = []
    products_file = BASE_DIR / 'data' / 'products.jsonl'
    
    if products_file.exists():
        with open(products_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    products.append(json.loads(line))
    
    return products

def get_product_by_id(product_id):
    """Get product by ID"""
    products = load_products()
    for product in products:
        if product['id'] == int(product_id):
            return product
    return None

def save_product(product_data):
    """Save or update product"""
    products = load_products()
    products_file = BASE_DIR / 'data' / 'products.jsonl'
    
    # Check if updating existing product
    product_id = product_data.get('id')
    if product_id:
        products = [p for p in products if p['id'] != product_id]
    else:
        # Generate new ID
        max_id = max([p['id'] for p in products], default=0)
        product_data['id'] = max_id + 1
        product_data['created_at'] = datetime.utcnow().isoformat()
    
    products.append(product_data)
    
    # Save back to file
    with open(products_file, 'w', encoding='utf-8') as f:
        for product in products:
            f.write(json.dumps(product, ensure_ascii=False) + '\n')
    
    return product_data

def delete_product(product_id):
    """Delete product"""
    products = load_products()
    products = [p for p in products if p['id'] != int(product_id)]
    
    products_file = BASE_DIR / 'data' / 'products.jsonl'
    with open(products_file, 'w', encoding='utf-8') as f:
        for product in products:
            f.write(json.dumps(product, ensure_ascii=False) + '\n')

def update_product_stock(product_id, quantity_change):
    """Update product stock"""
    product = get_product_by_id(product_id)
    if product:
        product['stock'] = max(0, product.get('stock', 0) + quantity_change)
        product['sold'] = product.get('sold', 0) + abs(quantity_change) if quantity_change < 0 else product.get('sold', 0)
        save_product(product)
        return product
    return None

def send_low_stock_alert(product):
    """Send low stock alert to all admins"""
    try:
        stock = product.get('stock', 0)
        if stock >= 10:
            return
        
        # Find all admin users
        users = load_users()
        admin_users = [u for u in users if u.get('is_admin', False)]
        
        for admin in admin_users:
            create_notification(
                user_id=admin['id'],
                title='⚠️ Cảnh báo tồn kho thấp',
                message=f'Sản phẩm "{product["name"]}" chỉ còn {stock} sản phẩm trong kho!',
                notification_type='warning',
                link='/admin/products'
            )
        
        app.logger.warning(f'Low stock alert: {product["name"]} -只 còn {stock}')
    except Exception as e:
        app.logger.error(f'Error sending low stock alert: {e}')

@app.route('/api/products', methods=['GET'])
def api_get_products():
    """API to get products with filters"""
    try:
        products = load_products()
        
        # Filter by category
        category = request.args.get('category')
        if category and category != 'all':
            products = [p for p in products if p['category'] == category]
        
        # Filter by status
        status = request.args.get('status', 'active')
        if status != 'all':
            products = [p for p in products if p.get('status', 'active') == status]
        
        # Sort
        sort_by = request.args.get('sort', 'name')
        if sort_by == 'price_asc':
            products = sorted(products, key=lambda x: x['price'])
        elif sort_by == 'price_desc':
            products = sorted(products, key=lambda x: x['price'], reverse=True)
        elif sort_by == 'rating':
            products = sorted(products, key=lambda x: x.get('rating_avg', 0), reverse=True)
        elif sort_by == 'sold':
            products = sorted(products, key=lambda x: x.get('sold', 0), reverse=True)
        
        return {'ok': True, 'products': products}
    except Exception as e:
        app.logger.error(f'Error loading products: {e}')
        return {'ok': False, 'error': str(e)}, 500


# ==================== Wishlist / Favorites ====================

def load_wishlists():
    """Load all wishlists"""
    wishlists = []
    wishlists_file = BASE_DIR / 'data' / 'wishlists.jsonl'
    
    if wishlists_file.exists():
        with open(wishlists_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    wishlists.append(json.loads(line))
    
    return wishlists

def get_user_wishlist(user_id):
    """Get wishlist for a specific user"""
    wishlists = load_wishlists()
    for wishlist in wishlists:
        if wishlist['user_id'] == user_id:
            return wishlist
    
    # Create new wishlist if doesn't exist
    return {
        'user_id': user_id,
        'product_ids': [],
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }

def save_wishlist(wishlist_data):
    """Save or update user's wishlist"""
    wishlists = load_wishlists()
    wishlists_file = BASE_DIR / 'data' / 'wishlists.jsonl'
    
    user_id = wishlist_data['user_id']
    wishlists = [w for w in wishlists if w['user_id'] != user_id]
    
    wishlist_data['updated_at'] = datetime.utcnow().isoformat()
    wishlists.append(wishlist_data)
    
    with open(wishlists_file, 'w', encoding='utf-8') as f:
        for wishlist in wishlists:
            f.write(json.dumps(wishlist, ensure_ascii=False) + '\n')
    
    return wishlist_data

def add_to_wishlist(user_id, product_id):
    """Add product to user's wishlist"""
    wishlist = get_user_wishlist(user_id)
    product_id = int(product_id)
    
    if product_id not in wishlist['product_ids']:
        wishlist['product_ids'].append(product_id)
        save_wishlist(wishlist)
        return True
    return False

def remove_from_wishlist(user_id, product_id):
    """Remove product from user's wishlist"""
    wishlist = get_user_wishlist(user_id)
    product_id = int(product_id)
    
    if product_id in wishlist['product_ids']:
        wishlist['product_ids'].remove(product_id)
        save_wishlist(wishlist)
        return True
    return False

@app.route('/api/wishlist', methods=['GET'])
@login_required
def api_get_wishlist():
    """Get user's wishlist with product details"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403

        user_id = session.get('user_id')
        wishlist = get_user_wishlist(user_id)
        
        # Get full product details for each wishlist item
        products = []
        for product_id in wishlist['product_ids']:
            product = get_product_by_id(product_id)
            if product and product.get('status') == 'active':
                products.append(product)
        
        return {'ok': True, 'products': products, 'count': len(products)}
    except Exception as e:
        app.logger.error(f'Error loading wishlist: {e}')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/wishlist/add/<int:product_id>', methods=['POST'])
@login_required
def api_add_to_wishlist(product_id):
    """Add product to wishlist"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403

        user_id = session.get('user_id')
        
        # Check if product exists
        product = get_product_by_id(product_id)
        if not product:
            return {'ok': False, 'error': 'Sản phẩm không tồn tại'}, 404
        
        success = add_to_wishlist(user_id, product_id)
        
        if success:
            return {'ok': True, 'message': 'Đã thêm vào danh sách yêu thích'}
        else:
            return {'ok': False, 'error': 'Sản phẩm đã có trong danh sách yêu thích'}, 400
    except Exception as e:
        app.logger.error(f'Error adding to wishlist: {e}')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/wishlist/remove/<int:product_id>', methods=['POST'])
@login_required
def api_remove_from_wishlist(product_id):
    """Remove product from wishlist"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403

        user_id = session.get('user_id')
        success = remove_from_wishlist(user_id, product_id)
        
        if success:
            return {'ok': True, 'message': 'Đã xóa khỏi danh sách yêu thích'}
        else:
            return {'ok': False, 'error': 'Sản phẩm không có trong danh sách yêu thích'}, 400
    except Exception as e:
        app.logger.error(f'Error removing from wishlist: {e}')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/wishlist/check/<int:product_id>', methods=['GET'])
@login_required
def api_check_wishlist(product_id):
    """Check if product is in user's wishlist"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403

        user_id = session.get('user_id')
        wishlist = get_user_wishlist(user_id)
        
        is_in_wishlist = int(product_id) in wishlist['product_ids']
        return {'ok': True, 'in_wishlist': is_in_wishlist}
    except Exception as e:
        app.logger.error(f'Error checking wishlist: {e}')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/wishlist')
@login_required
def wishlist_page():
    """Wishlist page"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    return render_template('wishlist.html', 
                         user=session,
                         active_page='wishlist')


# ==================== Advanced Search & Filter ====================

@app.route('/api/search', methods=['GET'])
def api_search():
    """Universal search endpoint - search across products, wiki, history, community"""
    try:
        query = request.args.get('q', '').strip()
        search_type = request.args.get('type', 'all')  # all, products, wiki, history, community
        
        if not query:
            return {'ok': False, 'error': 'Query is required'}, 400
        
        results = {}
        query_lower = query.lower()
        
        # Search products
        if search_type in ['all', 'products']:
            products = load_products()
            products = [p for p in products if p.get('status') == 'active']
            product_results = []
            
            for product in products:
                score = 0
                if query_lower in product.get('name', '').lower():
                    score += 10
                if query_lower in product.get('description', '').lower():
                    score += 5
                if query_lower in product.get('category', '').lower():
                    score += 3
                
                if score > 0:
                    product['search_score'] = score
                    product_results.append(product)
            
            product_results.sort(key=lambda x: x.get('search_score', 0), reverse=True)
            results['products'] = product_results[:10]
        
        # Search wiki articles
        if search_type in ['all', 'wiki']:
            wiki_results = search_wiki_articles(query)
            results['wiki'] = wiki_results[:10]
        
        # Search history (if user logged in)
        if search_type in ['all', 'history'] and 'user_id' in session:
            history_list = _read_history_file()
            user_id = session.get('user_id')
            user_history = [h for h in history_list if h.get('user_id') == user_id]
            
            history_results = []
            for entry in user_history:
                if (query_lower in entry.get('predicted_label', '').lower() or
                    query_lower in entry.get('model_name', '').lower() or
                    query_lower in DISEASE_INFO.get(entry.get('predicted_label', ''), {}).get('name', '').lower()):
                    history_results.append(entry)
            
            results['history'] = history_results[:10]
        
        # Search community posts
        if search_type in ['all', 'community']:
            posts = load_posts()
            post_results = []
            
            for post in posts:
                if (query_lower in post.get('content', '').lower() or
                    query_lower in post.get('user_name', '').lower()):
                    post_results.append(post)
            
            post_results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            results['community'] = post_results[:10]
        
        return {'ok': True, 'query': query, 'results': results}
    except Exception as e:
        app.logger.error(f'Error in universal search: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/products/filter', methods=['GET'])
def api_products_filter():
    """Advanced products filter with multiple criteria"""
    try:
        products = load_products()
        
        # Filter by category
        category = request.args.get('category')
        if category and category != 'all':
            products = [p for p in products if p.get('category') == category]
        
        # Filter by status
        status = request.args.get('status', 'active')
        if status != 'all':
            products = [p for p in products if p.get('status', 'active') == status]
        
        # Filter by price range
        min_price = request.args.get('min_price', type=int)
        max_price = request.args.get('max_price', type=int)
        if min_price is not None:
            products = [p for p in products if p.get('price', 0) >= min_price]
        if max_price is not None:
            products = [p for p in products if p.get('price', 0) <= max_price]
        
        # Filter by stock availability
        in_stock_only = request.args.get('in_stock', '').lower() == 'true'
        if in_stock_only:
            products = [p for p in products if p.get('stock', 0) > 0]
        
        # Filter by rating
        min_rating = request.args.get('min_rating', type=float)
        if min_rating is not None:
            products = [p for p in products if p.get('rating_avg', 0) >= min_rating]
        
        # Search by name or description
        search_query = request.args.get('q', '').strip()
        if search_query:
            query_lower = search_query.lower()
            products = [p for p in products 
                       if query_lower in p.get('name', '').lower() or 
                          query_lower in p.get('description', '').lower()]
        
        # Sort
        sort_by = request.args.get('sort', 'name')
        if sort_by == 'price_asc':
            products = sorted(products, key=lambda x: x.get('price', 0))
        elif sort_by == 'price_desc':
            products = sorted(products, key=lambda x: x.get('price', 0), reverse=True)
        elif sort_by == 'rating':
            products = sorted(products, key=lambda x: x.get('rating_avg', 0), reverse=True)
        elif sort_by == 'sold':
            products = sorted(products, key=lambda x: x.get('sold', 0), reverse=True)
        elif sort_by == 'newest':
            products = sorted(products, key=lambda x: x.get('created_at', ''), reverse=True)
        elif sort_by == 'name':
            products = sorted(products, key=lambda x: x.get('name', ''))
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 12, type=int)
        total = len(products)
        total_pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        products_page = products[start:end]
        
        return {
            'ok': True, 
            'products': products_page,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': total_pages,
                'has_prev': page > 1,
                'has_next': page < total_pages
            }
        }
    except Exception as e:
        app.logger.error(f'Error filtering products: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/history/filter', methods=['GET'])
@login_required
def api_history_filter():
    """Filter prediction history with multiple criteria"""
    try:
        history_list = _read_history_file()
        
        # Filter by user (non-admin only sees their own)
        if not session.get('is_admin'):
            user_id = session.get('user_id')
            history_list = [h for h in history_list if h.get('user_id') == user_id]
        
        # Filter by disease type
        disease = request.args.get('disease')
        if disease and disease != 'all':
            history_list = [h for h in history_list if h.get('predicted_label') == disease]
        
        # Filter by model
        model = request.args.get('model')
        if model and model != 'all':
            history_list = [h for h in history_list if h.get('model_name') == model]
        
        # Filter by confidence range
        min_conf = request.args.get('min_confidence', type=float)
        max_conf = request.args.get('max_confidence', type=float)
        if min_conf is not None:
            history_list = [h for h in history_list if h.get('probability', 0) >= min_conf]
        if max_conf is not None:
            history_list = [h for h in history_list if h.get('probability', 0) <= max_conf]
        
        # Filter by date range
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        if start_date:
            history_list = [h for h in history_list if h.get('timestamp', '') >= start_date]
        if end_date:
            # Add 1 day to include end_date
            end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
            history_list = [h for h in history_list if h.get('timestamp', '') < end_dt.isoformat()]
        
        # Search by filename or label
        search_query = request.args.get('q', '').strip()
        if search_query:
            query_lower = search_query.lower()
            history_list = [h for h in history_list 
                          if query_lower in h.get('filename', '').lower() or
                             query_lower in h.get('predicted_label', '').lower() or
                             query_lower in DISEASE_INFO.get(h.get('predicted_label', ''), {}).get('name', '').lower()]
        
        # Sort
        sort_by = request.args.get('sort', 'date_desc')
        if sort_by == 'date_desc':
            history_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        elif sort_by == 'date_asc':
            history_list.sort(key=lambda x: x.get('timestamp', ''))
        elif sort_by == 'confidence_desc':
            history_list.sort(key=lambda x: x.get('probability', 0), reverse=True)
        elif sort_by == 'confidence_asc':
            history_list.sort(key=lambda x: x.get('probability', 0))
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        total = len(history_list)
        total_pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        history_page = history_list[start:end]
        
        return {
            'ok': True,
            'history': history_page,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': total_pages,
                'has_prev': page > 1,
                'has_next': page < total_pages
            },
            'stats': {
                'total_predictions': total,
                'diseases': list(set(h.get('predicted_label', '') for h in history_list))
            }
        }
    except Exception as e:
        app.logger.error(f'Error filtering history: {e}')
        return {'ok': False, 'error': str(e)}, 500


# ==================== Order Management ====================

def load_orders():
    """Load all orders"""
    orders = []
    orders_file = BASE_DIR / 'data' / 'orders.jsonl'
    
    if orders_file.exists():
        with open(orders_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    orders.append(json.loads(line))
    
    return orders

def get_order_by_id(order_id):
    """Get order by ID"""
    orders = load_orders()
    for order in orders:
        if order['id'] == order_id:
            return order
    return None

def update_order_status_new(order_id, new_status, admin_note=''):
    """Update order status"""
    orders = load_orders()
    orders_file = BASE_DIR / 'data' / 'orders.jsonl'
    
    for order in orders:
        if order['id'] == order_id:
            old_status = order.get('status', 'pending')
            order['status'] = new_status
            order['status_updated_at'] = datetime.utcnow().isoformat()
            
            # Restore stock if order is being cancelled
            if new_status == 'cancelled' and old_status != 'cancelled':
                try:
                    for item in order.get('items', []):
                        update_product_stock(item['id'], item['quantity'])
                        app.logger.info(f"Restored {item['quantity']} units of product {item['id']} from cancelled order {order_id}")
                except Exception as e:
                    app.logger.error(f'Error restoring stock for cancelled order {order_id}: {e}')
            
            if admin_note:
                if 'status_history' not in order:
                    order['status_history'] = []
                order['status_history'].append({
                    'from': old_status,
                    'to': new_status,
                    'note': admin_note,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Save
            with open(orders_file, 'w', encoding='utf-8') as f:
                for o in orders:
                    f.write(json.dumps(o, ensure_ascii=False) + '\n')
            
            # Send notification to user
            if order.get('user_id'):
                status_messages = {
                    'processing': 'Đơn hàng đang được xử lý',
                    'shipped': 'Đơn hàng đã được giao cho đơn vị vận chuyển',
                    'delivered': 'Đơn hàng đã được giao thành công',
                    'cancelled': 'Đơn hàng đã bị hủy'
                }
                
                try:
                    create_notification(
                        user_id=order['user_id'],
                        title=f'📦 Cập nhật đơn hàng #{order_id}',
                        message=status_messages.get(new_status, f'Trạng thái: {new_status}'),
                        notification_type='info',
                        link='/profile'
                    )
                except Exception as e:
                    app.logger.error(f'Failed to create notification: {e}')
                
                # Check purchase achievements when order is completed/delivered
                if new_status in ['completed', 'delivered']:
                    try:
                        check_and_award_achievements(
                            order['user_id'], 
                            'purchase', 
                            {'order_id': order_id, 'amount': order.get('total', 0)}
                        )
                    except Exception as e:
                        app.logger.error(f'Failed to check purchase achievements: {e}')
            
            return True
    
    return False


def get_order_tracking_info(order_id):
    """Get detailed tracking information for an order with timeline and milestones"""
    order = get_order_by_id(order_id)
    if not order:
        return None
    
    # Define order lifecycle milestones
    milestones = [
        {
            'status': 'pending',
            'title': 'Đơn hàng đã đặt',
            'icon': '📝',
            'description': 'Đơn hàng của bạn đã được tiếp nhận'
        },
        {
            'status': 'processing',
            'title': 'Đang xử lý',
            'icon': '⚙️',
            'description': 'Đơn hàng đang được chuẩn bị'
        },
        {
            'status': 'shipped',
            'title': 'Đã giao vận',
            'icon': '🚚',
            'description': 'Đơn hàng đã được giao cho đơn vị vận chuyển'
        },
        {
            'status': 'delivered',
            'title': 'Đã giao hàng',
            'icon': '✅',
            'description': 'Đơn hàng đã được giao thành công'
        }
    ]
    
    # Map current status to milestone
    current_status = order.get('status', 'pending')
    status_order = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
    current_index = status_order.index(current_status) if current_status in status_order else 0
    
    # Mark completed milestones
    for i, milestone in enumerate(milestones):
        if i <= current_index and current_status != 'cancelled':
            milestone['completed'] = True
            milestone['active'] = (i == current_index)
        else:
            milestone['completed'] = False
            milestone['active'] = False
    
    # Handle cancelled status
    if current_status == 'cancelled':
        for milestone in milestones:
            milestone['completed'] = False
            milestone['active'] = False
        milestones.append({
            'status': 'cancelled',
            'title': 'Đã hủy',
            'icon': '❌',
            'description': 'Đơn hàng đã bị hủy',
            'completed': True,
            'active': True
        })
    
    # Calculate estimated delivery date
    created_at = datetime.fromisoformat(order.get('timestamp', datetime.now().isoformat()))
    
    # Estimate based on status
    if current_status == 'pending':
        estimated_delivery = created_at + timedelta(days=5)
    elif current_status == 'processing':
        estimated_delivery = created_at + timedelta(days=4)
    elif current_status == 'shipped':
        estimated_delivery = created_at + timedelta(days=2)
    elif current_status == 'delivered':
        estimated_delivery = datetime.fromisoformat(order.get('status_updated_at', order.get('timestamp')))
    else:
        estimated_delivery = None
    
    # Get status history timeline
    status_history = order.get('status_history', [])
    timeline = []
    
    # Add order created event
    timeline.append({
        'status': 'pending',
        'title': 'Đơn hàng được tạo',
        'timestamp': order.get('timestamp'),
        'note': f'Khách hàng: {order.get("customer_name", "N/A")}'
    })
    
    # Add status changes
    for history_item in status_history:
        timeline.append({
            'status': history_item.get('to'),
            'title': f'Cập nhật: {history_item.get("to", "N/A")}',
            'timestamp': history_item.get('timestamp'),
            'note': history_item.get('note', '')
        })
    
    # Sort timeline by timestamp
    timeline.sort(key=lambda x: x.get('timestamp', ''))
    
    return {
        'order': order,
        'milestones': milestones,
        'timeline': timeline,
        'current_status': current_status,
        'estimated_delivery': estimated_delivery.isoformat() if estimated_delivery else None,
        'estimated_delivery_formatted': estimated_delivery.strftime('%d/%m/%Y') if estimated_delivery else None
    }


@app.route('/api/order/<order_id>/tracking', methods=['GET'])
def api_order_tracking(order_id):
    """API endpoint for order tracking"""
    try:
        # Get order
        order = get_order_by_id(order_id)
        if not order:
            return {'ok': False, 'error': 'Đơn hàng không tồn tại'}, 404
        
        # Check permission
        if not session.get('is_admin'):
            # User can only track their own orders
            if 'user_id' not in session or order.get('user_id') != session.get('user_id'):
                return {'ok': False, 'error': 'Bạn không có quyền xem đơn hàng này'}, 403
        
        tracking_info = get_order_tracking_info(order_id)
        
        return {'ok': True, 'tracking': tracking_info}
    except Exception as e:
        app.logger.error(f'Error getting order tracking: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/order/track/<order_id>')
def order_tracking_page(order_id):
    """User-facing order tracking page"""
    try:
        # Get order
        order = get_order_by_id(order_id)
        if not order:
            flash('Đơn hàng không tồn tại', 'error')
            return redirect(url_for('index'))
        
        # Check permission
        if not session.get('is_admin'):
            if 'user_id' not in session or order.get('user_id') != session.get('user_id'):
                flash('Bạn không có quyền xem đơn hàng này', 'error')
                return redirect(url_for('index'))
        
        tracking_info = get_order_tracking_info(order_id)
        
        return render_template('order_tracking.html', 
                             tracking=tracking_info,
                             datetime=datetime)
    except Exception as e:
        app.logger.error(f'Error loading order tracking page: {e}')
        flash('Không thể tải thông tin đơn hàng', 'error')
        return redirect(url_for('profile'))


@app.route('/api/orders/my', methods=['GET'])
@login_required
def api_my_orders():
    """Get current user's orders with filters"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin account'}, 403
        
        user_id = session.get('user_id')
        all_orders = load_orders()
        user_orders = [o for o in all_orders if o.get('user_id') == user_id]
        
        # Filter by status
        status = request.args.get('status')
        if status and status != 'all':
            user_orders = [o for o in user_orders if o.get('status') == status]
        
        # Filter by payment method
        payment = request.args.get('payment')
        if payment and payment != 'all':
            user_orders = [o for o in user_orders if o.get('payment_method') == payment]
        
        # Filter by date range
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        if start_date:
            user_orders = [o for o in user_orders if o.get('timestamp', '') >= start_date]
        if end_date:
            end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
            user_orders = [o for o in user_orders if o.get('timestamp', '') < end_dt.isoformat()]
        
        # Sort
        sort_by = request.args.get('sort', 'date_desc')
        if sort_by == 'date_desc':
            user_orders.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        elif sort_by == 'date_asc':
            user_orders.sort(key=lambda x: x.get('timestamp', ''))
        elif sort_by == 'total_desc':
            user_orders.sort(key=lambda x: x.get('total', 0), reverse=True)
        elif sort_by == 'total_asc':
            user_orders.sort(key=lambda x: x.get('total', 0))
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        total = len(user_orders)
        total_pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        orders_page = user_orders[start:end]
        
        # Calculate stats
        stats = {
            'total_orders': total,
            'total_spent': sum(o.get('total', 0) for o in user_orders),
            'pending_orders': len([o for o in user_orders if o.get('status') == 'pending']),
            'delivered_orders': len([o for o in user_orders if o.get('status') == 'delivered'])
        }
        
        return {
            'ok': True,
            'orders': orders_page,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': total_pages,
                'has_prev': page > 1,
                'has_next': page < total_pages
            },
            'stats': stats
        }
    except Exception as e:
        app.logger.error(f'Error getting user orders: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/order/<order_id>/cancel', methods=['POST'])
@login_required
def api_cancel_order(order_id):
    """Cancel order (only pending orders can be cancelled by users)"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin cannot cancel orders this way'}, 403
        
        user_id = session.get('user_id')
        order = get_order_by_id(order_id)
        
        if not order:
            return {'ok': False, 'error': 'Đơn hàng không tồn tại'}, 404
        
        # Check permission
        if order.get('user_id') != user_id:
            return {'ok': False, 'error': 'Bạn không có quyền hủy đơn hàng này'}, 403
        
        # Only pending orders can be cancelled
        if order.get('status') != 'pending':
            return {'ok': False, 'error': 'Chỉ có thể hủy đơn hàng đang chờ xử lý'}, 400
        
        # Update status
        success = update_order_status_new(
            order_id, 
            'cancelled', 
            'Khách hàng hủy đơn'
        )
        
        if success:
            # Create notification
            try:
                create_notification(
                    user_id=user_id,
                    title='❌ Đơn hàng đã hủy',
                    message=f'Đơn hàng #{order_id} đã được hủy thành công',
                    notification_type='info',
                    link='/profile'
                )
            except Exception as e:
                app.logger.error(f'Failed to create cancel notification: {e}')
            
            app.logger.info(f'Order {order_id} cancelled by user {user_id}')
            return {'ok': True, 'message': 'Đơn hàng đã được hủy'}
        else:
            return {'ok': False, 'error': 'Không thể hủy đơn hàng'}, 500
    except Exception as e:
        app.logger.error(f'Error cancelling order: {e}')
        return {'ok': False, 'error': str(e)}, 500


# ==================== Weather API Routes ====================

@app.route('/api/weather/current', methods=['GET'])
def api_weather_current():
    """Get current weather with disease risk assessment"""
    try:
        city = request.args.get('city', 'Hanoi')
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        
        # Create cache key
        if lat and lon:
            cache_key = f"weather:coord:{lat}:{lon}"
        else:
            cache_key = f"weather:city:{city}"
        
        # Check cache
        cached = get_cached_weather(cache_key)
        if cached:
            return {'success': True, 'weather': cached, 'cached': True}
        
        # Fetch from API
        weather_data = fetch_weather_from_api(city=city, lat=lat, lon=lon)
        
        if not weather_data:
            return {
                'success': False, 
                'error': 'API key not configured or request failed. Check OPENWEATHER_API_KEY in .env file.'
            }, 500
        
        # Assess disease risk
        risk = assess_disease_risk_from_weather(weather_data)
        weather_data['disease_risk'] = risk
        
        # Cache result
        cache_weather(cache_key, weather_data)
        
        return {'success': True, 'weather': weather_data, 'cached': False}
    
    except Exception as e:
        app.logger.error(f'Weather API error: {e}')
        return {'success': False, 'error': str(e)}, 500


@app.route('/api/weather/forecast', methods=['GET'])
def api_weather_forecast():
    """Get weather forecast for next 5 days"""
    try:
        city = request.args.get('city', 'Hanoi')
        days = int(request.args.get('days', 5))
        
        if not OPENWEATHER_API_KEY:
            return {'success': False, 'error': 'API key not configured'}, 500
        
        # Check cache
        cache_key = f"forecast:city:{city}"
        cached = get_cached_weather(cache_key)
        if cached:
            return {'success': True, 'forecast': cached, 'cached': True}
        
        # Fetch forecast
        base_url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            'q': city,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'lang': 'vi'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Process forecast (group by day)
        daily_forecasts = {}
        for item in data['list'][:40]:  # 5 days * 8 intervals
            date_str = item['dt_txt'].split(' ')[0]
            if date_str not in daily_forecasts:
                daily_forecasts[date_str] = []
            daily_forecasts[date_str].append(item)
        
        # Aggregate daily data
        forecast_list = []
        for date_str, intervals in list(daily_forecasts.items())[:days]:
            temps = [i['main']['temp'] for i in intervals]
            humidities = [i['main']['humidity'] for i in intervals]
            
            forecast_list.append({
                'date': date_str,
                'day_name': datetime.strptime(date_str, '%Y-%m-%d').strftime('%A'),
                'temp_min': min(temps),
                'temp_max': max(temps),
                'temp_avg': sum(temps) / len(temps),
                'humidity_avg': sum(humidities) / len(humidities),
                'description': intervals[len(intervals)//2]['weather'][0]['description'],
                'icon': intervals[len(intervals)//2]['weather'][0]['icon'],
                'rain_total': sum([i.get('rain', {}).get('3h', 0) for i in intervals])
            })
        
        forecast_data = {
            'city': data['city']['name'],
            'country': data['city']['country'],
            'forecasts': forecast_list
        }
        
        # Cache
        cache_weather(cache_key, forecast_data)
        
        return {'success': True, 'forecast': forecast_data, 'cached': False}
    
    except Exception as e:
        app.logger.error(f'Forecast API error: {e}')
        return {'success': False, 'error': str(e)}, 500


# ==================== Wallet API Routes ====================

@app.route('/api/wallet', methods=['GET'])
@login_required
def api_get_wallet():
    """Get user wallet info"""
    try:
        user_id = session.get('user_id')
        wallet = get_user_wallet(user_id)
        transactions = load_wallet_transactions(user_id)[:20]  # Last 20 transactions
        
        return {
            'ok': True,
            'wallet': wallet,
            'transactions': transactions
        }
    except Exception as e:
        app.logger.error(f'Error getting wallet: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/wallet/topup', methods=['GET', 'POST'])
@login_required
def api_wallet_topup():
    """Demo top-up: credit wallet immediately (supports GET from profile + POST API)."""
    try:
        user_id = session.get('user_id')
        if request.method == 'GET':
            amount_raw = request.args.get('amount', '0')
            payment_method = request.args.get('method', 'vnpay').lower()  # vnpay or momo
        else:
            data = request.get_json(silent=True) or {}
            amount_raw = data.get('amount', 0)
            payment_method = data.get('payment_method', 'vnpay').lower()  # vnpay or momo

        try:
            amount = int(float(amount_raw))
        except (TypeError, ValueError):
            amount = 0
        
        if amount < 10000:
            if request.method == 'GET':
                flash('Số tiền nạp tối thiểu 10,000đ', 'error')
                return redirect(url_for('profile'))
            return {'ok': False, 'error': 'Số tiền nạp tối thiểu 10,000đ'}, 400
        
        if amount > 50000000:
            if request.method == 'GET':
                flash('Số tiền nạp tối đa 50,000,000đ', 'error')
                return redirect(url_for('profile'))
            return {'ok': False, 'error': 'Số tiền nạp tối đa 50,000,000đ'}, 400

        if payment_method not in {'vnpay', 'momo'}:
            payment_method = 'vnpay'

        # Demo mode: credit wallet immediately, no external gateway redirect
        tx = add_wallet_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type='topup',
            description=f'Nạp tiền demo qua {payment_method.upper()}'
        )

        wallet = get_user_wallet(user_id)

        try:
            create_notification(
                user_id=user_id,
                title='💰 Nạp tiền thành công (Demo)',
                message=f'Đã nạp {amount:,}đ vào ví. Số dư: {wallet.get("balance", 0):,.0f}đ',
                notification_type='success',
                link='/profile'
            )
        except Exception as notify_error:
            app.logger.warning(f'Cannot create topup notification: {notify_error}')

        app.logger.info(f'Demo topup success: user={user_id}, amount={amount}, method={payment_method}')

        if request.method == 'GET':
            flash(f'Nạp tiền demo thành công! +{amount:,}đ vào ví', 'success')
            return redirect(url_for('profile'))

        return {
            'ok': True,
            'message': 'Nạp tiền demo thành công',
            'transaction_id': tx.get('id'),
            'wallet_balance': wallet.get('balance', 0)
        }
    
    except Exception as e:
        app.logger.error(f'Error creating topup: {e}')
        if request.method == 'GET':
            flash('Có lỗi khi nạp tiền demo', 'error')
            return redirect(url_for('profile'))
        return {'ok': False, 'error': str(e)}, 500


@app.route('/wallet/topup/callback', methods=['GET'])
def wallet_topup_callback():
    """Handle wallet top-up payment callback"""
    try:
        # Similar to payment callback, but update wallet instead
        payment_method = request.args.get('payment_method', 'vnpay')
        
        if payment_method == 'vnpay' and vnpay_payment:
            is_valid = vnpay_payment.validate_response(dict(request.args))
            transaction_id = request.args.get('vnp_TxnRef')
            response_code = request.args.get('vnp_ResponseCode')
        elif payment_method == 'momo' and momo_payment:
            is_valid = True  # MoMo has separate signature check
            transaction_id = request.args.get('orderId')
            response_code = request.args.get('resultCode')
        else:
            flash('Phương thức thanh toán không hợp lệ', 'error')
            return redirect(url_for('profile'))
        
        # Load transaction
        transactions = []
        transaction_file = BASE_DIR / 'data' / 'wallet_transactions.jsonl'
        target_tx = None
        
        if transaction_file.exists():
            with open(transaction_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tx = json.loads(line)
                        transactions.append(tx)
                        if tx['id'] == transaction_id:
                            target_tx = tx
        
        if not target_tx:
            flash('Giao dịch không tồn tại', 'error')
            return redirect(url_for('profile'))
        
        # Check payment result
        if (payment_method == 'vnpay' and response_code == '00') or \
           (payment_method == 'momo' and response_code == '0'):
            # Success - add money to wallet
            user_id = target_tx['user_id']
            amount = target_tx['amount']
            
            # Update transaction status
            for tx in transactions:
                if tx['id'] == transaction_id:
                    tx['status'] = 'completed'
                    tx['transaction_type'] = 'topup'
                    tx['completed_at'] = datetime.utcnow().isoformat()
            
            # Save transactions
            with open(transaction_file, 'w', encoding='utf-8') as f:
                for tx in transactions:
                    f.write(json.dumps(tx, ensure_ascii=False) + '\n')
            
            # Credit wallet
            wallet = get_user_wallet(user_id)
            wallet['balance'] = float(wallet.get('balance', 0)) + float(amount)
            wallet['updated_at'] = datetime.utcnow().isoformat()
            save_wallet(wallet)
            
            # Notify
            create_notification(
                user_id=user_id,
                title='💰 Nạp tiền thành công',
                message=f'Đã nạp {amount:,}đ vào ví. Số dư: {wallet["balance"]:,}đ',
                notification_type='success',
                link='/profile'
            )
            
            flash(f'Nạp tiền thành công! Số dư mới: {wallet["balance"]:,}đ', 'success')
        else:
            flash('Nạp tiền thất bại hoặc bị hủy', 'error')
        
        return redirect(url_for('profile'))
    
    except Exception as e:
        app.logger.error(f'Topup callback error: {e}')
        flash('Có lỗi xảy ra khi xử lý giao dịch', 'error')
        return redirect(url_for('profile'))


# ==================== Referral API Routes ====================

@app.route('/api/referral/info', methods=['GET'])
@login_required
def api_referral_info():
    """Get user's referral information"""
    try:
        user_id = session.get('user_id')
        referral_info = get_user_referral_info(user_id)
        leaderboard = get_referral_leaderboard(10)
        
        return {
            'ok': True,
            'referral': referral_info,
            'leaderboard': leaderboard
        }
    except Exception as e:
        app.logger.error(f'Error getting referral info: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/referral/apply', methods=['POST'])
@login_required
def api_apply_referral():
    """Apply referral code (for new users)"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        referral_code = data.get('code', '').strip().upper()
        
        if not referral_code:
            return {'ok': False, 'error': 'Mã giới thiệu không hợp lệ'}, 400
        
        # Check if user already used a referral code
        user_ref_info = get_user_referral_info(user_id)
        if user_ref_info.get('referred_by'):
            return {'ok': False, 'error': 'Bạn đã sử dụng mã giới thiệu rồi'}, 400
        
        success = apply_referral_code(user_id, referral_code)
        
        if success:
            # Bonus for new user
            add_points(user_id, 50, 'referral_welcome', 'Thưởng từ mã giới thiệu')
            
            return {'ok': True, 'message': 'Áp dụng mã giới thiệu thành công! +50 điểm'}
        else:
            return {'ok': False, 'error': 'Mã giới thiệu không hợp lệ'}, 400
    
    except Exception as e:
        app.logger.error(f'Error applying referral: {e}')
        return {'ok': False, 'error': str(e)}, 500


# ==================== Achievement & Quest API Routes ====================

@app.route('/api/achievements', methods=['GET'])
@login_required
def api_get_achievements():
    """Get user's achievements and available achievements"""
    try:
        user_id = session.get('user_id')
        users = load_users()
        user = users.get(user_id, {})
        
        unlocked = user.get('achievements', [])
        
        # Prepare achievement list
        achievements_list = []
        for ach_id, ach_data in ACHIEVEMENTS.items():
            achievements_list.append({
                'id': ach_id,
                'name': ach_data['name'],
                'description': ach_data['desc'],
                'points': ach_data['points'],
                'unlocked': ach_id in unlocked
            })
        
        # Sort: unlocked first, then by points
        achievements_list.sort(key=lambda x: (not x['unlocked'], -x['points']))
        
        return {
            'ok': True,
            'achievements': achievements_list,
            'total_unlocked': len(unlocked),
            'total_available': len(ACHIEVEMENTS)
        }
    
    except Exception as e:
        app.logger.error(f'Error getting achievements: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/quests/daily', methods=['GET'])
@login_required
def api_daily_quests():
    """Get daily quests for user"""
    try:
        user_id = session.get('user_id')
        quests = get_user_daily_quests(user_id)
        
        completed_count = len([q for q in quests if q['completed']])
        total_rewards = sum([q['reward'] for q in quests if q['completed']])
        
        return {
            'ok': True,
            'quests': quests,
            'completed_count': completed_count,
            'total_count': len(quests),
            'total_rewards': total_rewards
        }
    
    except Exception as e:
        app.logger.error(f'Error getting daily quests: {e}')
        return {'ok': False, 'error': str(e)}, 500


# ==================== Farmer Dashboard API Routes ====================

@app.route('/api/dashboard/farmer', methods=['GET'])
@login_required
def api_farmer_dashboard():
    """Get comprehensive farmer dashboard data"""
    try:
        user_id = session.get('user_id')
        dashboard_data = get_farmer_dashboard_data(user_id)
        
        if not dashboard_data:
            return {'ok': False, 'error': 'Không thể tải dữ liệu dashboard'}, 500
        
        return {'ok': True, 'dashboard': dashboard_data}
    
    except Exception as e:
        app.logger.error(f'Error getting dashboard: {e}')
        return {'ok': False, 'error': str(e)}, 500


@app.route('/dashboard')
@login_required
def farmer_dashboard_page():
    """Farmer dashboard page"""
    try:
        user_id = session.get('user_id')
        users = load_users()
        user = users.get(user_id, {})
        
        # Get translations
        lang = session.get('lang', 'vi')
        t = load_translations().get(lang, {})
        
        return render_template(
            'farmer_dashboard.html',
            user=user,
            t=t
        )
    
    except Exception as e:
        app.logger.error(f'Dashboard page error: {e}')
        flash('Có lỗi xảy ra khi tải dashboard', 'error')
        return redirect(url_for('index'))


@app.route('/api/care/schedule', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_care_schedule():
    """Manage care schedule (calendar)"""
    try:
        user_id = session.get('user_id')
        
        if request.method == 'GET':
            schedules = load_care_schedule(user_id)
            upcoming = get_upcoming_care_tasks(user_id, days_ahead=7)
            
            return {
                'ok': True,
                'schedules': schedules,
                'upcoming': upcoming
            }
        
        elif request.method == 'POST':
            data = request.get_json()
            
            schedule = {
                'id': data.get('id', str(uuid4())),
                'user_id': user_id,
                'title': data.get('title', ''),
                'description': data.get('description', ''),
                'task_type': data.get('task_type', 'other'),  # water, fertilize, spray, prune, other
                'frequency': data.get('frequency', 'once'),  # once, daily, weekly, monthly
                'next_due': data.get('next_due'),
                'active': True,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            save_care_schedule(schedule)
            
            return {'ok': True, 'schedule': schedule}
        
        elif request.method == 'DELETE':
            schedule_id = request.args.get('id')
            if not schedule_id:
                return {'ok': False, 'error': 'Missing schedule ID'}, 400
            
            # Load and filter schedules
            schedules = load_care_schedule(user_id)
            schedules = [s for s in schedules if s['id'] != schedule_id or s['user_id'] != user_id]
            
            # Save filtered list
            schedule_file = BASE_DIR / 'data' / 'care_schedules.jsonl'
            with open(schedule_file, 'w', encoding='utf-8') as f:
                for s in schedules:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
            
            return {'ok': True, 'message': 'Đã xóa lịch chăm sóc'}
    
    except Exception as e:
        app.logger.error(f'Error managing care schedule: {e}')
        return {'ok': False, 'error': str(e)}, 500


# ==================== Weather API Integration ====================

def fetch_weather_from_api(city=None, lat=None, lon=None):
    """
    Fetch current weather data from OpenWeatherMap API
    
    Args:
        city: City name (e.g., "Hanoi", "Ho Chi Minh City")
        lat: Latitude
        lon: Longitude
    
    Returns:
        dict: Weather data or None if failed
    """
    if not OPENWEATHER_API_KEY:
        return None
    
    try:
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',  # Celsius
            'lang': 'vi'  # Vietnamese descriptions
        }
        
        if lat and lon:
            params['lat'] = lat
            params['lon'] = lon
        elif city:
            params['q'] = city
        else:
            params['q'] = 'Hanoi'  # Default
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Format response
        weather_data = {
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'] * 3.6,  # m/s to km/h
            'description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon'],
            'city_name': data['name'],
            'country': data['sys'].get('country', ''),
            'clouds': data.get('clouds', {}).get('all', 0),
            'rain': data.get('rain', {}).get('1h', 0),  # Rain in last hour
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return weather_data
    
    except requests.exceptions.RequestException as e:
        app.logger.error(f'Weather API request failed: {e}')
        return None
    except Exception as e:
        app.logger.error(f'Weather parsing error: {e}')
        return None


def assess_disease_risk_from_weather(weather_data):
    """
    Assess tomato disease risk based on weather conditions
    
    Risk factors:
    - High humidity (>80%) + Moderate temp (20-30°C) = High risk for fungal diseases
    - Heavy rain + Wind = High risk for disease spread
    - Very hot (>32°C) + High humidity = High risk for bacterial diseases
    
    Returns:
        dict: {risk_level: 'low'|'medium'|'high', risk_text: str, risk_color: str, recommendations: list}
    """
    if not weather_data:
        return {
            'risk_level': 'unknown',
            'risk_text': 'Không xác định',
            'risk_color': 'gray',
            'recommendations': []
        }
    
    temp = weather_data['temperature']
    humidity = weather_data['humidity']
    rain = weather_data.get('rain', 0)
    wind = weather_data['wind_speed']
    
    risk_score = 0
    recommendations = []
    
    # Temperature assessment
    if 20 <= temp <= 30:
        risk_score += 2
    elif temp > 32:
        risk_score += 1
        recommendations.append('🌡️ Nhiệt độ cao - Tăng tưới nước, che bóng mát cho cây')
    elif temp < 15:
        recommendations.append('❄️ Nhiệt độ thấp - Che phủ cây vào ban đêm')
    
    # Humidity assessment
    if humidity > 80:
        risk_score += 3
        recommendations.append('💧 Độ ẩm rất cao - Nguy cơ bệnh nấm! Tăng cường thông gió')
    elif humidity > 70:
        risk_score += 2
        recommendations.append('💦 Độ ẩm cao - Theo dõi dấu hiệu bệnh, cân nhắc phun phòng ngừa')
    elif humidity < 40:
        recommendations.append('🏜️ Độ ẩm thấp - Tăng tần suất tưới nước')
    
    # Rain assessment
    if rain > 5:
        risk_score += 2
        recommendations.append('🌧️ Mưa nhiều - Đảm bảo thoát nước tốt, tránh úng rễ')
    elif rain > 0:
        risk_score += 1
    
    # Wind assessment
    if wind > 30:
        risk_score += 1
        recommendations.append('💨 Gió mạnh - Kiểm tra giàn đỡ, tránh cây bị gãy')
    
    # Combined high-risk conditions
    if humidity > 80 and 20 <= temp <= 30:
        risk_score += 2
        recommendations.append('⚠️ Điều kiện thuận lợi cho bệnh nấm! Xem xét phun thuốc ngay')
    
    if temp > 32 and humidity > 70:
        risk_score += 2
        recommendations.append('🦠 Nguy cơ bệnh vi khuẩn! Kiểm tra lá vàng, héo')
    
    # Determine risk level
    if risk_score >= 6:
        risk_level = 'high'
        risk_text = 'Cao'
        risk_color = 'red'
        if not any('phun thuốc' in r for r in recommendations):
            recommendations.append('🛡️ Xem xét phun thuốc phòng bệnh')
    elif risk_score >= 3:
        risk_level = 'medium'
        risk_text = 'Trung bình'
        risk_color = 'orange'
        if not recommendations:
            recommendations.append('👀 Theo dõi cây thường xuyên, kiểm tra dấu hiệu bệnh')
    else:
        risk_level = 'low'
        risk_text = 'Thấp'
        risk_color = 'green'
        if not recommendations:
            recommendations.append('✅ Điều kiện tốt cho cây trồng')
            recommendations.append('🌱 Duy trì chế độ chăm sóc hiện tại')
    
    return {
        'risk_level': risk_level,
        'risk_text': risk_text,
        'risk_color': risk_color,
        'risk_score': risk_score,
        'recommendations': recommendations
    }


def get_cached_weather(cache_key):
    """Get weather from cache if not expired"""
    if cache_key in WEATHER_CACHE:
        cached_data, timestamp = WEATHER_CACHE[cache_key]
        age = (datetime.utcnow() - timestamp).total_seconds()
        if age < WEATHER_CACHE_TTL:
            return cached_data
    return None


def cache_weather(cache_key, data):
    """Cache weather data"""
    WEATHER_CACHE[cache_key] = (data, datetime.utcnow())


# ==================== Wallet System ====================

def load_wallets():
    """Load all wallet data"""
    wallets = {}
    wallet_file = BASE_DIR / 'data' / 'wallets.jsonl'
    
    if wallet_file.exists():
        with open(wallet_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        wallet = json.loads(line)
                        wallets[wallet['user_id']] = wallet
                    except json.JSONDecodeError:
                        continue
    
    return wallets


def save_wallet(wallet):
    """Save updated wallet"""
    wallet_file = BASE_DIR / 'data' / 'wallets.jsonl'
    wallets = load_wallets()
    wallets[wallet['user_id']] = wallet
    
    with open(wallet_file, 'w', encoding='utf-8') as f:
        for w in wallets.values():
            f.write(json.dumps(w, ensure_ascii=False) + '\n')


def get_user_wallet(user_id):
    """Get or create user wallet"""
    wallets = load_wallets()
    
    if user_id not in wallets:
        wallet = {
            'user_id': user_id,
            'balance': 0.0,
            'currency': 'VND',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        save_wallet(wallet)
        return wallet
    
    return wallets[user_id]


def add_wallet_transaction(user_id, amount, transaction_type, description, reference_id=None):
    """
    Add transaction to wallet history
    
    Args:
        user_id: User ID
        amount: Amount (positive for credit, negative for debit)
        transaction_type: Type (topup, purchase, refund, referral_bonus, etc.)
        description: Description text
        reference_id: Optional reference (order_id, payment_id, etc.)
    
    Returns:
        dict: Transaction record
    """
    transaction_file = BASE_DIR / 'data' / 'wallet_transactions.jsonl'
    
    transaction = {
        'id': str(uuid4()),
        'user_id': user_id,
        'amount': float(amount),
        'transaction_type': transaction_type,
        'description': description,
        'reference_id': reference_id,
        'timestamp': datetime.utcnow().isoformat(),
        'balance_after': 0.0  # Will be updated
    }
    
    # Update wallet balance
    wallet = get_user_wallet(user_id)
    wallet['balance'] = float(wallet.get('balance', 0)) + float(amount)
    wallet['updated_at'] = datetime.utcnow().isoformat()
    transaction['balance_after'] = wallet['balance']
    save_wallet(wallet)
    
    # Save transaction
    with open(transaction_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(transaction, ensure_ascii=False) + '\n')
    
    return transaction


def load_wallet_transactions(user_id=None):
    """Load wallet transactions for a user or all transactions"""
    transactions = []
    transaction_file = BASE_DIR / 'data' / 'wallet_transactions.jsonl'
    
    if transaction_file.exists():
        with open(transaction_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        tx = json.loads(line)
                        if user_id is None or tx['user_id'] == user_id:
                            transactions.append(tx)
                    except json.JSONDecodeError:
                        continue
    
    return sorted(transactions, key=lambda x: x['timestamp'], reverse=True)


# ==================== Referral Program ====================

def generate_referral_code(user_id):
    """Generate unique referral code for user"""
    # Use user_id hash for uniqueness
    hash_val = hashlib.md5(f"{user_id}{datetime.utcnow()}".encode()).hexdigest()[:8]
    return f"REF{hash_val.upper()}"


def get_user_referral_info(user_id):
    """Get or create user referral info"""
    referral_file = BASE_DIR / 'data' / 'referrals.jsonl'
    referrals = {}
    
    if referral_file.exists():
        with open(referral_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        ref = json.loads(line)
                        referrals[ref['user_id']] = ref
                    except json.JSONDecodeError:
                        continue
    
    if user_id not in referrals:
        referral_code = generate_referral_code(user_id)
        # Ensure code is unique
        existing_codes = {r['code'] for r in referrals.values()}
        while referral_code in existing_codes:
            referral_code = generate_referral_code(user_id)
        
        referrals[user_id] = {
            'user_id': user_id,
            'code': referral_code,
            'referred_by': None,
            'referral_count': 0,
            'total_earned': 0,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Save all referrals
        with open(referral_file, 'w', encoding='utf-8') as f:
            for ref in referrals.values():
                f.write(json.dumps(ref, ensure_ascii=False) + '\n')
    
    return referrals[user_id]


def apply_referral_code(user_id, referral_code):
    """
    Apply referral code when new user registers
    
    Returns:
        bool: Success status
    """
    if not referral_code:
        return False
    
    referral_file = BASE_DIR / 'data' / 'referrals.jsonl'
    referrals = {}
    
    if referral_file.exists():
        with open(referral_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        ref = json.loads(line)
                        referrals[ref['user_id']] = ref
                    except json.JSONDecodeError:
                        continue
    
    # Find referrer
    referrer_id = None
    for uid, ref in referrals.items():
        if ref['code'] == referral_code.upper():
            referrer_id = uid
            break
    
    if not referrer_id or referrer_id == user_id:
        return False
    
    # Update referrer stats
    referrals[referrer_id]['referral_count'] = referrals[referrer_id].get('referral_count', 0) + 1
    
    # Update new user's referred_by
    if user_id in referrals:
        referrals[user_id]['referred_by'] = referrer_id
    
    # Save
    with open(referral_file, 'w', encoding='utf-8') as f:
        for ref in referrals.values():
            f.write(json.dumps(ref, ensure_ascii=False) + '\n')
    
    # Reward referrer with bonus points and wallet credit
    reward_points = 100
    reward_money = 50000  # 50k VND
    
    try:
        # Add points
        add_points(
            referrer_id,
            reward_points,
            'referral_bonus',
            f'Thưởng giới thiệu bạn mới'
        )
        
        # Add wallet credit
        add_wallet_transaction(
            referrer_id,
            reward_money,
            'referral_bonus',
            f'Thưởng giới thiệu người dùng mới',
            reference_id=user_id
        )
        
        # Update referral stats
        referrals[referrer_id]['total_earned'] = referrals[referrer_id].get('total_earned', 0) + reward_money
        with open(referral_file, 'w', encoding='utf-8') as f:
            for ref in referrals.values():
                f.write(json.dumps(ref, ensure_ascii=False) + '\n')
        
        # Notify referrer
        create_notification(
            user_id=referrer_id,
            title='🎉 Thưởng giới thiệu!',
            message=f'Bạn nhận {reward_points} điểm + {reward_money:,}đ từ việc giới thiệu bạn mới',
            notification_type='success',
            link='/profile'
        )
        
        # Notify new user
        create_notification(
            user_id=user_id,
            title='🎁 Chào mừng!',
            message=f'Bạn đã nhận 50 điểm từ mã giới thiệu',
            notification_type='success',
            link='/profile'
        )
        
    except Exception as e:
        app.logger.error(f'Error rewarding referral: {e}')
    
    return True


def get_referral_leaderboard(limit=10):
    """Get top referrers"""
    referral_file = BASE_DIR / 'data' / 'referrals.jsonl'
    referrals = []
    
    if referral_file.exists():
        with open(referral_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        referrals.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Sort by referral count
    referrals.sort(key=lambda x: x.get('referral_count', 0), reverse=True)
    
    # Get user info for top referrers
    users = load_users()
    leaderboard = []
    
    for ref in referrals[:limit]:
        if ref.get('referral_count', 0) > 0:
            user = users.get(ref['user_id'], {})
            leaderboard.append({
                'user_id': ref['user_id'],
                'username': user.get('username', 'Unknown'),
                'display_name': user.get('display_name') or user.get('username', 'Unknown'),
                'avatar': user.get('avatar'),
                'referral_count': ref.get('referral_count', 0),
                'total_earned': ref.get('total_earned', 0)
            })
    
    return leaderboard


# ==================== Enhanced Achievement System ====================

# Extended achievement definitions
ACHIEVEMENTS = {
    # Disease Detection Achievements
    'first_scan': {'name': '🔍 Lần đầu tiên', 'desc': 'Quét ảnh đầu tiên', 'points': 10},
    'scan_10': {'name': '🎯 Người mới', 'desc': 'Quét 10 ảnh', 'points': 50},
    'scan_50': {'name': '👨‍🔬 Chuyên gia tập sự', 'desc': 'Quét 50 ảnh', 'points': 200},
    'scan_100': {'name': '🏆 Chuyên gia bệnh hại', 'desc': 'Quét 100 ảnh', 'points': 500},
    'scan_streak_7': {'name': '🔥 Kiên trì 7 ngày', 'desc': 'Quét ảnh 7 ngày liên tiếp', 'points': 100},
    
    # Shopping Achievements
    'first_purchase': {'name': '🛒 Khách hàng mới', 'desc': 'Hoàn thành đơn đầu tiên', 'points': 20},
    'purchase_5': {'name': '💰 Khách quen', 'desc': 'Mua 5 đơn hàng', 'points': 100},
    'purchase_10': {'name': '👑 VIP', 'desc': 'Mua 10 đơn hàng', 'points': 300},
    'big_spender': {'name': '💎 Đại gia', 'desc': 'Tổng chi tiêu >1M', 'points': 500},
    
    # Community Achievements
    'first_post': {'name': '✍️ Bài viết đầu tiên', 'desc': 'Đăng bài đầu tiên', 'points': 15},
    'post_10': {'name': '📝 Người viết', 'desc': 'Đăng 10 bài viết', 'points': 75},
    'popular_post': {'name': '⭐ Bài hot', 'desc': 'Bài viết có >20 likes', 'points': 100},
    'first_comment': {'name': '💬 Bình luận đầu tiên', 'desc': 'Bình luận lần đầu', 'points': 10},
    'active_commenter': {'name': '🗣️ Người hay bình luận', 'desc': 'Bình luận 50 lần', 'points': 100},
    
    # Game Achievements
    'quiz_master': {'name': '🧠 Bậc thầy', 'desc': 'Hoàn thành quiz 10/10', 'points': 150},
    'memory_pro': {'name': '🎮 Trí nhớ siêu hạng', 'desc': 'Memory game <20s', 'points': 100},
    'farm_beginner': {'name': '🌱 Nông dân mới', 'desc': 'Thu hoạch 10 cây', 'points': 50},
    'farm_expert': {'name': '🚜 Nông dân chuyên nghiệp', 'desc': 'Thu hoạch 100 cây', 'points': 300},
    
    # Referral Achievements
    'first_referral': {'name': '🤝 Người giới thiệu', 'desc': 'Giới thiệu 1 người', 'points': 50},
    'referral_5': {'name': '📢 Đại sứ thương hiệu', 'desc': 'Giới thiệu 5 người', 'points': 250},
    'referral_10': {'name': '🌟 Super Referrer', 'desc': 'Giới thiệu 10 người', 'points': 600},
    
    # Loyalty Achievements
    'loyal_30': {'name': '📅 Người dùng trung thành', 'desc': 'Dùng app 30 ngày', 'points': 200},
    'loyal_90': {'name': '💪 Người dùng cống hiến', 'desc': 'Dùng app 90 ngày', 'points': 500},
    'loyal_365': {'name': '👑 Huyền thoại', 'desc': 'Dùng app 1 năm', 'points': 1000}
}


def check_and_award_achievements(user_id, event_type, event_data=None):
    """
    Check and award achievements based on user activity
    
    Args:
        user_id: User ID
        event_type: Type of event (scan, purchase, post, comment, etc.)
        event_data: Additional event data (optional)
    
    Returns:
        list: List of newly unlocked achievements
    """
    users = load_users()
    user = users.get(user_id, {})
    
    unlocked = user.get('achievements', [])
    new_achievements = []
    
    # Get user stats
    history = load_user_history(user_id)
    orders = [o for o in load_orders() if o.get('user_id') == user_id and o.get('status') == 'completed']
    posts = [p for p in load_posts() if p.get('author_id') == user_id]
    comments = [c for c in load_post_comments() if c.get('user_id') == user_id]
    referral_info = get_user_referral_info(user_id)
    
    # Check scan achievements
    scan_count = len(history)
    if scan_count >= 1 and 'first_scan' not in unlocked:
        new_achievements.append('first_scan')
    if scan_count >= 10 and 'scan_10' not in unlocked:
        new_achievements.append('scan_10')
    if scan_count >= 50 and 'scan_50' not in unlocked:
        new_achievements.append('scan_50')
    if scan_count >= 100 and 'scan_100' not in unlocked:
        new_achievements.append('scan_100')
    
    # Check purchase achievements
    purchase_count = len(orders)
    total_spent = sum(o.get('total', 0) for o in orders)
    
    if purchase_count >= 1 and 'first_purchase' not in unlocked:
        new_achievements.append('first_purchase')
    if purchase_count >= 5 and 'purchase_5' not in unlocked:
        new_achievements.append('purchase_5')
    if purchase_count >= 10 and 'purchase_10' not in unlocked:
        new_achievements.append('purchase_10')
    if total_spent >= 1000000 and 'big_spender' not in unlocked:
        new_achievements.append('big_spender')
    
    # Check community achievements
    post_count = len(posts)
    comment_count = len(comments)
    
    if post_count >= 1 and 'first_post' not in unlocked:
        new_achievements.append('first_post')
    if post_count >= 10 and 'post_10' not in unlocked:
        new_achievements.append('post_10')
    if comment_count >= 1 and 'first_comment' not in unlocked:
        new_achievements.append('first_comment')
    if comment_count >= 50 and 'active_commenter' not in unlocked:
        new_achievements.append('active_commenter')
    
    # Check popular post
    for post in posts:
        if post.get('likes', 0) >= 20 and 'popular_post' not in unlocked:
            new_achievements.append('popular_post')
            break
    
    # Check referral achievements
    referral_count = referral_info.get('referral_count', 0)
    if referral_count >= 1 and 'first_referral' not in unlocked:
        new_achievements.append('first_referral')
    if referral_count >= 5 and 'referral_5' not in unlocked:
        new_achievements.append('referral_5')
    if referral_count >= 10 and 'referral_10' not in unlocked:
        new_achievements.append('referral_10')
    
    # Check loyalty achievements
    user_created = datetime.fromisoformat(user.get('created_at', datetime.utcnow().isoformat()))
    days_since_join = (datetime.utcnow() - user_created).days
    
    if days_since_join >= 30 and 'loyal_30' not in unlocked:
        new_achievements.append('loyal_30')
    if days_since_join >= 90 and 'loyal_90' not in unlocked:
        new_achievements.append('loyal_90')
    if days_since_join >= 365 and 'loyal_365' not in unlocked:
        new_achievements.append('loyal_365')
    
    # Award new achievements
    if new_achievements:
        user['achievements'] = unlocked + new_achievements
        users[user_id] = user
        save_users(users)
        
        # Award points and notifications
        for ach in new_achievements:
            ach_data = ACHIEVEMENTS.get(ach, {})
            points = ach_data.get('points', 0)
            
            if points > 0:
                add_points(user_id, points, 'achievement', f'Mở khóa: {ach_data.get("name", ach)}')
            
            create_notification(
                user_id=user_id,
                title=f'🏆 Achievement Unlocked!',
                message=f'{ach_data.get("name", ach)} - {ach_data.get("desc", "")} (+{points} điểm)',
                notification_type='success',
                link='/profile'
            )
    
    return new_achievements


# ==================== Daily Quests System ====================

DAILY_QUESTS = [
    {'id': 'daily_scan_3', 'title': 'Quét 3 ảnh', 'desc': 'Quét 3 ảnh cà chua hôm nay', 'target': 3, 'reward': 30, 'type': 'scan'},
    {'id': 'daily_post_1', 'title': 'Đăng bài', 'desc': 'Đăng 1 bài viết hôm nay', 'target': 1, 'reward': 25, 'type': 'post'},
    {'id': 'daily_comment_2', 'title': 'Bình luận 2 lần', 'desc': 'Bình luận 2 bài viết hôm nay', 'target': 2, 'reward': 20, 'type': 'comment'},
    {'id': 'daily_like_5', 'title': 'Like 5 bài', 'desc': 'Like 5 bài viết hôm nay', 'target': 5, 'reward': 15, 'type': 'like'},
    {'id': 'daily_play_game', 'title': 'Chơi game', 'desc': 'Chơi 1 minigame bất kỳ', 'target': 1, 'reward': 20, 'type': 'game'}
]


def load_user_quest_progress(user_id, date_str=None):
    """Load user's quest progress for a specific date"""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    
    quest_file = BASE_DIR / 'data' / 'quest_progress.jsonl'
    
    if quest_file.exists():
        with open(quest_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        progress = json.loads(line)
                        if progress['user_id'] == user_id and progress['date'] == date_str:
                            return progress
                    except json.JSONDecodeError:
                        continue
    
    # Create new progress for today
    return {
        'user_id': user_id,
        'date': date_str,
        'quests': {q['id']: {'progress': 0, 'completed': False} for q in DAILY_QUESTS},
        'updated_at': datetime.utcnow().isoformat()
    }


def update_quest_progress(user_id, quest_type, increment=1):
    """Update progress for daily quests"""
    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    progress = load_user_quest_progress(user_id, date_str)
    
    newly_completed = []
    
    # Update relevant quests
    for quest in DAILY_QUESTS:
        if quest['type'] == quest_type:
            quest_id = quest['id']
            if not progress['quests'][quest_id]['completed']:
                progress['quests'][quest_id]['progress'] += increment
                
                # Check if completed
                if progress['quests'][quest_id]['progress'] >= quest['target']:
                    progress['quests'][quest_id]['completed'] = True
                    newly_completed.append(quest)
    
    # Save progress
    progress['updated_at'] = datetime.utcnow().isoformat()
    quest_file = BASE_DIR / 'data' / 'quest_progress.jsonl'
    
    # Read all progress, update this user's today progress
    all_progress = []
    found = False
    if quest_file.exists():
        with open(quest_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        p = json.loads(line)
                        if p['user_id'] == user_id and p['date'] == date_str:
                            all_progress.append(progress)
                            found = True
                        else:
                            all_progress.append(p)
                    except json.JSONDecodeError:
                        continue
    
    if not found:
        all_progress.append(progress)
    
    with open(quest_file, 'w', encoding='utf-8') as f:
        for p in all_progress:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # Award rewards for completed quests
    for quest in newly_completed:
        add_points(user_id, quest['reward'], 'daily_quest', f'Hoàn thành: {quest["title"]}')
        create_notification(
            user_id=user_id,
            title='✅ Quest hoàn thành!',
            message=f'{quest["title"]} - +{quest["reward"]} điểm',
            notification_type='success',
            link='/game'
        )
    
    return newly_completed


def get_user_daily_quests(user_id):
    """Get user's daily quests with progress"""
    try:
        progress = load_user_quest_progress(user_id)
        
        quests_with_progress = []
        for quest in DAILY_QUESTS:
            q = quest.copy()
            # FIXED: Safely get progress data
            quest_progress = progress.get('quests', {}).get(quest['id'], {'progress': 0, 'completed': False})
            q['progress'] = quest_progress.get('progress', 0)
            q['completed'] = quest_progress.get('completed', False)
            q['percentage'] = min(100, int(q['progress'] / q['target'] * 100)) if q['target'] > 0 else 0
            quests_with_progress.append(q)
        
        return quests_with_progress
    except Exception as e:
        app.logger.error(f'Error getting daily quests: {e}')
        # Return quests with 0 progress on error
        return [{
            **quest,
            'progress': 0,
            'completed': False,
            'percentage': 0
        } for quest in DAILY_QUESTS]


# ==================== Farmer Dashboard Analytics ====================

def get_farmer_dashboard_data(user_id):
    """
    Get comprehensive dashboard data for farmer
    
    Returns:
        dict: Dashboard metrics and charts data
    """
    try:
        # Load user data
        history = load_user_history(user_id)
        farm_progress = load_farm_progress(user_id)
        
        # Overall stats
        total_scans = len(history)
        
        # Disease distribution
        disease_counts = Counter([h.get('disease', 'Unknown') for h in history])
        disease_distribution = [
            {'disease': disease, 'count': count, 'percentage': round(count/total_scans*100, 1) if total_scans > 0 else 0}
            for disease, count in disease_counts.most_common()
        ]
        
        # Scan timeline (last 30 days)
        today = datetime.utcnow()
        timeline = []
        for i in range(30):
            date = (today - timedelta(days=29-i)).strftime('%Y-%m-%d')
            scans = [h for h in history if h.get('timestamp', '').startswith(date)]
            timeline.append({
                'date': date,
                'scans': len(scans),
                'healthy': len([h for h in scans if h.get('disease') == 'Healthy']),
                'diseased': len([h for h in scans if h.get('disease') != 'Healthy'])
            })
        
        # Model performance
        model_usage = Counter([h.get('model', 'Unknown') for h in history])
        model_stats = [
            {'model': model, 'usage': count}
            for model, count in model_usage.most_common()
        ]
        
        # Severity distribution
        severity_counts = Counter([h.get('severity', 'Unknown') for h in history])
        
        # Health score (based on recent scans)
        recent_scans = [h for h in history if (datetime.utcnow() - datetime.fromisoformat(h.get('timestamp', datetime.utcnow().isoformat()))).days <= 7]
        healthy_count = len([h for h in recent_scans if h.get('disease') == 'Healthy'])
        health_score = int((healthy_count / len(recent_scans) * 100)) if recent_scans else 100
        
        # Farm game stats - FIXED: Handle farm_progress correctly (it's a dict, not a list)
        if farm_progress and isinstance(farm_progress, dict):
            plants_list = farm_progress.get('plants', [])
            total_harvested = sum([p.get('harvested', 0) for p in plants_list])
            active_plants = len([p for p in plants_list if p.get('stage', 0) < 4])
        else:
            total_harvested = 0
            active_plants = 0
        
        # Calculate streak (consecutive days with scans)
        streak = 0
        current_date = today
        while True:
            date_str = current_date.strftime('%Y-%m-%d')
            day_scans = [h for h in history if h.get('timestamp', '').startswith(date_str)]
            if day_scans:
                streak += 1
                current_date -= timedelta(days=1)
            else:
                break
        
        return {
            'overview': {
                'total_scans': total_scans,
                'health_score': health_score,
                'scan_streak': streak,
                'total_harvested': total_harvested,
                'active_plants': active_plants
            },
            'disease_distribution': disease_distribution,
            'scan_timeline': timeline,
            'model_stats': model_stats,
            'severity_distribution': [
                {'severity': severity, 'count': count}
                for severity, count in severity_counts.items()
            ],
            'recent_activities': [
                {
                    'id': h.get('id'),
                    'disease': h.get('disease'),
                    'confidence': h.get('confidence'),
                    'severity': h.get('severity'),
                    'timestamp': h.get('timestamp')
                }
                for h in sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
            ]
        }
    
    except Exception as e:
        app.logger.error(f'Error generating dashboard data: {e}')
        return None


# ==================== Care Calendar System ====================

def load_care_schedule(user_id):
    """Load user's care calendar schedule"""
    schedule_file = BASE_DIR / 'data' / 'care_schedules.jsonl'
    schedules = []
    
    if schedule_file.exists():
        with open(schedule_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        schedule = json.loads(line)
                        if schedule['user_id'] == user_id:
                            schedules.append(schedule)
                    except json.JSONDecodeError:
                        continue
    
    return sorted(schedules, key=lambda x: x.get('next_due', ''))


def save_care_schedule(schedule):
    """Save care schedule entry"""
    schedule_file = BASE_DIR / 'data' / 'care_schedules.jsonl'
    
    # Read existing schedules
    schedules = []
    if schedule_file.exists():
        with open(schedule_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        schedules.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Update or add schedule
    found = False
    for i, s in enumerate(schedules):
        if s['id'] == schedule['id']:
            schedules[i] = schedule
            found = True
            break
    
    if not found:
        schedules.append(schedule)
    
    # Save all
    with open(schedule_file, 'w', encoding='utf-8') as f:
        for s in schedules:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


def get_upcoming_care_tasks(user_id, days_ahead=7):
    """Get upcoming care tasks for the next N days"""
    schedules = load_care_schedule(user_id)
    upcoming = []
    
    now = datetime.utcnow()
    cutoff = now + timedelta(days=days_ahead)
    
    for schedule in schedules:
        if not schedule.get('active', True):
            continue
        
        next_due = datetime.fromisoformat(schedule.get('next_due', now.isoformat()))
        
        if now <= next_due <= cutoff:
            days_until = (next_due - now).days
            upcoming.append({
                **schedule,
                'days_until': days_until,
                'overdue': False
            })
        elif next_due < now:
            # Overdue task
            days_overdue = (now - next_due).days
            upcoming.append({
                **schedule,
                'days_until': -days_overdue,
                'overdue': True
            })
    
    return sorted(upcoming, key=lambda x: x['days_until'])


# ==================== Wiki Knowledge Base ====================

def load_wiki_articles():
    """Load all wiki articles"""
    articles = []
    wiki_file = BASE_DIR / 'data' / 'wiki_articles.jsonl'
    
    if wiki_file.exists():
        with open(wiki_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        articles.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return articles

def get_wiki_article_by_id(article_id):
    """Get wiki article by ID"""
    articles = load_wiki_articles()
    for article in articles:
        if article['id'] == article_id:
            return article
    return None

def get_wiki_article_by_slug(slug):
    """Get wiki article by slug"""
    articles = load_wiki_articles()
    for article in articles:
        if article['slug'] == slug:
            return article
    return None

def save_wiki_articles(articles):
    """Save all wiki articles"""
    wiki_file = BASE_DIR / 'data' / 'wiki_articles.jsonl'
    wiki_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(wiki_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

def increment_wiki_views(article_id):
    """Increment views count for a wiki article"""
    try:
        articles = load_wiki_articles()
        for article in articles:
            if article['id'] == article_id:
                article['views'] = article.get('views', 0) + 1
                break
        save_wiki_articles(articles)
    except Exception as e:
        app.logger.error(f'Error incrementing wiki views: {e}')

def update_wiki_article(updated_article):
    """Update a specific wiki article"""
    try:
        articles = load_wiki_articles()
        for i, article in enumerate(articles):
            if article['id'] == updated_article['id']:
                articles[i] = updated_article
                break
        save_wiki_articles(articles)
        return True
    except Exception as e:
        app.logger.error(f'Error updating wiki article: {e}')
        return False

def toggle_wiki_like(article_id, user_id):
    """Toggle like for a wiki article"""
    try:
        articles = load_wiki_articles()
        for article in articles:
            if article['id'] == article_id:
                if 'liked_by' not in article:
                    article['liked_by'] = []
                
                if user_id in article['liked_by']:
                    article['liked_by'].remove(user_id)
                    article['likes'] = len(article['liked_by'])
                    liked = False
                else:
                    article['liked_by'].append(user_id)
                    article['likes'] = len(article['liked_by'])
                    liked = True
                
                break
        
        save_wiki_articles(articles)
        return liked
    except Exception as e:
        app.logger.error(f'Error toggling wiki like: {e}')
        return False

def search_wiki_articles(query):
    """Search wiki articles by title, content, tags, category"""
    articles = load_wiki_articles()
    query_lower = query.lower()
    
    results = []
    for article in articles:
        score = 0
        
        # Search in title (highest weight)
        if query_lower in article.get('title', '').lower():
            score += 10
        
        # Search in content
        if query_lower in article.get('content', '').lower():
            score += 5
        
        # Search in tags
        for tag in article.get('tags', []):
            if query_lower in tag.lower():
                score += 3
        
        # Search in category
        if query_lower in article.get('category', '').lower():
            score += 2
        
        if score > 0:
            article['search_score'] = score
            results.append(article)
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['search_score'], reverse=True)
    
    return results

def get_wiki_categories():
    """Get all unique categories"""
    articles = load_wiki_articles()
    categories = set()
    for article in articles:
        if article.get('category'):
            categories.add(article['category'])
    return sorted(list(categories))

def get_wiki_tags():
    """Get all unique tags with counts as list of tuples"""
    articles = load_wiki_articles()
    tag_counts = {}
    for article in articles:
        for tag in article.get('tags', []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    # Return list of tuples (tag_name, count) sorted by count descending
    return sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

@app.route('/wiki')
def wiki_index():
    """Wiki home page"""
    try:
        articles = load_wiki_articles()
        
        # Sort by views (popular) and featured
        featured_articles = [a for a in articles if a.get('featured', False)]
        popular_articles = sorted(articles, key=lambda x: x.get('views', 0), reverse=True)[:6]
        recent_articles = sorted(articles, key=lambda x: x.get('created_at', ''), reverse=True)[:6]
        
        categories = get_wiki_categories()
        tags = get_wiki_tags()
        
        # Get search query if exists
        query = request.args.get('q', '').strip()
        category = request.args.get('category', '').strip()
        tag = request.args.get('tag', '').strip()
        
        if query:
            articles = search_wiki_articles(query)
        elif category:
            articles = [a for a in articles if a.get('category') == category]
        elif tag:
            articles = [a for a in articles if tag in a.get('tags', [])]
        
        return render_template('wiki_index.html',
                             articles=articles,
                             featured_articles=featured_articles,
                             popular_articles=popular_articles,
                             recent_articles=recent_articles,
                             categories=categories,
                             tags=tags,
                             query=query,
                             category=category,
                             tag=tag)
    except Exception as e:
        app.logger.error(f'Error loading wiki index: {e}')
        flash('Lỗi khi tải trang wiki', 'error')
        return redirect(url_for('index'))

@app.route('/wiki/<slug>')
def wiki_article(slug):
    """View a wiki article"""
    try:
        article = get_wiki_article_by_slug(slug)
        
        if not article:
            flash('Không tìm thấy bài viết', 'warning')
            return redirect(url_for('wiki_index'))
        
        # Increment views
        increment_wiki_views(article['id'])
        article['views'] = article.get('views', 0) + 1
        
        # Get user like status
        user_liked = False
        if 'user_id' in session:
            user_liked = session['user_id'] in article.get('liked_by', [])
        
        # Get related articles (same category or tags)
        all_articles = load_wiki_articles()
        related = []
        for a in all_articles:
            if a['id'] != article['id']:
                score = 0
                if a.get('category') == article.get('category'):
                    score += 5
                common_tags = set(a.get('tags', [])) & set(article.get('tags', []))
                score += len(common_tags) * 2
                if score > 0:
                    a['relevance_score'] = score
                    related.append(a)
        
        related.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        related = related[:5]
        
        # Render markdown content
        if markdown:
            try:
                article['content_html'] = markdown.markdown(article.get('content', ''))
            except Exception:
                # Fallback: simple newline to <br> conversion
                article['content_html'] = article.get('content', '').replace('\n', '<br>')
        else:
            # Fallback: simple newline to <br> conversion with paragraph breaks
            content = article.get('content', '')
            content = content.replace('\n\n', '</p><p>')
            content = content.replace('\n', '<br>')
            article['content_html'] = f'<p>{content}</p>'
        
        return render_template('wiki_article.html',
                             article=article,
                             related_articles=related,
                             user_liked=user_liked)
    except Exception as e:
        app.logger.error(f'Error loading wiki article: {e}')
        flash('Lỗi khi tải bài viết', 'error')
        return redirect(url_for('wiki_index'))

@app.route('/api/wiki/<article_id>/like', methods=['POST'])
def wiki_like(article_id):
    """Toggle like for wiki article"""
    try:
        if 'user_id' not in session:
            return {'success': False, 'message': 'Vui lòng đăng nhập'}, 401
        
        user_id = session['user_id']
        liked = toggle_wiki_like(article_id, user_id)
        
        article = get_wiki_article_by_id(article_id)
        likes = article.get('likes', 0) if article else 0
        
        return {
            'success': True,
            'liked': liked,
            'likes': likes
        }
    except Exception as e:
        app.logger.error(f'Error liking wiki article: {e}')
        return {'success': False, 'message': str(e)}, 500

@app.route('/api/wiki/<article_id>/feedback', methods=['POST'])
def wiki_feedback(article_id):
    """Record article helpfulness feedback"""
    try:
        data = request.get_json()
        helpful = data.get('helpful', False)
        
        # Update article feedback count
        article = get_wiki_article_by_id(article_id)
        if article:
            if 'helpful_count' not in article:
                article['helpful_count'] = 0
            if 'not_helpful_count' not in article:
                article['not_helpful_count'] = 0
            
            if helpful:
                article['helpful_count'] += 1
            else:
                article['not_helpful_count'] += 1
            
            update_wiki_article(article)
        
        return {'success': True}
    except Exception as e:
        app.logger.error(f'Error recording wiki feedback: {e}')
        return {'success': False, 'message': str(e)}, 500

@app.route('/admin/orders')
@requires_admin_auth
def admin_orders():
    """Admin orders management page"""
    try:
        orders = load_orders()
        
        # Sort by timestamp descending
        orders = sorted(orders, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Calculate statistics
        total_orders = len(orders)
        total_revenue = sum(o.get('total', 0) for o in orders)
        
        pending_orders = [o for o in orders if o.get('status') == 'pending']
        processing_orders = [o for o in orders if o.get('status') == 'processing']
        shipped_orders = [o for o in orders if o.get('status') == 'shipped']
        delivered_orders = [o for o in orders if o.get('status') == 'delivered']
        
        stats = {
            'total': total_orders,
            'revenue': total_revenue,
            'pending': len(pending_orders),
            'processing': len(processing_orders),
            'shipped': len(shipped_orders),
            'delivered': len(delivered_orders)
        }
        
        return render_template('admin_orders.html', orders=orders, stats=stats)
    except Exception as e:
        app.logger.error(f'Error loading orders: {e}')
        flash('Lỗi khi tải danh sách đơn hàng', 'error')
        return redirect(url_for('profile'))

@app.route('/admin/order/<order_id>')
@requires_admin_auth
def admin_order_detail(order_id):
    """Admin order detail page"""
    order = get_order_by_id(order_id)
    if not order:
        flash('Không tìm thấy đơn hàng', 'error')
        return redirect(url_for('admin_orders'))
    
    return render_template('admin_order_detail.html', order=order)

@app.route('/admin/order/<order_id>/update_status', methods=['POST'])
@requires_admin_auth
def admin_update_order_status(order_id):
    """Update order status"""
    try:
        new_status = request.form.get('status')
        admin_note = request.form.get('note', '')
        
        if update_order_status_new(order_id, new_status, admin_note):
            flash(f'Đã cập nhật trạng thái đơn hàng #{order_id}', 'success')
        else:
            flash('Không tìm thấy đơn hàng', 'error')
    except Exception as e:
        app.logger.error(f'Error updating order status: {e}')
        flash('Lỗi khi cập nhật trạng thái', 'error')
    
    return redirect(url_for('admin_order_detail', order_id=order_id))

@app.route('/admin/products')
@requires_admin_auth
def admin_products():
    """Admin products management page"""
    try:
        products = load_products()
        products = sorted(products, key=lambda x: x.get('created_at', ''), reverse=True)
        
        return render_template('admin_products.html', products=products)
    except Exception as e:
        app.logger.error(f'Error loading products: {e}')
        flash('Lỗi khi tải danh sách sản phẩm', 'error')
        return redirect(url_for('profile'))

@app.route('/admin/product/add', methods=['GET', 'POST'])
@requires_admin_auth
def admin_add_product():
    """Add new product"""
    if request.method == 'POST':
        try:
            product_data = {
                'name': request.form.get('name', '').strip(),
                'category': request.form.get('category', 'pesticide'),
                'price': int(request.form.get('price', 0)),
                'description': request.form.get('description', '').strip(),
                'image': request.form.get('image', 'images/tomato.jpg'),
                'stock': int(request.form.get('stock', 0)),
                'sold': 0,
                'rating_avg': 0,
                'rating_count': 0,
                'status': 'active'
            }
            
            save_product(product_data)
            flash('Đã thêm sản phẩm thành công', 'success')
            return redirect(url_for('admin_products'))
        except Exception as e:
            app.logger.error(f'Error adding product: {e}')
            flash('Lỗi khi thêm sản phẩm', 'error')
    
    return render_template('admin_product_form.html', product=None)

@app.route('/admin/product/<int:product_id>/edit', methods=['GET', 'POST'])
@requires_admin_auth
def admin_edit_product(product_id):
    """Edit product"""
    product = get_product_by_id(product_id)
    if not product:
        flash('Không tìm thấy sản phẩm', 'error')
        return redirect(url_for('admin_products'))
    
    if request.method == 'POST':
        try:
            product['name'] = request.form.get('name', '').strip()
            product['category'] = request.form.get('category', 'pesticide')
            product['price'] = int(request.form.get('price', 0))
            product['description'] = request.form.get('description', '').strip()
            product['image'] = request.form.get('image', 'images/tomato.jpg')
            product['stock'] = int(request.form.get('stock', 0))
            product['status'] = request.form.get('status', 'active')
            
            save_product(product)
            flash('Đã cập nhật sản phẩm thành công', 'success')
            return redirect(url_for('admin_products'))
        except Exception as e:
            app.logger.error(f'Error updating product: {e}')
            flash('Lỗi khi cập nhật sản phẩm', 'error')
    
    return render_template('admin_product_form.html', product=product)

@app.route('/admin/product/<int:product_id>/delete', methods=['POST'])
@requires_admin_auth
def admin_delete_product(product_id):
    """Delete product"""
    try:
        delete_product(product_id)
        flash('Đã xóa sản phẩm', 'success')
    except Exception as e:
        app.logger.error(f'Error deleting product: {e}')
        flash('Lỗi khi xóa sản phẩm', 'error')
    
    return redirect(url_for('admin_products'))


# ==================== Review System ====================

def load_reviews(product_id=None):
    """Load reviews"""
    reviews = []
    reviews_file = BASE_DIR / 'data' / 'reviews.jsonl'
    
    if reviews_file.exists():
        with open(reviews_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    review = json.loads(line)
                    if product_id is None or review['product_id'] == product_id:
                        reviews.append(review)
    
    return reviews

def save_review(review_data):
    """Save review"""
    reviews_file = BASE_DIR / 'data' / 'reviews.jsonl'
    
    # Generate ID
    reviews = load_reviews()
    max_id = max([r.get('id', 0) for r in reviews], default=0)
    review_data['id'] = max_id + 1
    review_data['created_at'] = datetime.utcnow().isoformat()
    review_data['status'] = 'approved'  # Auto-approve for now
    
    # Save
    with open(reviews_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(review_data, ensure_ascii=False) + '\n')
    
    # Update product rating
    product_id = review_data['product_id']
    product_reviews = load_reviews(product_id)
    avg_rating = sum(r['rating'] for r in product_reviews) / len(product_reviews)
    
    product = get_product_by_id(product_id)
    if product:
        product['rating_avg'] = round(avg_rating, 1)
        product['rating_count'] = len(product_reviews)
        save_product(product)
    
    return review_data

@app.route('/api/product/<int:product_id>/reviews', methods=['GET'])
def api_get_reviews(product_id):
    """Get reviews for product"""
    try:
        reviews = load_reviews(product_id)
        reviews = sorted(reviews, key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {'ok': True, 'reviews': reviews}
    except Exception as e:
        app.logger.error(f'Error loading reviews: {e}')
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/product/<int:product_id>/review', methods=['POST'])
@login_required
def api_add_review(product_id):
    """Add review for product with optional images"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không thể đánh giá sản phẩm'}, 403
        
        user_id = session.get('user_id')
        user = get_user_by_id(user_id)
        
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            rating = int(data.get('rating', 5))
            comment = data.get('comment', '').strip()
            images = []
        else:
            rating = int(request.form.get('rating', 5))
            comment = request.form.get('comment', '').strip()
            
            # Handle image uploads
            images = []
            uploaded_files = request.files.getlist('images')
            
            if uploaded_files:
                review_images_dir = BASE_DIR / 'static' / 'uploaded' / 'reviews'
                review_images_dir.mkdir(parents=True, exist_ok=True)
                
                for file in uploaded_files[:5]:  # Max 5 images per review
                    if file and file.filename and allowed_file(file.filename):
                        try:
                            # Generate unique filename
                            ext = file.filename.rsplit('.', 1)[1].lower()
                            filename = f"review_{user_id}_{product_id}_{uuid4().hex[:8]}.{ext}"
                            filepath = review_images_dir / filename
                            
                            # Save file
                            file.save(str(filepath))
                            
                            # Store relative URL
                            images.append(f"uploaded/reviews/{filename}")
                        except Exception as e:
                            app.logger.error(f'Error saving review image: {e}')
        
        if rating < 1 or rating > 5:
            return {'ok': False, 'error': 'Đánh giá phải từ 1-5 sao'}, 400
        
        # Check if already reviewed
        user_reviews = [r for r in load_reviews(product_id) if r['user_id'] == user_id]
        if user_reviews:
            return {'ok': False, 'error': 'Bạn đã đánh giá sản phẩm này rồi'}, 400
        
        review_data = {
            'product_id': product_id,
            'user_id': user_id,
            'user_name': user.get('full_name', 'Anonymous'),
            'rating': rating,
            'comment': comment,
            'images': images
        }
        
        save_review(review_data)
        
        # Award points for review (more points if includes images)
        try:
            points = 20 if images else 15
            add_points(user_id, points, 'review', f'Đánh giá sản phẩm #{product_id}')
            user = get_user_by_id(user_id)
            if user:
                session['user_points'] = user.get('points', 0)
        except Exception as e:
            app.logger.error(f'Failed to award review points: {e}')
        
        return {'ok': True, 'message': 'Đã thêm đánh giá thành công', 'images_count': len(images)}
    except Exception as e:
        app.logger.error(f'Error adding review: {e}')
        return {'ok': False, 'error': 'Lỗi khi thêm đánh giá'}, 500
        return {'ok': False, 'error': 'Lỗi khi thêm đánh giá'}, 500


# ==================== Multi-Language System ====================

def load_translations():
    """Load translations from JSON file"""
    translations_file = BASE_DIR / 'data' / 'translations.json'
    
    if translations_file.exists():
        with open(translations_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Default fallback
    return {'vi': {}, 'en': {}}

# Load translations into memory
TRANSLATIONS = load_translations()

def get_translation(key, lang=None):
    """Get translation for a key in specified language"""
    if lang is None:
        lang = session.get('language', 'vi')
    
    # Navigate nested keys like 'nav.home'
    keys = key.split('.')
    trans = TRANSLATIONS.get(lang, {})
    
    for k in keys:
        if isinstance(trans, dict):
            trans = trans.get(k, key)
        else:
            return key
    
    return trans if trans else key

@app.context_processor
def inject_language():
    """Inject language utilities into all templates"""
    current_lang = session.get('language', 'vi')
    
    def t(key):
        """Translation helper function"""
        return get_translation(key, current_lang)
    
    return {
        'current_lang': current_lang,
        't': t,
        'get_translation': get_translation
    }

@app.route('/set_language/<lang>')
def set_language(lang):
    """Change language"""
    if lang in ['vi', 'en']:
        session['language'] = lang
        flash(f'Ngôn ngữ đã được thay đổi / Language changed', 'success')
    
    # Redirect to referrer or home
    return redirect(request.referrer or url_for('index'))


# ==================== Loyalty Points System ====================

def load_points_history(user_id=None):
    """Load points history"""
    history = []
    points_file = BASE_DIR / 'data' / 'points_history.jsonl'
    
    if points_file.exists():
        with open(points_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if user_id is None or entry.get('user_id') == user_id:
                            history.append(entry)
                    except json.JSONDecodeError:
                        continue
    
    # Sort by created_at descending
    history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return history

def save_points_entry(entry):
    """Save a points history entry"""
    points_file = BASE_DIR / 'data' / 'points_history.jsonl'
    points_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate ID if not exists
    if 'id' not in entry:
        history = load_points_history()
        max_id = 0
        for h in history:
            if h.get('id', '').startswith('PH'):
                try:
                    num = int(h['id'][2:])
                    max_id = max(max_id, num)
                except:
                    pass
        entry['id'] = f"PH{max_id + 1:03d}"
    
    # Add timestamp if not exists
    if 'created_at' not in entry:
        entry['created_at'] = datetime.now().isoformat()
    
    with open(points_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def add_points(user_id, points, action, description):
    """Add points to user and record history"""
    try:
        # Update user points
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        if not users_file.exists():
            return False
        
        users = []
        updated = False
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    user = json.loads(line)
                    if user['id'] == user_id:
                        user['points'] = user.get('points', 0) + points
                        updated = True
                    users.append(user)
        
        if not updated:
            return False
        
        # Save updated users
        with open(users_file, 'w', encoding='utf-8') as f:
            for user in users:
                f.write(json.dumps(user, ensure_ascii=False) + '\n')
        
        # Record history
        save_points_entry({
            'user_id': user_id,
            'points': points,
            'action': action,
            'description': description
        })
        
        app.logger.info(f'Added {points} points to user {user_id} for {action}')
        return True
    except Exception as e:
        app.logger.error(f'Error adding points: {e}')
        return False

# Helper aliases and utility functions for Tier 1 features
def load_users():
    """Load all users from users.jsonl"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        if not users_file.exists():
            return []
        
        users = []
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    users.append(json.loads(line))
        return users
    except Exception as e:
        app.logger.error(f'Error loading users: {e}')
        return []

def save_users(users):
    """Save all users to users.jsonl"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        users_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(users_file, 'w', encoding='utf-8') as f:
            for user in users:
                f.write(json.dumps(user, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        app.logger.error(f'Error saving users: {e}')
        return False

def load_user_history(user_id):
    """Load prediction history for user"""
    try:
        history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
        if not history_file.exists():
            return []
        
        history = []
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get('user_id') == user_id:
                        history.append(entry)
        return sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
    except Exception as e:
        app.logger.error(f'Error loading user history: {e}')
        return []

def load_farm_progress(user_id):
    """Load farm game progress for user"""
    try:
        farm_file = BASE_DIR / 'data' / 'farm_progress.jsonl'
        if not farm_file.exists():
            return None
        
        with open(farm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    progress = json.loads(line)
                    if progress.get('user_id') == user_id:
                        return progress
        return None
    except Exception as e:
        app.logger.error(f'Error loading farm progress: {e}')
        return None


def get_leaderboard(limit=10):
    """Get top users by points"""
    try:
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        if not users_file.exists():
            return []
        
        users = []
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    user = json.loads(line)
                    users.append({
                        'user_id': user['id'],
                        'full_name': user.get('full_name', 'Anonymous'),
                        'email': user.get('email', ''),
                        'points': user.get('points', 0)
                    })
        
        # Sort by points descending
        users.sort(key=lambda x: x['points'], reverse=True)
        return users[:limit]
    except Exception as e:
        app.logger.error(f'Error getting leaderboard: {e}')
        return []

@app.route('/points')
def points_dashboard():
    """Points dashboard page"""
    if 'user_id' not in session:
        flash('Vui lòng đăng nhập để xem điểm thưởng', 'warning')
        return redirect(url_for('login'))
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))
    
    user_id = session['user_id']
    user = get_user_by_id(user_id)
    
    if not user:
        flash('Không tìm thấy thông tin người dùng', 'error')
        return redirect(url_for('index'))
    
    # Get points history
    history = load_points_history(user_id)
    
    # Get leaderboard
    leaderboard = get_leaderboard(10)
    
    # Find user rank
    user_rank = None
    for idx, entry in enumerate(leaderboard, 1):
        if entry['user_id'] == user_id:
            user_rank = idx
            break
    
    # Redemption options
    redemption_options = [
        {'id': 'coupon_5k', 'name': 'Mã giảm 5.000đ', 'points': 50, 'value': 5000},
        {'id': 'coupon_10k', 'name': 'Mã giảm 10.000đ', 'points': 100, 'value': 10000},
        {'id': 'coupon_15k', 'name': 'Mã giảm 15.000đ', 'points': 150, 'value': 15000},
        {'id': 'coupon_20k', 'name': 'Mã giảm 20.000đ', 'points': 200, 'value': 20000},
        {'id': 'coupon_30k', 'name': 'Mã giảm 30.000đ', 'points': 300, 'value': 30000},
        {'id': 'coupon_50k', 'name': 'Mã giảm 50.000đ', 'points': 500, 'value': 50000},
    ]
    
    return render_template('points.html', 
                         user=user,
                         history=history,
                         leaderboard=leaderboard,
                         user_rank=user_rank,
                         redemption_options=redemption_options)

@app.route('/api/points/redeem', methods=['POST'])
def redeem_points():
    """Redeem points for rewards"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Chưa đăng nhập'}, 401
    if session.get('is_admin'):
        return {'ok': False, 'error': 'Admin không có quyền truy cập chức năng này'}, 403
    
    try:
        data = request.get_json()
        option_id = data.get('option_id')
        
        # Redemption options mapping
        options = {
            'coupon_5k': {'points': 50, 'value': 5000},
            'coupon_10k': {'points': 100, 'value': 10000},
            'coupon_15k': {'points': 150, 'value': 15000},
            'coupon_20k': {'points': 200, 'value': 20000},
            'coupon_30k': {'points': 300, 'value': 30000},
            'coupon_50k': {'points': 500, 'value': 50000},
        }
        
        if option_id not in options:
            return {'ok': False, 'error': 'Lựa chọn không hợp lệ'}, 400
        
        user_id = session['user_id']
        user = get_user_by_id(user_id)
        
        if not user:
            return {'ok': False, 'error': 'Không tìm thấy người dùng'}, 404
        
        option = options[option_id]
        required_points = option['points']
        coupon_value = option['value']
        
        # Check if user has enough points
        if user.get('points', 0) < required_points:
            return {'ok': False, 'error': 'Không đủ điểm'}, 400
        
        # Deduct points
        success = add_points(user_id, -required_points, 'redeem_coupon', 
                           f'Đổi điểm lấy mã giảm giá {coupon_value}đ')
        
        if not success:
            return {'ok': False, 'error': 'Lỗi khi trừ điểm'}, 500
        
        # Generate coupon code
        coupon_code = f"PTS{random.randint(100000, 999999)}"
        
        # Add coupon to user
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        users = []
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    u = json.loads(line)
                    if u['id'] == user_id:
                        if 'vouchers' not in u:
                            u['vouchers'] = []
                        u['vouchers'].append({
                            'code': coupon_code,
                            'value': coupon_value,
                            'source': 'points_redemption',
                            'created_at': datetime.now().isoformat(),
                            'used': False,
                            'expires_at': (datetime.now() + timedelta(days=30)).isoformat()
                        })
                    users.append(u)
        
        with open(users_file, 'w', encoding='utf-8') as f:
            for u in users:
                f.write(json.dumps(u, ensure_ascii=False) + '\n')
        
        return {
            'ok': True, 
            'message': 'Đổi điểm thành công!',
            'coupon_code': coupon_code,
            'remaining_points': user.get('points', 0) - required_points
        }
    except Exception as e:
        app.logger.error(f'Error redeeming points: {e}')
        return {'ok': False, 'error': 'Lỗi khi đổi điểm'}, 500

@app.route('/api/points/checkin', methods=['POST'])
def daily_checkin():
    """Daily check-in to earn points"""
    if 'user_id' not in session:
        return {'ok': False, 'error': 'Chưa đăng nhập'}, 401
    
    try:
        user_id = session['user_id']
        user = get_user_by_id(user_id)
        
        if not user:
            return {'ok': False, 'error': 'Không tìm thấy người dùng'}, 404
        
        # Check if already checked in today
        last_checkin = user.get('last_checkin')
        today = datetime.now().date()
        
        if last_checkin:
            last_checkin_date = datetime.fromisoformat(last_checkin).date()
            if last_checkin_date == today:
                return {'ok': False, 'error': 'Bạn đã check-in hôm nay rồi'}, 400
        
        # Add check-in points
        checkin_points = 5
        success = add_points(user_id, checkin_points, 'daily_checkin', 
                           'Check-in hàng ngày')
        
        if not success:
            return {'ok': False, 'error': 'Lỗi khi thêm điểm'}, 500
        
        # Update last_checkin
        users_file = BASE_DIR / 'data' / 'users.jsonl'
        users = []
        with open(users_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    u = json.loads(line)
                    if u['id'] == user_id:
                        u['last_checkin'] = datetime.now().isoformat()
                    users.append(u)
        
        with open(users_file, 'w', encoding='utf-8') as f:
            for u in users:
                f.write(json.dumps(u, ensure_ascii=False) + '\n')
        
        return {
            'ok': True,
            'message': f'Check-in thành công! +{checkin_points} điểm',
            'points_earned': checkin_points,
            'total_points': user.get('points', 0) + checkin_points
        }
    except Exception as e:
        app.logger.error(f'Error during check-in: {e}')
        return {'ok': False, 'error': 'Lỗi khi check-in'}, 500

@app.route('/leaderboard')
def leaderboard():
    """Leaderboard page"""
    leaders = get_leaderboard(50)
    
    user_rank = None
    if 'user_id' in session:
        user_id = session['user_id']
        for idx, entry in enumerate(leaders, 1):
            if entry['user_id'] == user_id:
                user_rank = idx
                break
    
    return render_template('leaderboard.html', 
                         leaderboard=leaders,
                         user_rank=user_rank)


# ==================== Community & Social Network ====================

SOCIAL_STOP_WORDS = {
    'about', 'again', 'anh', 'bạn', 'benh', 'biet', 'biết', 'care', 'chia', 'chiase',
    'chăm', 'cho', 'cua', 'cũng', 'dang', 'đang', 'đây', 'dieu', 'điều', 'duoc', 'được',
    'farm', 'hien', 'hiện', 'hoi', 'hỏi', 'hom', 'hôm', 'kinh', 'kinhnghiem', 'lam', 'làm',
    'link', 'loai', 'loại', 'luon', 'luôn', 'minh', 'mình', 'mot', 'một', 'ngay', 'người',
    'nhieu', 'nhiều', 'post', 'share', 'tham', 'them', 'thêm', 'thi', 'thì', 'this', 'trai',
    'trong', 'tuoi', 'tui', 'vay', 'vậy', 'về', 'viet', 'viết', 'with'
}


def parse_iso_datetime(value):
    """Parse ISO datetime safely."""
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def extract_post_topics(content, limit=5):
    """Extract lightweight discussion topics from post text."""
    if not content:
        return []

    lowered = content.lower()
    hashtags = [tag.strip('_') for tag in re.findall(r'#([0-9A-Za-zÀ-ỹ_]+)', lowered)]
    words = re.findall(r'[0-9A-Za-zÀ-ỹ]+', lowered)

    candidates = []
    for word in words:
        if word.isdigit() or len(word) < 4 or word in SOCIAL_STOP_WORDS:
            continue
        candidates.append(word.strip('_'))

    topics = []
    seen = set()
    for topic, _ in Counter(hashtags + candidates).most_common():
        if not topic or topic in seen:
            continue
        seen.add(topic)
        topics.append(topic)
        if len(topics) >= limit:
            break

    return topics


def normalize_post(post):
    """Normalize post payload for backward compatibility."""
    normalized = dict(post)
    normalized.setdefault('likes', 0)
    normalized.setdefault('comments', 0)
    normalized.setdefault('shares', 0)
    normalized.setdefault('saves', 0)
    normalized.setdefault('user_name', 'Anonymous')
    topics = normalized.get('topics')
    if not isinstance(topics, list) or not topics:
        normalized['topics'] = extract_post_topics(normalized.get('content', ''))
    return normalized


def _rewrite_jsonl_without_matching(file_path, predicate):
    """Remove JSONL records that match predicate and return removed count."""
    if not file_path.exists():
        return 0

    kept_records = []
    removed_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if predicate(record):
                removed_count += 1
            else:
                kept_records.append(record)

    with open(file_path, 'w', encoding='utf-8') as f:
        for record in kept_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return removed_count


def load_saved_posts(user_id=None):
    """Load saved posts across users or for a single user."""
    saved_posts = []
    saved_file = BASE_DIR / 'data' / 'saved_posts.jsonl'

    if saved_file.exists():
        with open(saved_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if user_id is None or record.get('user_id') == user_id:
                    saved_posts.append(record)

    return saved_posts


def get_saved_post_ids(user_id):
    """Return saved post IDs for a user."""
    if not user_id:
        return set()
    return {record.get('post_id') for record in load_saved_posts(user_id) if record.get('post_id')}


def toggle_saved_post(post_id, user_id):
    """Toggle saved state for a post."""
    saved_file = BASE_DIR / 'data' / 'saved_posts.jsonl'
    saved_file.parent.mkdir(parents=True, exist_ok=True)

    saved_posts = load_saved_posts()
    existing_record = next(
        (
            record for record in saved_posts
            if record.get('post_id') == post_id and record.get('user_id') == user_id
        ),
        None
    )

    if existing_record:
        saved_posts.remove(existing_record)
        is_saved = False
    else:
        saved_posts.append({
            'post_id': post_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat()
        })
        is_saved = True

    with open(saved_file, 'w', encoding='utf-8') as f:
        for record in saved_posts:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    saves_count = len([record for record in saved_posts if record.get('post_id') == post_id])
    update_post(post_id, {'saves': saves_count})

    return is_saved, saves_count


def load_post_shares(post_id=None):
    """Load share events for community posts."""
    shares = []
    shares_file = BASE_DIR / 'data' / 'post_shares.jsonl'

    if shares_file.exists():
        with open(shares_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if post_id is None or record.get('post_id') == post_id:
                    shares.append(record)

    return shares


def record_post_share(post_id, user_id=None, channel='copy_link'):
    """Record a share event for analytics and counters."""
    shares_file = BASE_DIR / 'data' / 'post_shares.jsonl'
    shares_file.parent.mkdir(parents=True, exist_ok=True)

    with open(shares_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'post_id': post_id,
            'user_id': user_id,
            'channel': channel,
            'created_at': datetime.now().isoformat()
        }, ensure_ascii=False) + '\n')

    share_count = len(load_post_shares(post_id))
    update_post(post_id, {'shares': share_count})
    return share_count


def sync_user_follow_cache(follower_id, following_id, is_following):
    """Keep lightweight follow arrays in users.jsonl aligned with follow graph."""
    users = load_users()
    changed = False

    for user in users:
        if user.get('id') == follower_id:
            following = list(user.get('following', []))
            if is_following and following_id not in following:
                following.append(following_id)
                user['following'] = following
                changed = True
            if not is_following and following_id in following:
                following.remove(following_id)
                user['following'] = following
                changed = True

        if user.get('id') == following_id:
            followers = list(user.get('followers', []))
            if is_following and follower_id not in followers:
                followers.append(follower_id)
                user['followers'] = followers
                changed = True
            if not is_following and follower_id in followers:
                followers.remove(follower_id)
                user['followers'] = followers
                changed = True

    if changed:
        save_users(users)


def annotate_posts_for_user(posts, current_user_id=None, likes=None, saved_post_ids=None):
    """Attach per-user state flags to posts."""
    if not current_user_id:
        return posts

    if likes is None:
        likes = load_post_likes()
    if saved_post_ids is None:
        saved_post_ids = get_saved_post_ids(current_user_id)

    user_likes = {like.get('post_id') for like in likes if like.get('user_id') == current_user_id}

    for post in posts:
        post['liked_by_user'] = post.get('id') in user_likes
        post['saved_by_user'] = post.get('id') in saved_post_ids

    return posts


def get_user_interest_topics(user_id, posts=None):
    """Derive coarse interest topics from authored, liked and saved posts."""
    if not user_id:
        return []

    posts = posts or load_posts()
    posts_by_id = {post.get('id'): post for post in posts}
    topic_counter = Counter()

    for post in posts:
        if post.get('user_id') == user_id:
            topic_counter.update(post.get('topics', []))

    liked_post_ids = {
        like.get('post_id') for like in load_post_likes()
        if like.get('user_id') == user_id
    }
    saved_post_ids = get_saved_post_ids(user_id)

    for post_id in liked_post_ids | saved_post_ids:
        post = posts_by_id.get(post_id)
        if post:
            topic_counter.update(post.get('topics', []))

    return [topic for topic, _ in topic_counter.most_common(6)]


def score_post_for_feed(post, current_user_id=None, following_ids=None, interest_topics=None):
    """Score post for personalized feed ranking."""
    following_ids = following_ids or set()
    interest_topics = interest_topics or []

    score = (
        post.get('likes', 0) * 2.0 +
        post.get('comments', 0) * 3.0 +
        post.get('shares', 0) * 3.5 +
        post.get('saves', 0) * 2.5
    )

    created_at = parse_iso_datetime(post.get('created_at'))
    if created_at:
        age_hours = max((datetime.now() - created_at).total_seconds() / 3600, 0)
        score += max(0, 72 - age_hours) / 6

    if current_user_id and post.get('user_id') == current_user_id:
        score += 8
    if post.get('user_id') in following_ids:
        score += 20

    topic_overlap = len(set(post.get('topics', [])) & set(interest_topics))
    score += topic_overlap * 12

    return score


def get_trending_posts(posts, limit=5):
    """Return trending posts using recent engagement."""
    recent_cutoff = datetime.now() - timedelta(days=14)
    recent_posts = [
        post for post in posts
        if (parse_iso_datetime(post.get('created_at')) or datetime.min) >= recent_cutoff
    ]
    candidate_posts = recent_posts or posts
    ranked = sorted(
        candidate_posts,
        key=lambda post: (
            score_post_for_feed(post),
            parse_iso_datetime(post.get('created_at')) or datetime.min
        ),
        reverse=True
    )
    return ranked[:limit]


def build_community_feed(posts, current_user_id=None, feed_type='latest', limit=20):
    """Build feed variants for the community page."""
    normalized_posts = [normalize_post(post) for post in posts]
    following_ids = set()
    if current_user_id:
        following_ids = {
            follow.get('following_id') for follow in load_user_follows()
            if follow.get('follower_id') == current_user_id
        }

    if feed_type == 'saved':
        saved_post_ids = get_saved_post_ids(current_user_id)
        filtered = [post for post in normalized_posts if post.get('id') in saved_post_ids]
        filtered.sort(key=lambda post: parse_iso_datetime(post.get('created_at')) or datetime.min, reverse=True)
        return filtered[:limit]

    if feed_type == 'following':
        filtered = [post for post in normalized_posts if post.get('user_id') in following_ids]
        filtered.sort(key=lambda post: parse_iso_datetime(post.get('created_at')) or datetime.min, reverse=True)
        return filtered[:limit]

    if feed_type == 'trending':
        return get_trending_posts(normalized_posts, limit=limit)

    if feed_type == 'for_you' and current_user_id:
        interest_topics = get_user_interest_topics(current_user_id, posts=normalized_posts)
        ranked = sorted(
            normalized_posts,
            key=lambda post: (
                score_post_for_feed(post, current_user_id, following_ids, interest_topics),
                parse_iso_datetime(post.get('created_at')) or datetime.min
            ),
            reverse=True
        )
        return ranked[:limit]

    normalized_posts.sort(
        key=lambda post: parse_iso_datetime(post.get('created_at')) or datetime.min,
        reverse=True
    )
    return normalized_posts[:limit]


def get_suggested_users(current_user_id, limit=5):
    """Suggest users based on network and topic overlap."""
    if not current_user_id:
        return []

    users = load_users()
    follows = load_user_follows()
    posts = load_posts()

    following_ids = {
        follow.get('following_id') for follow in follows
        if follow.get('follower_id') == current_user_id
    }
    user_topics = set(get_user_interest_topics(current_user_id, posts=posts))

    suggestions = []
    for user in users:
        candidate_id = user.get('id')
        if not candidate_id or candidate_id == current_user_id or candidate_id in following_ids:
            continue

        candidate_posts = [post for post in posts if post.get('user_id') == candidate_id]
        candidate_topics = {topic for post in candidate_posts for topic in post.get('topics', [])}
        candidate_stats = get_user_stats(candidate_id)

        mutual_count = len({
            follow.get('follower_id') for follow in follows
            if follow.get('following_id') == candidate_id
        } & following_ids)
        topic_overlap = len(user_topics & candidate_topics)
        activity_score = (
            candidate_stats.get('posts_count', 0) * 3 +
            candidate_stats.get('followers_count', 0) * 2 +
            candidate_stats.get('total_likes', 0) +
            candidate_stats.get('shares_received', 0) * 2
        )
        score = mutual_count * 18 + topic_overlap * 14 + activity_score

        if score <= 0:
            continue

        if mutual_count:
            reason = f'{mutual_count} kết nối chung'
        elif topic_overlap:
            reason = f'Cùng quan tâm: {", ".join(sorted(user_topics & candidate_topics)[:2])}'
        else:
            reason = 'Tác giả đang hoạt động tốt trong cộng đồng'

        suggestions.append({
            'id': candidate_id,
            'full_name': user.get('full_name', 'Anonymous'),
            'followers_count': candidate_stats.get('followers_count', 0),
            'posts_count': candidate_stats.get('posts_count', 0),
            'mutual_count': mutual_count,
            'reason': reason,
            'score': score
        })

    suggestions.sort(key=lambda item: (item.get('score', 0), item.get('followers_count', 0)), reverse=True)
    return suggestions[:limit]


def get_social_highlights(current_user_id, posts, suggested_users=None):
    """Build compact social stats for the sidebar."""
    suggested_users = suggested_users or []
    if not current_user_id:
        return {
            'active_authors': len({post.get('user_id') for post in posts if post.get('user_id')}),
            'recent_posts': len([
                post for post in posts
                if (parse_iso_datetime(post.get('created_at')) or datetime.min) >= datetime.now() - timedelta(days=7)
            ]),
            'top_topic': get_trending_posts(posts, limit=1)[0].get('topics', ['cộng đồng'])[0] if posts else 'cộng đồng',
            'saved_posts': 0,
            'network_posts': 0,
            'suggested_users': 0
        }

    following_ids = {
        follow.get('following_id') for follow in load_user_follows()
        if follow.get('follower_id') == current_user_id
    }
    interest_topics = get_user_interest_topics(current_user_id, posts=posts)

    return {
        'active_authors': len({post.get('user_id') for post in posts if post.get('user_id')}),
        'recent_posts': len([
            post for post in posts
            if (parse_iso_datetime(post.get('created_at')) or datetime.min) >= datetime.now() - timedelta(days=7)
        ]),
        'top_topic': interest_topics[0] if interest_topics else 'cộng đồng',
        'saved_posts': len(get_saved_post_ids(current_user_id)),
        'network_posts': len([post for post in posts if post.get('user_id') in following_ids]),
        'suggested_users': len(suggested_users)
    }

def load_posts():
    """Load all posts"""
    posts = []
    posts_file = BASE_DIR / 'data' / 'posts.jsonl'
    
    if posts_file.exists():
        with open(posts_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        posts.append(normalize_post(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
    
    return posts

def save_post(post):
    """Save a new post"""
    posts_file = BASE_DIR / 'data' / 'posts.jsonl'
    posts_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate ID if not exists
    if 'id' not in post:
        posts = load_posts()
        max_id = 0
        for p in posts:
            if p.get('id', '').startswith('POST'):
                try:
                    num = int(p['id'][4:])
                    max_id = max(max_id, num)
                except:
                    pass
        post['id'] = f"POST{max_id + 1:03d}"
    
    # Add timestamps
    if 'created_at' not in post:
        post['created_at'] = datetime.now().isoformat()
    post['updated_at'] = datetime.now().isoformat()
    
    # Initialize counters
    post.setdefault('likes', 0)
    post.setdefault('comments', 0)
    post.setdefault('shares', 0)
    post.setdefault('saves', 0)
    post.setdefault('topics', extract_post_topics(post.get('content', '')))
    
    with open(posts_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(post, ensure_ascii=False) + '\n')
    
    return post

def update_post(post_id, updates):
    """Update post"""
    posts_file = BASE_DIR / 'data' / 'posts.jsonl'
    if not posts_file.exists():
        return False
    
    posts = []
    updated = False
    with open(posts_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                post = json.loads(line)
                if post['id'] == post_id:
                    post.update(updates)
                    post['updated_at'] = datetime.now().isoformat()
                    updated = True
                posts.append(post)
    
    if updated:
        with open(posts_file, 'w', encoding='utf-8') as f:
            for post in posts:
                f.write(json.dumps(post, ensure_ascii=False) + '\n')
    
    return updated

def delete_post(post_id):
    """Delete post"""
    posts_file = BASE_DIR / 'data' / 'posts.jsonl'
    if not posts_file.exists():
        return False
    
    posts = []
    deleted = False
    with open(posts_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                post = json.loads(line)
                if post['id'] == post_id:
                    deleted = True
                else:
                    posts.append(post)
    
    if deleted:
        for file_name in ('post_comments.jsonl', 'post_likes.jsonl', 'saved_posts.jsonl', 'post_shares.jsonl'):
            file_path = BASE_DIR / 'data' / file_name
            _rewrite_jsonl_without_matching(file_path, lambda record: record.get('post_id') == post_id)

        with open(posts_file, 'w', encoding='utf-8') as f:
            for post in posts:
                f.write(json.dumps(post, ensure_ascii=False) + '\n')
    
    return deleted

def load_post_comments(post_id=None):
    """Load comments for a post"""
    comments = []
    comments_file = BASE_DIR / 'data' / 'post_comments.jsonl'
    
    if comments_file.exists():
        with open(comments_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        comment = json.loads(line)
                        if post_id is None or comment.get('post_id') == post_id:
                            comments.append(comment)
                    except json.JSONDecodeError:
                        continue
    
    return comments

def save_comment(comment):
    """Save a new comment"""
    comments_file = BASE_DIR / 'data' / 'post_comments.jsonl'
    comments_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate ID if not exists
    if 'id' not in comment:
        comments = load_post_comments()
        max_id = 0
        for c in comments:
            if c.get('id', '').startswith('CMT'):
                try:
                    num = int(c['id'][3:])
                    max_id = max(max_id, num)
                except:
                    pass
        comment['id'] = f"CMT{max_id + 1:03d}"
    
    if 'created_at' not in comment:
        comment['created_at'] = datetime.now().isoformat()
    
    with open(comments_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(comment, ensure_ascii=False) + '\n')
    
    # Update post comment count
    post_id = comment.get('post_id')
    if post_id:
        comments_count = len(load_post_comments(post_id))
        update_post(post_id, {'comments': comments_count})
    
    return comment

def load_post_likes():
    """Load all post likes"""
    likes = []
    likes_file = BASE_DIR / 'data' / 'post_likes.jsonl'
    
    if likes_file.exists():
        with open(likes_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        likes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return likes

def toggle_post_like(post_id, user_id):
    """Toggle like on a post"""
    likes_file = BASE_DIR / 'data' / 'post_likes.jsonl'
    likes_file.parent.mkdir(parents=True, exist_ok=True)
    
    likes = load_post_likes()
    
    # Check if already liked
    existing_like = None
    for like in likes:
        if like.get('post_id') == post_id and like.get('user_id') == user_id:
            existing_like = like
            break
    
    if existing_like:
        # Unlike - remove the like
        likes.remove(existing_like)
        liked = False
    else:
        # Like - add new like
        likes.append({
            'post_id': post_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat()
        })
        liked = True
    
    # Save updated likes
    with open(likes_file, 'w', encoding='utf-8') as f:
        for like in likes:
            f.write(json.dumps(like, ensure_ascii=False) + '\n')
    
    # Update post like count
    likes_count = len([l for l in likes if l.get('post_id') == post_id])
    update_post(post_id, {'likes': likes_count})
    
    return liked, likes_count

def load_user_follows():
    """Load all user follows"""
    follows = []
    follows_file = BASE_DIR / 'data' / 'user_follows.jsonl'
    
    if follows_file.exists():
        with open(follows_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        follows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return follows

def toggle_follow(follower_id, following_id):
    """Toggle follow relationship"""
    follows_file = BASE_DIR / 'data' / 'user_follows.jsonl'
    follows_file.parent.mkdir(parents=True, exist_ok=True)
    
    follows = load_user_follows()
    
    # Check if already following
    existing_follow = None
    for follow in follows:
        if (follow.get('follower_id') == follower_id and 
            follow.get('following_id') == following_id):
            existing_follow = follow
            break
    
    if existing_follow:
        # Unfollow
        follows.remove(existing_follow)
        is_following = False
    else:
        # Follow
        follows.append({
            'follower_id': follower_id,
            'following_id': following_id,
            'created_at': datetime.now().isoformat()
        })
        is_following = True
    
    # Save updated follows
    with open(follows_file, 'w', encoding='utf-8') as f:
        for follow in follows:
            f.write(json.dumps(follow, ensure_ascii=False) + '\n')

    try:
        sync_user_follow_cache(follower_id, following_id, is_following)
    except Exception:
        app.logger.exception('Failed to sync user follow cache')
    
    return is_following

def get_user_stats(user_id):
    """Get user statistics for profile"""
    posts = load_posts()
    follows = load_user_follows()
    
    user_posts = [p for p in posts if p.get('user_id') == user_id]
    followers = [f for f in follows if f.get('following_id') == user_id]
    following = [f for f in follows if f.get('follower_id') == user_id]
    
    total_likes = sum(p.get('likes', 0) for p in user_posts)
    total_shares = sum(p.get('shares', 0) for p in user_posts)
    saves_received = sum(p.get('saves', 0) for p in user_posts)
    
    return {
        'posts_count': len(user_posts),
        'followers_count': len(followers),
        'following_count': len(following),
        'total_likes': total_likes,
        'total_shares': total_shares,
        'saves_received': saves_received
    }

@app.route('/community')
def community():
    """Community feed page"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    posts = load_posts()
    likes = load_post_likes()
    current_user_id = session.get('user_id')
    feed_type = request.args.get('feed', 'for_you' if current_user_id else 'latest')
    allowed_feeds = {'for_you', 'following', 'trending', 'saved', 'latest'}
    if feed_type not in allowed_feeds:
        feed_type = 'for_you' if current_user_id else 'latest'

    if not current_user_id and feed_type in {'for_you', 'saved', 'following'}:
        feed_type = 'latest'

    feed_posts = build_community_feed(posts, current_user_id=current_user_id, feed_type=feed_type, limit=20)
    saved_post_ids = get_saved_post_ids(current_user_id)
    annotate_posts_for_user(feed_posts, current_user_id, likes=likes, saved_post_ids=saved_post_ids)

    trending = get_trending_posts(posts, limit=5)
    annotate_posts_for_user(trending, current_user_id, likes=likes, saved_post_ids=saved_post_ids)

    suggested_users = []
    if current_user_id and not session.get('is_admin'):
        suggested_users = get_suggested_users(current_user_id, limit=5)

    social_highlights = get_social_highlights(current_user_id, posts, suggested_users)
    
    return render_template('community.html',
                         posts=feed_posts,
                         trending=trending,
                         feed_type=feed_type,
                         suggested_users=suggested_users,
                         social_highlights=social_highlights,
                         datetime=datetime)

@app.route('/community/post/<post_id>')
def post_detail(post_id):
    """Single post detail page"""
    posts = load_posts()
    post = next((p for p in posts if p['id'] == post_id), None)
    
    if not post:
        flash('Không tìm thấy bài viết', 'error')
        return redirect(url_for('community'))
    
    # Get comments
    comments = load_post_comments(post_id)
    comments.sort(key=lambda x: x.get('created_at', ''))
    
    # Get liked status
    current_user_id = session.get('user_id')
    likes = load_post_likes()
    saved_post_ids = get_saved_post_ids(current_user_id)
    annotate_posts_for_user([post], current_user_id, likes=likes, saved_post_ids=saved_post_ids)

    related_posts = [
        candidate for candidate in posts
        if candidate.get('id') != post_id and (
            candidate.get('user_id') == post.get('user_id') or
            set(candidate.get('topics', [])) & set(post.get('topics', []))
        )
    ]
    related_posts = sorted(
        related_posts,
        key=lambda candidate: (
            len(set(candidate.get('topics', [])) & set(post.get('topics', []))),
            score_post_for_feed(candidate)
        ),
        reverse=True
    )[:3]
    annotate_posts_for_user(related_posts, current_user_id, likes=likes, saved_post_ids=saved_post_ids)
    
    return render_template('post_detail.html',
                         post=post,
                         comments=comments,
                         related_posts=related_posts,
                         datetime=datetime)

@app.route('/api/community/post', methods=['POST'])
@login_required
def api_create_post():
    """Create new post"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không thể đăng bài'}, 403
        
        user_id = session.get('user_id')
        user = get_user_by_id(user_id)
        
        if not user:
            return {'ok': False, 'error': 'Không tìm thấy người dùng'}, 404
        
        data = request.get_json()
        content = data.get('content', '').strip()
        
        if not content:
            return {'ok': False, 'error': 'Nội dung không được để trống'}, 400
        
        if len(content) > 1000:
            return {'ok': False, 'error': 'Nội dung quá dài (tối đa 1000 ký tự)'}, 400
        
        post = {
            'user_id': user_id,
            'user_name': user.get('full_name', 'Anonymous'),
            'content': content,
            'image': data.get('image')  # URL or base64 if needed
        }
        
        saved_post = save_post(post)
        
        # Award points for creating post
        try:
            add_points(user_id, 5, 'create_post', 'Đăng bài viết cộng đồng')
            
            # Update quest progress
            update_quest_progress(user_id, 'post', increment=1)
            
            # Check achievements
            check_and_award_achievements(user_id, 'post', {'post_id': saved_post['id']})
            
            user = get_user_by_id(user_id)
            if user:
                session['user_points'] = user.get('points', 0)
        except Exception as e:
            app.logger.error(f'Failed to award post points: {e}')
        
        return {'ok': True, 'post': saved_post}
    except Exception as e:
        app.logger.error(f'Error creating post: {e}')
        return {'ok': False, 'error': 'Lỗi khi tạo bài viết'}, 500

@app.route('/api/community/post/<post_id>/comment', methods=['POST'])
@login_required
def api_add_comment(post_id):
    """Add comment to post"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không thể bình luận'}, 403
        
        user_id = session.get('user_id')
        user = get_user_by_id(user_id)
        
        if not user:
            return {'ok': False, 'error': 'Không tìm thấy người dùng'}, 404
        
        data = request.get_json()
        content = data.get('content', '').strip()
        
        if not content:
            return {'ok': False, 'error': 'Bình luận không được để trống'}, 400
        
        if len(content) > 500:
            return {'ok': False, 'error': 'Bình luận quá dài (tối đa 500 ký tự)'}, 400
        
        comment = {
            'post_id': post_id,
            'user_id': user_id,
            'user_name': user.get('full_name', 'Anonymous'),
            'content': content
        }
        
        saved_comment = save_comment(comment)
        
        # Update quest progress and check achievements
        try:
            update_quest_progress(user_id, 'comment', increment=1)
            check_and_award_achievements(user_id, 'comment', {'post_id': post_id})
        except Exception as e:
            app.logger.error(f'Failed to update quest for comment: {e}')
        
        return {'ok': True, 'comment': saved_comment}
    except Exception as e:
        app.logger.error(f'Error adding comment: {e}')
        return {'ok': False, 'error': 'Lỗi khi thêm bình luận'}, 500

@app.route('/api/community/post/<post_id>/like', methods=['POST'])
@login_required
def api_toggle_like(post_id):
    """Toggle like on post"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không thể like'}, 403
        
        user_id = session.get('user_id')
        liked, likes_count = toggle_post_like(post_id, user_id)
        
        # Update quest if liked (not unliked)
        if liked:
            try:
                update_quest_progress(user_id, 'like', increment=1)
            except Exception as e:
                app.logger.error(f'Failed to update quest for like: {e}')
        
        return {
            'ok': True,
            'liked': liked,
            'likes_count': likes_count
        }
    except Exception as e:
        app.logger.error(f'Error toggling like: {e}')
        return {'ok': False, 'error': 'Lỗi khi like bài viết'}, 500

@app.route('/api/community/post/<post_id>/save', methods=['POST'])
@login_required
def api_toggle_save_post(post_id):
    """Toggle save post for current user."""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không thể lưu bài viết'}, 403

        user_id = session.get('user_id')
        is_saved, saves_count = toggle_saved_post(post_id, user_id)
        return {
            'ok': True,
            'saved': is_saved,
            'saves_count': saves_count
        }
    except Exception as e:
        app.logger.error(f'Error saving post: {e}')
        return {'ok': False, 'error': 'Lỗi khi lưu bài viết'}, 500

@app.route('/api/community/post/<post_id>/share', methods=['POST'])
def api_share_post(post_id):
    """Record share action and return share link."""
    try:
        posts = load_posts()
        post = next((item for item in posts if item.get('id') == post_id), None)
        if not post:
            return {'ok': False, 'error': 'Không tìm thấy bài viết'}, 404

        data = request.get_json(silent=True) or {}
        channel = data.get('channel', 'copy_link')
        share_count = record_post_share(post_id, session.get('user_id'), channel=channel)

        return {
            'ok': True,
            'share_count': share_count,
            'share_url': url_for('post_detail', post_id=post_id, _external=True)
        }
    except Exception as e:
        app.logger.error(f'Error sharing post: {e}')
        return {'ok': False, 'error': 'Lỗi khi chia sẻ bài viết'}, 500

@app.route('/api/community/post/<post_id>/delete', methods=['POST'])
@login_required
def api_delete_post(post_id):
    """Delete post"""
    try:
        user_id = session.get('user_id')
        is_admin = session.get('is_admin', False)
        
        # Get post
        posts = load_posts()
        post = next((p for p in posts if p['id'] == post_id), None)
        
        if not post:
            return {'ok': False, 'error': 'Không tìm thấy bài viết'}, 404
        
        # Check permission
        if post['user_id'] != user_id and not is_admin:
            return {'ok': False, 'error': 'Bạn không có quyền xóa bài viết này'}, 403
        
        success = delete_post(post_id)
        
        if success:
            return {'ok': True, 'message': 'Đã xóa bài viết'}
        else:
            return {'ok': False, 'error': 'Không thể xóa bài viết'}, 500
    except Exception as e:
        app.logger.error(f'Error deleting post: {e}')
        return {'ok': False, 'error': 'Lỗi khi xóa bài viết'}, 500

@app.route('/user/<user_id>')
def user_profile_public(user_id):
    """Public user profile"""
    user = get_user_by_id(user_id)
    
    if not user:
        flash('Không tìm thấy người dùng', 'error')
        return redirect(url_for('community'))
    
    # Get user stats
    stats = get_user_stats(user_id)
    
    # Get user posts
    posts = load_posts()
    user_posts = [p for p in posts if p.get('user_id') == user_id]
    user_posts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Check if current user is following
    is_following = False
    current_user_id = session.get('user_id')
    if current_user_id and current_user_id != user_id:
        follows = load_user_follows()
        is_following = any(
            f.get('follower_id') == current_user_id and 
            f.get('following_id') == user_id 
            for f in follows
        )
    
    return render_template('user_profile_public.html',
                         user=user,
                         stats=stats,
                         posts=user_posts[:10],
                         is_following=is_following,
                         datetime=datetime)

@app.route('/api/user/<user_id>/follow', methods=['POST'])
@login_required
def api_toggle_follow(user_id):
    """Toggle follow user"""
    try:
        if session.get('is_admin'):
            return {'ok': False, 'error': 'Admin không thể follow'}, 403
        
        current_user_id = session.get('user_id')
        
        if current_user_id == user_id:
            return {'ok': False, 'error': 'Không thể follow chính mình'}, 400
        
        is_following = toggle_follow(current_user_id, user_id)
        target_stats = get_user_stats(user_id)
        
        return {
            'ok': True,
            'is_following': is_following,
            'followers_count': target_stats.get('followers_count', 0),
            'following_count': target_stats.get('following_count', 0)
        }
    except Exception as e:
        app.logger.error(f'Error toggling follow: {e}')
        return {'ok': False, 'error': 'Lỗi khi follow người dùng'}, 500


# ==================== DISEASE OUTBREAK MAP ROUTES ====================

@app.route('/outbreak-map')
def outbreak_map():
    """Disease outbreak map page"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))

    return render_template('outbreak_map.html')

@app.route('/api/outbreaks', methods=['GET'])
def api_get_outbreaks():
    """Get all disease outbreaks"""
    try:
        outbreaks_file = BASE_DIR / 'data' / 'disease_outbreaks.jsonl'
        outbreaks = []
        
        if outbreaks_file.exists():
            with open(outbreaks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            outbreak = json.loads(line)
                            outbreaks.append(outbreak)
                        except json.JSONDecodeError:
                            continue
        
        # Filter by status if provided
        status = request.args.get('status')
        if status:
            outbreaks = [o for o in outbreaks if o.get('status') == status]
        
        # Filter by severity if provided
        severity = request.args.get('severity')
        if severity:
            outbreaks = [o for o in outbreaks if o.get('severity') == severity]
        
        return jsonify({
            'ok': True,
            'outbreaks': outbreaks,
            'total': len(outbreaks)
        })
    
    except Exception as e:
        app.logger.error(f'Error getting outbreaks: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/outbreaks/report', methods=['POST'])
@login_required
def api_report_outbreak():
    """Report a new disease outbreak"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['disease_name', 'location', 'lat', 'lng', 'severity', 'description']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'ok': False, 'error': f'Thiếu thông tin: {field}'}), 400
        
        # Get user info
        user_email = session.get('user_email')
        user = get_user_by_email(user_email)
        reporter_name = user.get('full_name', user_email) if user else user_email
        
        # Create outbreak record
        outbreak_id = f"out{random.randint(1000, 9999)}"
        outbreak = {
            'id': outbreak_id,
            'disease_name': data['disease_name'],
            'disease_name_vn': data.get('disease_name_vn', data['disease_name']),
            'location': data['location'],
            'lat': float(data['lat']),
            'lng': float(data['lng']),
            'severity': data['severity'],  # low, medium, high
            'affected_area': data.get('affected_area', 'Chưa xác định'),
            'reported_by': reporter_name,
            'reported_date': datetime.utcnow().isoformat(),
            'description': data['description'],
            'status': 'active',  # active, monitoring, resolved
            'contact': data.get('contact', '')
        }
        
        # Save to file
        outbreaks_file = BASE_DIR / 'data' / 'disease_outbreaks.jsonl'
        with open(outbreaks_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(outbreak, ensure_ascii=False) + '\n')
        
        app.logger.info(f'New outbreak reported: {outbreak_id} by {reporter_name}')
        
        return jsonify({
            'ok': True,
            'outbreak': outbreak,
            'message': 'Đã báo cáo dịch bệnh thành công'
        })
    
    except Exception as e:
        app.logger.error(f'Error reporting outbreak: {e}')
        return jsonify({'ok': False, 'error': 'Không thể báo cáo dịch bệnh'}), 500

@app.route('/api/outbreaks/<outbreak_id>/update', methods=['POST'])
@login_required
def api_update_outbreak_status(outbreak_id):
    """Update outbreak status (admin only)"""
    try:
        if not session.get('is_admin'):
            return jsonify({'ok': False, 'error': 'Chỉ admin mới có quyền cập nhật'}), 403
        
        data = request.get_json()
        new_status = data.get('status')
        
        if new_status not in ['active', 'monitoring', 'resolved']:
            return jsonify({'ok': False, 'error': 'Trạng thái không hợp lệ'}), 400
        
        outbreaks_file = BASE_DIR / 'data' / 'disease_outbreaks.jsonl'
        updated_outbreaks = []
        found = False
        
        if outbreaks_file.exists():
            with open(outbreaks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            outbreak = json.loads(line)
                            if outbreak['id'] == outbreak_id:
                                outbreak['status'] = new_status
                                outbreak['updated_at'] = datetime.utcnow().isoformat()
                                found = True
                            updated_outbreaks.append(outbreak)
                        except json.JSONDecodeError:
                            continue
        
        if not found:
            return jsonify({'ok': False, 'error': 'Không tìm thấy dịch bệnh'}), 404
        
        # Write back
        with open(outbreaks_file, 'w', encoding='utf-8') as f:
            for outbreak in updated_outbreaks:
                f.write(json.dumps(outbreak, ensure_ascii=False) + '\n')
        
        return jsonify({
            'ok': True,
            'message': 'Đã cập nhật trạng thái'
        })
    
    except Exception as e:
        app.logger.error(f'Error updating outbreak: {e}')
        return jsonify({'ok': False, 'error': 'Không thể cập nhật'}), 500


# ==================== CROP MANAGEMENT SYSTEM ROUTES ====================

@app.route('/crop-management')
@login_required
def crop_management():
    """Crop management page"""
    if session.get('is_admin'):
        flash('Admin không có quyền truy cập chức năng này', 'error')
        return redirect(url_for('index'))
    
    return render_template('crop_management.html')

@app.route('/api/crops', methods=['GET'])
@login_required
def api_get_crops():
    """Get user's crops"""
    try:
        user_id = session.get('user_id')
        crops_file = BASE_DIR / 'data' / 'crops.jsonl'
        crops = []
        
        if crops_file.exists():
            with open(crops_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            crop = json.loads(line)
                            if crop.get('user_id') == user_id:
                                crops.append(crop)
                        except json.JSONDecodeError:
                            continue
        
        # Sort by planting_date descending
        crops.sort(key=lambda x: x.get('planting_date', ''), reverse=True)
        
        # Filter by status if provided
        status = request.args.get('status')
        if status:
            crops = [c for c in crops if c.get('status') == status]
        
        return jsonify({
            'ok': True,
            'crops': crops,
            'total': len(crops)
        })
    
    except Exception as e:
        app.logger.error(f'Error getting crops: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/crops/add', methods=['POST'])
@login_required
def api_add_crop():
    """Add new crop"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['crop_name', 'variety', 'area', 'planting_date', 'expected_harvest']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'ok': False, 'error': f'Thiếu thông tin: {field}'}), 400
        
        # Create crop record
        crop_id = f"crop{random.randint(1000, 9999)}"
        crop = {
            'id': crop_id,
            'user_id': user_id,
            'crop_name': data['crop_name'],
            'variety': data['variety'],
            'area': data['area'],
            'planting_date': data['planting_date'],
            'expected_harvest': data['expected_harvest'],
            'actual_harvest': None,
            'status': data.get('status', 'planning'),  # planning, growing, harvested
            'stage': data.get('stage', 'seedling'),  # seedling, vegetative, flowering, fruiting, completed
            'health_status': 'healthy',  # healthy, monitoring, diseased
            'yield_target': data.get('yield_target', ''),
            'yield_actual': None,
            'location': data.get('location', ''),
            'notes': data.get('notes', ''),
            'expenses': [],
            'activities': [],
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Add initial activity
        if data.get('planting_date'):
            crop['activities'].append({
                'date': data['planting_date'],
                'activity': 'Bắt đầu mùa vụ',
                'note': f"Khởi tạo mùa vụ {data['crop_name']}"
            })
        
        # Save to file
        crops_file = BASE_DIR / 'data' / 'crops.jsonl'
        with open(crops_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(crop, ensure_ascii=False) + '\n')
        
        app.logger.info(f'New crop added: {crop_id} by user {user_id}')
        
        return jsonify({
            'ok': True,
            'crop': crop,
            'message': 'Đã thêm mùa vụ mới'
        })
    
    except Exception as e:
        app.logger.error(f'Error adding crop: {e}')
        return jsonify({'ok': False, 'error': 'Không thể thêm mùa vụ'}), 500

@app.route('/api/crops/<crop_id>', methods=['GET'])
@login_required
def api_get_crop_detail(crop_id):
    """Get crop detail"""
    try:
        user_id = session.get('user_id')
        crops_file = BASE_DIR / 'data' / 'crops.jsonl'
        
        if crops_file.exists():
            with open(crops_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            crop = json.loads(line)
                            if crop['id'] == crop_id and crop['user_id'] == user_id:
                                return jsonify({'ok': True, 'crop': crop})
                        except json.JSONDecodeError:
                            continue
        
        return jsonify({'ok': False, 'error': 'Không tìm thấy mùa vụ'}), 404
    
    except Exception as e:
        app.logger.error(f'Error getting crop detail: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/crops/<crop_id>/update', methods=['POST'])
@login_required
def api_update_crop(crop_id):
    """Update crop information"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        crops_file = BASE_DIR / 'data' / 'crops.jsonl'
        updated_crops = []
        found = False
        
        if crops_file.exists():
            with open(crops_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            crop = json.loads(line)
                            if crop['id'] == crop_id and crop['user_id'] == user_id:
                                # Update fields
                                for key in ['crop_name', 'variety', 'area', 'location', 'notes', 
                                           'status', 'stage', 'health_status', 'yield_target', 
                                           'yield_actual', 'actual_harvest', 'expected_harvest']:
                                    if key in data:
                                        crop[key] = data[key]
                                crop['updated_at'] = datetime.utcnow().isoformat()
                                found = True
                            updated_crops.append(crop)
                        except json.JSONDecodeError:
                            continue
        
        if not found:
            return jsonify({'ok': False, 'error': 'Không tìm thấy mùa vụ'}), 404
        
        # Write back
        with open(crops_file, 'w', encoding='utf-8') as f:
            for crop in updated_crops:
                f.write(json.dumps(crop, ensure_ascii=False) + '\n')
        
        return jsonify({
            'ok': True,
            'message': 'Đã cập nhật thông tin mùa vụ'
        })
    
    except Exception as e:
        app.logger.error(f'Error updating crop: {e}')
        return jsonify({'ok': False, 'error': 'Không thể cập nhật'}), 500

@app.route('/api/crops/<crop_id>/activity', methods=['POST'])
@login_required
def api_add_crop_activity(crop_id):
    """Add activity to crop"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        if not data.get('activity'):
            return jsonify({'ok': False, 'error': 'Thiếu thông tin hoạt động'}), 400
        
        new_activity = {
            'date': data.get('date', datetime.utcnow().strftime('%Y-%m-%d')),
            'activity': data['activity'],
            'note': data.get('note', '')
        }
        
        crops_file = BASE_DIR / 'data' / 'crops.jsonl'
        updated_crops = []
        found = False
        
        if crops_file.exists():
            with open(crops_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            crop = json.loads(line)
                            if crop['id'] == crop_id and crop['user_id'] == user_id:
                                crop['activities'].append(new_activity)
                                crop['updated_at'] = datetime.utcnow().isoformat()
                                found = True
                            updated_crops.append(crop)
                        except json.JSONDecodeError:
                            continue
        
        if not found:
            return jsonify({'ok': False, 'error': 'Không tìm thấy mùa vụ'}), 404
        
        # Write back
        with open(crops_file, 'w', encoding='utf-8') as f:
            for crop in updated_crops:
                f.write(json.dumps(crop, ensure_ascii=False) + '\n')
        
        return jsonify({
            'ok': True,
            'activity': new_activity,
            'message': 'Đã thêm hoạt động'
        })
    
    except Exception as e:
        app.logger.error(f'Error adding activity: {e}')
        return jsonify({'ok': False, 'error': 'Không thể thêm hoạt động'}), 500

@app.route('/api/crops/<crop_id>/expense', methods=['POST'])
@login_required
def api_add_crop_expense(crop_id):
    """Add expense to crop"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        
        if not data.get('type') or not data.get('amount'):
            return jsonify({'ok': False, 'error': 'Thiếu thông tin chi phí'}), 400
        
        new_expense = {
            'date': data.get('date', datetime.utcnow().strftime('%Y-%m-%d')),
            'type': data['type'],
            'amount': float(data['amount']),
            'note': data.get('note', '')
        }
        
        crops_file = BASE_DIR / 'data' / 'crops.jsonl'
        updated_crops = []
        found = False
        
        if crops_file.exists():
            with open(crops_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            crop = json.loads(line)
                            if crop['id'] == crop_id and crop['user_id'] == user_id:
                                crop['expenses'].append(new_expense)
                                crop['updated_at'] = datetime.utcnow().isoformat()
                                found = True
                            updated_crops.append(crop)
                        except json.JSONDecodeError:
                            continue
        
        if not found:
            return jsonify({'ok': False, 'error': 'Không tìm thấy mùa vụ'}), 404
        
        # Write back
        with open(crops_file, 'w', encoding='utf-8') as f:
            for crop in updated_crops:
                f.write(json.dumps(crop, ensure_ascii=False) + '\n')
        
        return jsonify({
            'ok': True,
            'expense': new_expense,
            'message': 'Đã thêm chi phí'
        })
    
    except Exception as e:
        app.logger.error(f'Error adding expense: {e}')
        return jsonify({'ok': False, 'error': 'Không thể thêm chi phí'}), 500

@app.route('/api/crops/<crop_id>/delete', methods=['POST'])
@login_required
def api_delete_crop(crop_id):
    """Delete crop"""
    try:
        user_id = session.get('user_id')
        crops_file = BASE_DIR / 'data' / 'crops.jsonl'
        remaining_crops = []
        found = False
        
        if crops_file.exists():
            with open(crops_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            crop = json.loads(line)
                            if crop['id'] == crop_id and crop['user_id'] == user_id:
                                found = True
                                continue  # Skip this crop (delete it)
                            remaining_crops.append(crop)
                        except json.JSONDecodeError:
                            continue
        
        if not found:
            return jsonify({'ok': False, 'error': 'Không tìm thấy mùa vụ'}), 404
        
        # Write back
        with open(crops_file, 'w', encoding='utf-8') as f:
            for crop in remaining_crops:
                f.write(json.dumps(crop, ensure_ascii=False) + '\n')
        
        return jsonify({
            'ok': True,
            'message': 'Đã xóa mùa vụ'
        })
    
    except Exception as e:
        app.logger.error(f'Error deleting crop: {e}')
        return jsonify({'ok': False, 'error': 'Không thể xóa'}), 500


@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Xóa lịch sử dự đoán"""
    try:
        history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
        
        if not history_file.exists():
            flash('Đã xóa toàn bộ lịch sử dự đoán', 'success')
            return redirect(url_for('history'))
        
        # Nếu là admin, xóa toàn bộ
        if session.get('is_admin'):
            history_file.unlink()
            app.logger.info('Admin cleared all prediction history')
            flash('Đã xóa toàn bộ lịch sử dự đoán', 'success')
        else:
            # User thường chỉ xóa lịch sử của mình
            current_user_id = session.get('user_id')
            current_user_email = session.get('user_email')
            
            # Đọc tất cả và chỉ giữ lại của user khác
            remaining_entries = []
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # Giữ lại nếu không phải của user hiện tại
                        if (entry.get('user_id') != current_user_id and 
                            entry.get('user_email') != current_user_email):
                            remaining_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            # Ghi lại file
            with open(history_file, 'w', encoding='utf-8') as f:
                for entry in remaining_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            app.logger.info(f'User {current_user_email} cleared their history')
            flash('Đã xóa lịch sử của bạn', 'success')
        
        return redirect(url_for('history'))
    except Exception as e:
        app.logger.exception('Error clearing history')
        flash('Không thể xóa lịch sử', 'error')
        return redirect(url_for('history'))

@app.route('/history/<prediction_id>')
def view_prediction(prediction_id):
    """Xem chi tiết một dự đoán cụ thể"""
    try:
        history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
        
        if not history_file.exists():
            flash('Không tìm thấy lịch sử dự đoán')
            return redirect(url_for('history'))
        
        current_user_id = session.get('user_id')
        current_user_email = session.get('user_email')
        is_admin = session.get('is_admin')
        
        # Tìm prediction theo ID
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('id') == prediction_id:
                        # Kiểm tra quyền truy cập
                        if not is_admin:
                            # User thường chỉ xem được của mình
                            if (entry.get('user_id') != current_user_id and 
                                entry.get('user_email') != current_user_email and
                                (entry.get('user_id') or entry.get('user_email'))):
                                flash('Bạn không có quyền xem dự đoán này')
                                return redirect(url_for('history'))
                        
                        # Format timestamp
                        try:
                            ts = datetime.fromisoformat(entry.get('timestamp', ''))
                            entry['formatted_time'] = ts.strftime('%d/%m/%Y %H:%M:%S')
                        except:
                            entry['formatted_time'] = entry.get('timestamp', 'N/A')
                        
                        # Get disease info
                        label = entry.get('predicted_label', '')
                        disease_info = DISEASE_INFO.get(label, {
                            'name': label,
                            'definition': 'Không có thông tin',
                            'prevention': []
                        })
                        
                        return render_template('prediction_detail.html',
                                             prediction=entry,
                                             disease_info=disease_info)
                except json.JSONDecodeError:
                    continue
        
        flash('Không tìm thấy dự đoán này')
        return redirect(url_for('history'))
    except Exception as e:
        app.logger.exception('Error viewing prediction')
        flash('Không thể xem chi tiết dự đoán')
        return redirect(url_for('history'))

@app.route('/export/<prediction_id>')
def export_prediction(prediction_id):
    """Export prediction report as PDF"""
    app.logger.info(f"PDF export requested for prediction: {prediction_id}")
    
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        app.logger.info("ReportLab libraries imported successfully")
        
        # Find prediction
        history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
        if not history_file.exists():
            flash('Không tìm thấy lịch sử dự đoán', 'error')
            return redirect(url_for('history'))
        
        current_user_id = session.get('user_id')
        current_user_email = session.get('user_email')
        is_admin = session.get('is_admin')
        
        prediction = None
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('id') == prediction_id:
                        # Kiểm tra quyền truy cập
                        if not is_admin:
                            # User thường chỉ export được của mình
                            if (entry.get('user_id') != current_user_id and 
                                entry.get('user_email') != current_user_email and
                                (entry.get('user_id') or entry.get('user_email'))):
                                flash('Bạn không có quyền xuất dự đoán này', 'error')
                                return redirect(url_for('history'))
                        
                        prediction = entry
                        break
                except json.JSONDecodeError:
                    continue
        
        if not prediction:
            app.logger.warning(f"Prediction not found: {prediction_id}")
            flash('Không tìm thấy dự đoán này', 'error')
            return redirect(url_for('history'))
        
        app.logger.info(f"Found prediction: {prediction_id}, label: {prediction.get('predicted_label')}")
        
        # Get disease info
        label = prediction.get('predicted_label', '')
        disease_info = DISEASE_INFO.get(label, {
            'name': label,
            'definition': 'Không có thông tin',
            'prevention': []
        })
        
        # Register Vietnamese font - try multiple common fonts
        font_registered = False
        font_dirs = [
            r'C:\Windows\Fonts',  # Windows
            '/usr/share/fonts/truetype/dejavu',  # Linux
            '/System/Library/Fonts',  # macOS
        ]
        
        font_names = ['Arial.ttf', 'DejaVuSans.ttf', 'arial.ttf', 'arialuni.ttf']
        
        for font_dir in font_dirs:
            if not Path(font_dir).exists():
                continue
            for font_file in font_names:
                font_path = Path(font_dir) / font_file
                if font_path.exists():
                    try:
                        pdfmetrics.registerFont(TTFont('Vietnamese', str(font_path)))
                        font_registered = True
                        app.logger.info(f"Registered font: {font_path}")
                        break
                    except Exception as e:
                        app.logger.warning(f"Failed to register font {font_path}: {e}")
                        continue
            if font_registered:
                break
        
        # Use registered font or fallback to default
        base_font = 'Vietnamese' if font_registered else 'Helvetica'
        base_font_bold = 'Vietnamese' if font_registered else 'Helvetica-Bold'
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                               topMargin=20*mm, bottomMargin=20*mm)
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#2d5016'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName=base_font_bold
        )
        elements.append(Paragraph('BÁO CÁO DỰ ĐOÁN BỆNH CÀ CHUA', title_style))
        elements.append(Spacer(1, 10*mm))
        
        # Prediction info table
        try:
            ts = datetime.fromisoformat(prediction.get('timestamp', ''))
            formatted_time = ts.strftime('%d/%m/%Y %H:%M:%S')
        except:
            formatted_time = prediction.get('timestamp', 'N/A')
        
        info_data = [
            ['Mã dự đoán:', prediction.get('id', 'N/A')],
            ['Thời gian:', formatted_time],
            ['Model:', prediction.get('model_name', 'N/A')],
            ['Pipeline:', prediction.get('pipeline_key', 'N/A')],
            ['Bệnh phát hiện:', disease_info['name']],
            ['Độ tin cậy:', f"{prediction.get('probability', 0) * 100:.1f}%"],
        ]
        
        info_table = Table(info_data, colWidths=[50*mm, 100*mm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), base_font_bold),
            ('FONTNAME', (1, 0), (1, -1), base_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 10*mm))
        
        # Disease definition
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=base_font_bold,
            fontSize=14,
            textColor=colors.HexColor('#2d5016')
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontName=base_font,
            fontSize=11,
            alignment=TA_LEFT
        )
        
        elements.append(Paragraph('THÔNG TIN BỆNH:', heading_style))
        elements.append(Spacer(1, 3*mm))
        elements.append(Paragraph(disease_info['definition'], body_style))
        elements.append(Spacer(1, 5*mm))
        
        # Prevention measures
        if disease_info.get('prevention'):
            elements.append(Paragraph('BIỆN PHÁP PHÒNG NGỪA:', heading_style))
            elements.append(Spacer(1, 3*mm))
            for i, measure in enumerate(disease_info['prevention'], 1):
                elements.append(Paragraph(f"{i}. {measure}", body_style))
                elements.append(Spacer(1, 2*mm))
        
        # Add image if available
        image_path_str = prediction.get('image_path', '')
        if image_path_str:
            # Remove /static/ prefix if present
            if image_path_str.startswith('/static/'):
                image_path_str = image_path_str[8:]
            
            img_file = BASE_DIR / 'static' / image_path_str
            if img_file.exists():
                elements.append(Spacer(1, 5*mm))
                elements.append(Paragraph('ẢNH ĐÃ XỬ LÝ:', heading_style))
                elements.append(Spacer(1, 3*mm))
                try:
                    img = RLImage(str(img_file), width=100*mm, height=100*mm)
                    elements.append(img)
                except Exception as e:
                    app.logger.warning(f"Could not add image to PDF: {e}")
                    elements.append(Paragraph(f"Không thể thêm ảnh: {str(e)}", body_style))
        
        # Footer
        elements.append(Spacer(1, 10*mm))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
            fontName=base_font
        )
        elements.append(Paragraph('Báo cáo được tạo tự động bởi Tomato AI System', footer_style))
        elements.append(Paragraph(f'Ngày xuất: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', footer_style))
        
        # Build PDF
        app.logger.info(f"Building PDF with {len(elements)} elements...")
        doc.build(elements)
        buffer.seek(0)
        
        filename = f"tomato_report_{prediction_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        app.logger.info(f"PDF built successfully. Size: {buffer.getbuffer().nbytes} bytes. Sending file: {filename}")
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')
        
    except ImportError as e:
        app.logger.error(f'Required library not installed: {e}')
        flash('Chức năng xuất PDF chưa được cài đặt. Vui lòng cài reportlab: pip install reportlab', 'error')
        return redirect(url_for('history'))
    except Exception as e:
        app.logger.exception('Error exporting PDF')
        flash(f'Không thể xuất báo cáo: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route('/feedback', methods=['POST'])
def feedback():
    """Receive simple JSON feedback actions from the result page.
    Expected JSON: {"action": "not_tomato"} or {"action": "confirm"}
    Behaviour: copy the most recent uploaded image (static/uploaded/last_input.png)
    into a feedback folder AND automatically add to sample directories if appropriate.
    """
    try:
        data = request.get_json(force=True)
        action = data.get('action')
    except Exception:
        return {"ok": False, "message": "Không nhận được dữ liệu JSON."}, 400

    uploaded = BASE_DIR / 'static' / 'uploaded' / 'last_input.png'
    if not uploaded.exists():
        return {"ok": False, "message": "Không tìm thấy ảnh đã tải lên để lưu."}, 404

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    try:
        # Save to feedback folder for admin review
        fb_dir = BASE_DIR / 'static' / 'feedback'
        fb_dir.mkdir(parents=True, exist_ok=True)
        if action == 'not_tomato':
            target_dir = fb_dir / 'not_tomato'
        else:
            target_dir = fb_dir / 'confirmed_tomato'
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = f'{ts}_{action}.png'
        target_path = target_dir / target_name
        shutil.copy2(str(uploaded), str(target_path))
        
        # Note: Admin will manually review and add to samples via admin panel
        # No automatic rebuild here - admin controls when to update samples
        
        return {"ok": True, "message": f"Đã lưu feedback. Quản trị viên sẽ xem xét."}
    except Exception as e:
        app.logger.exception('Lỗi khi lưu feedback')
        return {"ok": False, "message": f"Lỗi khi lưu: {e}"}, 500


@app.route('/admin/feedback', methods=['GET'])
@requires_admin_auth
def admin_feedback():
    """Show feedback images grouped by folder for admin review."""
    fb_root = BASE_DIR / 'static' / 'feedback'
    groups = {}
    if fb_root.exists():
        for child in fb_root.iterdir():
            if child.is_dir():
                items = []
                for f in sorted(child.iterdir()):
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                        items.append({
                            'name': f.name,
                            'url': url_for('static', filename=f'feedback/{child.name}/{f.name}'),
                            'dir': child.name
                        })
                if items:
                    groups[child.name] = items
    return render_template('admin_feedback.html', images=groups)

@app.route('/admin/stats', methods=['GET'])
@requires_admin_auth
def admin_stats():
    """Statistics dashboard for admin"""
    try:
        history = _read_history_file()
        
        if not history:
            return render_template('admin_stats.html', stats=None)
        
        # Basic stats
        total_predictions = len(history)
        
        # Disease distribution
        disease_counts = Counter([h.get('predicted_label', 'Unknown') for h in history])
        disease_stats = [
            {
                'label': label,
                'name': DISEASE_INFO.get(label, {}).get('name', label),
                'count': count,
                'percentage': (count / total_predictions * 100)
            }
            for label, count in disease_counts.most_common()
        ]
        
        # Model distribution
        model_counts = Counter([h.get('model_name', 'Unknown') for h in history])
        model_stats = [
            {'name': name, 'count': count, 'percentage': (count / total_predictions * 100)}
            for name, count in model_counts.most_common()
        ]
        
        # Pipeline distribution
        pipeline_counts = Counter([h.get('pipeline_key', 'Unknown') for h in history])
        pipeline_stats = [
            {'name': name, 'count': count, 'percentage': (count / total_predictions * 100)}
            for name, count in pipeline_counts.most_common()
        ]
        
        # Confidence stats
        confidences = [h.get('probability', 0) for h in history]
        avg_confidence = np.mean(confidences) if confidences else 0
        min_confidence = np.min(confidences) if confidences else 0
        max_confidence = np.max(confidences) if confidences else 0
        
        # Warning/rejection stats
        warnings = sum(1 for h in history if h.get('possibly_not_tomato', False))
        rejections = sum(1 for h in history if h.get('rejected', False))
        
        # Time-based stats (last 30 days)
        now = datetime.utcnow()
        daily_counts = {}
        for i in range(30):
            date = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            daily_counts[date] = 0
        
        for h in history:
            ts_obj = h.get('timestamp_obj')
            if ts_obj and (now - ts_obj).days < 30:
                date_key = ts_obj.strftime('%Y-%m-%d')
                if date_key in daily_counts:
                    daily_counts[date_key] += 1
        
        # Sort by date
        daily_data = sorted(
            [{'date': k, 'count': v} for k, v in daily_counts.items()],
            key=lambda x: x['date']
        )
        
        # Recent predictions (last 10)
        recent = sorted(
            history,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:10]
        
        stats = {
            'total_predictions': total_predictions,
            'disease_stats': disease_stats,
            'model_stats': model_stats,
            'pipeline_stats': pipeline_stats,
            'avg_confidence': avg_confidence * 100,
            'min_confidence': min_confidence * 100,
            'max_confidence': max_confidence * 100,
            'warnings': warnings,
            'rejections': rejections,
            'warning_rate': (warnings / total_predictions * 100) if total_predictions > 0 else 0,
            'rejection_rate': (rejections / total_predictions * 100) if total_predictions > 0 else 0,
            'daily_data': daily_data,
            'recent_predictions': recent
        }
        
        return render_template('admin_stats.html', stats=stats)
    
    except Exception as e:
        app.logger.exception('Error loading statistics')
        flash('Không thể tải thống kê', 'error')
        return redirect(url_for('admin_feedback'))

@app.route('/admin/export_chat', methods=['GET'])
@requires_admin_auth
def admin_export_chat():
    """Export chat logs (data/chat_logs.jsonl) as CSV for Excel."""
    import csv, io, json
    logs_file = BASE_DIR / 'data' / 'chat_logs.jsonl'
    if not logs_file.exists():
        return redirect(url_for('admin_feedback'))

    # read JSONL and write CSV into memory
    mem = io.StringIO()
    writer = csv.writer(mem)
    writer.writerow(['ts', 'question', 'answer'])
    with open(logs_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                writer.writerow([obj.get('ts',''), obj.get('question',''), obj.get('answer','')])
            except Exception:
                continue

    mem.seek(0)
    # Excel may expect BOM for UTF-8 CSV
    data = mem.getvalue().encode('utf-8-sig')
    mem_bytes = io.BytesIO(data)
    filename = f"chat_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(mem_bytes, as_attachment=True, download_name=filename, mimetype='text/csv')


@app.route('/admin/feedback_action', methods=['POST'])
@requires_admin_auth
def admin_feedback_action():
    try:
        data = request.get_json(force=True)
        action = data.get('action')
        items = data.get('items', [])
    except Exception:
        return {"ok": False, "message": "Dữ liệu không hợp lệ"}, 400

    fb_root = BASE_DIR / 'static' / 'feedback'
    samples_dir = BASE_DIR / 'static' / 'images' / 'tomato_samples'
    neg_samples_dir = BASE_DIR / 'static' / 'images' / 'not_tomato_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    neg_samples_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    try:
        for it in items:
            # sanitize client-provided dir/name to prevent path traversal
            dir_name = it.get('dir', '')
            file_name = it.get('name', '')
            if not dir_name or '..' in dir_name or '/' in dir_name or '\\' in dir_name:
                app.logger.warning('Skipping invalid dir name from admin payload: %s', dir_name)
                continue
            if not file_name or '..' in file_name or '/' in file_name or '\\' in file_name:
                app.logger.warning('Skipping invalid file name from admin payload: %s', file_name)
                continue

            src_dir = fb_root / dir_name
            src_file = src_dir / file_name
            if not src_file.exists():
                continue
            
            target_dir = None
            if action == 'add_to_samples':
                target_dir = samples_dir
            elif action == 'add_to_negative_samples':
                target_dir = neg_samples_dir

            if target_dir:
                target = target_dir / src_file.name
                if target.exists():
                    base, suffix = src_file.stem, src_file.suffix
                    i = 1
                    while (target_dir / f"{base}_{i}{suffix}").exists():
                        i += 1
                    target = target_dir / f"{base}_{i}{suffix}"
                shutil.move(str(src_file), str(target))
                processed += 1
            elif action == 'delete':
                src_file.unlink()
                processed += 1
        # If we moved new samples into positive/negative dirs, rebuild sample features in background
        if processed > 0 and action in ('add_to_samples', 'add_to_negative_samples'):
            def _background_rebuild():
                try:
                    import build_sample_features
                    app.logger.info('Starting background rebuild of sample features')
                    build_sample_features.main()
                    # reload in-memory cache
                    global CACHED_SAMPLE_FEATURES
                    CACHED_SAMPLE_FEATURES = None
                    _load_sample_features_cache()
                    app.logger.info('Background rebuild of sample features completed')
                except Exception:
                    app.logger.exception('Background rebuild of sample features failed')

            t = threading.Thread(target=_background_rebuild, daemon=True)
            t.start()

        return {"ok": True, "message": f"Đã xử lý {processed} ảnh."}
    except Exception as e:
        app.logger.exception('Lỗi admin action')
        return {"ok": False, "message": f"Lỗi: {e}"}, 500


@app.route('/admin/reload_samples', methods=['POST'])
@requires_admin_auth
def admin_reload_samples():
    """Reload the cached sample features from `data/sample_features.pkl`.
    This allows adding new samples and updating the in-memory cache without restarting the server.
    """
    global CACHED_SAMPLE_FEATURES
    try:
        # Invalidate and reload
        CACHED_SAMPLE_FEATURES = None
        sf = _load_sample_features_cache()
        if not sf:
            return {"ok": False, "message": "Không tìm thấy sample_features.pkl hoặc nó rỗng."}, 404
        pos = len(sf.get('positive', []))
        neg = len(sf.get('negative', []))
        return {"ok": True, "message": f"Reloaded sample features (positive={pos}, negative={neg})", "counts": {"positive": pos, "negative": neg}}
    except Exception as e:
        app.logger.exception('Error reloading sample features')
        return {"ok": False, "message": str(e)}, 500


@app.route('/admin/rebuild_samples', methods=['POST'])
@requires_admin_auth
def admin_rebuild_samples():
    """Trigger a background rebuild of `data/sample_features.pkl` by running
    `build_sample_features.main()` in a daemon thread. Returns immediately.
    """
    try:
        def _background_rebuild():
            try:
                import build_sample_features
                app.logger.info('Starting background rebuild of sample features (admin trigger)')
                build_sample_features.main()
                # invalidate and reload cache
                global CACHED_SAMPLE_FEATURES
                CACHED_SAMPLE_FEATURES = None
                _load_sample_features_cache()
                app.logger.info('Background rebuild of sample features completed (admin trigger)')
            except Exception:
                app.logger.exception('Background rebuild of sample features failed (admin trigger)')

        t = threading.Thread(target=_background_rebuild, daemon=True)
        t.start()
        return {"ok": True, "message": "Đã bắt đầu rebuild sample features (chạy nền)."}
    except Exception as e:
        app.logger.exception('Error triggering rebuild of sample features')
        return {"ok": False, "message": str(e)}, 500

# ==================== Video Tutorials & Live Streaming ====================

def load_video_tutorials():
    """Load all video tutorials"""
    tutorials = []
    tutorials_file = BASE_DIR / 'data' / 'video_tutorials.jsonl'
    
    if tutorials_file.exists():
        with open(tutorials_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        tutorials.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return tutorials

def load_live_streams():
    """Load all live streams"""
    streams = []
    streams_file = BASE_DIR / 'data' / 'live_streams.jsonl'
    
    if streams_file.exists():
        with open(streams_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        streams.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return streams

def get_video_tutorial_by_id(tutorial_id):
    """Get video tutorial by ID"""
    tutorials = load_video_tutorials()
    for tutorial in tutorials:
        if tutorial['id'] == tutorial_id:
            return tutorial
    return None

def get_live_stream_by_id(stream_id):
    """Get live stream by ID"""
    streams = load_live_streams()
    for stream in streams:
        if stream['id'] == stream_id:
            return stream
    return None

def save_video_tutorials(tutorials):
    """Save all video tutorials"""
    tutorials_file = BASE_DIR / 'data' / 'video_tutorials.jsonl'
    tutorials_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tutorials_file, 'w', encoding='utf-8') as f:
        for tutorial in tutorials:
            f.write(json.dumps(tutorial, ensure_ascii=False) + '\n')

def save_live_streams(streams):
    """Save all live streams"""
    streams_file = BASE_DIR / 'data' / 'live_streams.jsonl'
    streams_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(streams_file, 'w', encoding='utf-8') as f:
        for stream in streams:
            f.write(json.dumps(stream, ensure_ascii=False) + '\n')

def increment_video_views(tutorial_id):
    """Increment views count for a video tutorial"""
    try:
        tutorials = load_video_tutorials()
        for tutorial in tutorials:
            if tutorial['id'] == tutorial_id:
                tutorial['views'] = tutorial.get('views', 0) + 1
                break
        save_video_tutorials(tutorials)
    except Exception as e:
        app.logger.error(f'Error incrementing video views: {e}')

def increment_stream_views(stream_id):
    """Increment views count for a live stream"""
    try:
        streams = load_live_streams()
        for stream in streams:
            if stream['id'] == stream_id:
                stream['views'] = stream.get('views', 0) + 1
                break
        save_live_streams(streams)
    except Exception as e:
        app.logger.error(f'Error incrementing stream views: {e}')

def toggle_video_like(tutorial_id, user_id):
    """Toggle like for a video tutorial"""
    try:
        tutorials = load_video_tutorials()
        for tutorial in tutorials:
            if tutorial['id'] == tutorial_id:
                if 'liked_by' not in tutorial:
                    tutorial['liked_by'] = []
                
                if user_id in tutorial['liked_by']:
                    tutorial['liked_by'].remove(user_id)
                    tutorial['likes'] = tutorial.get('likes', 0) - 1
                    liked = False
                else:
                    tutorial['liked_by'].append(user_id)
                    tutorial['likes'] = tutorial.get('likes', 0) + 1
                    liked = True
                
                save_video_tutorials(tutorials)
                return liked
        return False
    except Exception as e:
        app.logger.error(f'Error toggling video like: {e}')
        return False

@app.route('/tutorials')
def video_tutorials():
    """Video tutorials page"""
    try:
        tutorials = load_video_tutorials()
        
        # Get filter parameters
        category = request.args.get('category', '').strip()
        level = request.args.get('level', '').strip()
        search_query = request.args.get('q', '').strip()
        
        # Filter tutorials
        filtered_tutorials = tutorials
        
        if category:
            filtered_tutorials = [t for t in filtered_tutorials if t.get('category') == category]
        
        if level:
            filtered_tutorials = [t for t in filtered_tutorials if t.get('level') == level]
        
        if search_query:
            search_lower = search_query.lower()
            filtered_tutorials = [
                t for t in filtered_tutorials 
                if search_lower in t.get('title', '').lower() 
                or search_lower in t.get('description', '').lower()
                or any(search_lower in tag.lower() for tag in t.get('tags', []))
            ]
        
        # Sort by views (popular) by default
        sort_by = request.args.get('sort', 'popular')
        if sort_by == 'popular':
            filtered_tutorials = sorted(filtered_tutorials, key=lambda x: x.get('views', 0), reverse=True)
        elif sort_by == 'recent':
            filtered_tutorials = sorted(filtered_tutorials, key=lambda x: x.get('created_at', ''), reverse=True)
        elif sort_by == 'likes':
            filtered_tutorials = sorted(filtered_tutorials, key=lambda x: x.get('likes', 0), reverse=True)
        
        # Get categories and levels for filters
        categories = list(set(t.get('category', '') for t in tutorials if t.get('category')))
        levels = ['beginner', 'intermediate', 'advanced']
        
        # Get featured/popular tutorials for sidebar
        popular_tutorials = sorted(tutorials, key=lambda x: x.get('views', 0), reverse=True)[:5]
        
        return render_template('video_tutorials.html',
                             tutorials=filtered_tutorials,
                             popular_tutorials=popular_tutorials,
                             categories=categories,
                             levels=levels,
                             selected_category=category,
                             selected_level=level,
                             search_query=search_query,
                             sort_by=sort_by)
    except Exception as e:
        app.logger.error(f'Error loading video tutorials: {e}')
        flash('Lỗi khi tải video tutorials', 'error')
        return redirect(url_for('index'))

@app.route('/tutorials/<tutorial_id>')
def video_tutorial_detail(tutorial_id):
    """Video tutorial detail page"""
    try:
        tutorial = get_video_tutorial_by_id(tutorial_id)
        
        if not tutorial:
            flash('Không tìm thấy video tutorial', 'warning')
            return redirect(url_for('video_tutorials'))
        
        # Increment views
        increment_video_views(tutorial_id)
        tutorial['views'] = tutorial.get('views', 0) + 1
        
        # Get user like status
        user_liked = False
        if 'user_id' in session:
            user_liked = session['user_id'] in tutorial.get('liked_by', [])
        
        # Get related tutorials (same category or tags)
        all_tutorials = load_video_tutorials()
        related = []
        for t in all_tutorials:
            if t['id'] != tutorial_id:
                score = 0
                if t.get('category') == tutorial.get('category'):
                    score += 5
                common_tags = set(t.get('tags', [])) & set(tutorial.get('tags', []))
                score += len(common_tags) * 2
                if score > 0:
                    t['relevance_score'] = score
                    related.append(t)
        
        related.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        related = related[:5]
        
        return render_template('video_tutorial_detail.html',
                             tutorial=tutorial,
                             related_tutorials=related,
                             user_liked=user_liked)
    except Exception as e:
        app.logger.error(f'Error loading video tutorial: {e}')
        flash('Lỗi khi tải video tutorial', 'error')
        return redirect(url_for('video_tutorials'))

@app.route('/live-streaming')
def live_streaming():
    """Live streaming page"""
    try:
        streams = load_live_streams()
        
        # Separate streams by status
        live_streams = [s for s in streams if s.get('status') == 'live']
        scheduled_streams = [s for s in streams if s.get('status') == 'scheduled']
        ended_streams = [s for s in streams if s.get('status') == 'ended']
        
        # Sort by scheduled time
        scheduled_streams = sorted(scheduled_streams, key=lambda x: x.get('scheduled_time', ''))
        ended_streams = sorted(ended_streams, key=lambda x: x.get('scheduled_time', ''), reverse=True)[:10]
        
        return render_template('live_streaming.html',
                             live_streams=live_streams,
                             scheduled_streams=scheduled_streams,
                             ended_streams=ended_streams)
    except Exception as e:
        app.logger.error(f'Error loading live streams: {e}')
        flash('Lỗi khi tải live streaming', 'error')
        return redirect(url_for('index'))

@app.route('/live-streaming/<stream_id>')
def live_stream_detail(stream_id):
    """Live stream detail page"""
    try:
        stream = get_live_stream_by_id(stream_id)
        
        if not stream:
            flash('Không tìm thấy live stream', 'warning')
            return redirect(url_for('live_streaming'))
        
        # Increment views for live and ended streams
        if stream.get('status') in ['live', 'ended']:
            increment_stream_views(stream_id)
            stream['views'] = stream.get('views', 0) + 1
        
        return render_template('live_stream_detail.html', stream=stream)
    except Exception as e:
        app.logger.error(f'Error loading live stream: {e}')
        flash('Lỗi khi tải live stream', 'error')
        return redirect(url_for('live_streaming'))

@app.route('/api/video/<tutorial_id>/like', methods=['POST'])
def video_like(tutorial_id):
    """Toggle like for video tutorial"""
    try:
        if 'user_id' not in session:
            return {'success': False, 'message': 'Vui lòng đăng nhập'}, 401
        
        user_id = session['user_id']
        liked = toggle_video_like(tutorial_id, user_id)
        
        tutorial = get_video_tutorial_by_id(tutorial_id)
        likes = tutorial.get('likes', 0) if tutorial else 0
        
        return {
            'success': True,
            'liked': liked,
            'likes': likes
        }
    except Exception as e:
        app.logger.error(f'Error liking video tutorial: {e}')
        return {'success': False, 'message': str(e)}, 500

# ==================== Admin Wiki Management ====================

@app.route('/admin/wiki')
@requires_admin_auth
def admin_wiki():
    """Admin page to manage wiki articles"""
    try:
        articles = load_wiki_articles()
        categories = get_wiki_categories()
        tags = get_wiki_tags()
        
        # Sort by created_at (newest first)
        articles.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return render_template('admin_wiki.html',
                             articles=articles,
                             categories=categories,
                             tags=tags)
    except Exception as e:
        app.logger.error(f'Error loading admin wiki: {e}')
        flash('Lỗi khi tải trang quản lý wiki', 'error')
        return redirect(url_for('admin_stats'))

@app.route('/admin/wiki/add', methods=['GET', 'POST'])
@requires_admin_auth
def admin_wiki_add():
    """Add new wiki article"""
    if request.method == 'POST':
        try:
            title = request.form.get('title', '').strip()
            slug = request.form.get('slug', '').strip()
            category = request.form.get('category', '').strip()
            content = request.form.get('content', '').strip()
            tags = request.form.get('tags', '').strip()
            featured = request.form.get('featured') == 'on'
            
            if not title or not slug or not content:
                flash('Vui lòng điền đầy đủ thông tin', 'warning')
                return render_template('admin_wiki_form.html')
            
            # Check if slug already exists
            if get_wiki_article_by_slug(slug):
                flash('Slug đã tồn tại, vui lòng chọn slug khác', 'warning')
                return render_template('admin_wiki_form.html')
            
            # Parse tags
            tags_list = [t.strip() for t in tags.split(',') if t.strip()]
            
            # Create article
            article = {
                'id': str(uuid4())[:12],
                'title': title,
                'slug': slug,
                'category': category,
                'content': content,
                'author': session.get('username', 'Admin'),
                'views': 0,
                'likes': 0,
                'liked_by': [],
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'tags': tags_list,
                'featured': featured
            }
            
            articles = load_wiki_articles()
            articles.append(article)
            save_wiki_articles(articles)
            
            flash('Đã thêm bài viết wiki thành công', 'success')
            return redirect(url_for('admin_wiki'))
            
        except Exception as e:
            app.logger.error(f'Error adding wiki article: {e}')
            flash('Lỗi khi thêm bài viết', 'error')
            return render_template('admin_wiki_form.html')
    
    return render_template('admin_wiki_form.html', article=None)

@app.route('/admin/wiki/<article_id>/edit', methods=['GET', 'POST'])
@requires_admin_auth
def admin_wiki_edit(article_id):
    """Edit wiki article"""
    article = get_wiki_article_by_id(article_id)
    
    if not article:
        flash('Không tìm thấy bài viết', 'warning')
        return redirect(url_for('admin_wiki'))
    
    if request.method == 'POST':
        try:
            title = request.form.get('title', '').strip()
            slug = request.form.get('slug', '').strip()
            category = request.form.get('category', '').strip()
            content = request.form.get('content', '').strip()
            tags = request.form.get('tags', '').strip()
            featured = request.form.get('featured') == 'on'
            
            if not title or not slug or not content:
                flash('Vui lòng điền đầy đủ thông tin', 'warning')
                return render_template('admin_wiki_form.html', article=article)
            
            # Check if slug changed and conflicts with another article
            if slug != article['slug']:
                existing = get_wiki_article_by_slug(slug)
                if existing and existing['id'] != article_id:
                    flash('Slug đã tồn tại, vui lòng chọn slug khác', 'warning')
                    return render_template('admin_wiki_form.html', article=article)
            
            # Parse tags
            tags_list = [t.strip() for t in tags.split(',') if t.strip()]
            
            # Update article
            articles = load_wiki_articles()
            for a in articles:
                if a['id'] == article_id:
                    a['title'] = title
                    a['slug'] = slug
                    a['category'] = category
                    a['content'] = content
                    a['tags'] = tags_list
                    a['featured'] = featured
                    a['updated_at'] = datetime.utcnow().isoformat()
                    break
            
            save_wiki_articles(articles)
            
            flash('Đã cập nhật bài viết thành công', 'success')
            return redirect(url_for('admin_wiki'))
            
        except Exception as e:
            app.logger.error(f'Error editing wiki article: {e}')
            flash('Lỗi khi cập nhật bài viết', 'error')
            return render_template('admin_wiki_form.html', article=article)
    
    # Convert tags list to comma-separated string for form
    if article.get('tags'):
        article['tags_str'] = ', '.join(article['tags'])
    else:
        article['tags_str'] = ''
    
    return render_template('admin_wiki_form.html', article=article)

@app.route('/admin/wiki/<article_id>/delete', methods=['POST'])
@requires_admin_auth
def admin_wiki_delete(article_id):
    """Delete wiki article"""
    try:
        articles = load_wiki_articles()
        articles = [a for a in articles if a['id'] != article_id]
        save_wiki_articles(articles)
        
        flash('Đã xóa bài viết thành công', 'success')
        return redirect(url_for('admin_wiki'))
        
    except Exception as e:
        app.logger.error(f'Error deleting wiki article: {e}')
        flash('Lỗi khi xóa bài viết', 'error')
        return redirect(url_for('admin_wiki'))

@app.route('/chat')
def chat():
    """Render chat page"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chatbot sử dụng Gemini API để trả lời câu hỏi về cà chua."""
    start_time = time.time()
    user_ip = request.remote_addr or 'unknown'
    session_id = session.get('id', 'no-session')
    
    # Rate limiting
    current_time = time.time()
    CHAT_RATE_LIMITER[user_ip] = [
        ts for ts in CHAT_RATE_LIMITER[user_ip] 
        if current_time - ts < 60  # Chỉ giữ timestamps trong 1 phút
    ]
    
    if len(CHAT_RATE_LIMITER[user_ip]) >= CHAT_RATE_LIMIT_PER_MINUTE:
        app.logger.warning(f"Rate limit exceeded for IP: {user_ip}")
        return {
            "answer": "[!] Bạn đang hỏi quá nhanh. Vui lòng đợi một chút rồi thử lại.",
            "error": "rate_limit"
        }, 429
    
    CHAT_RATE_LIMITER[user_ip].append(current_time)
    
    # Parse request
    try:
        data = request.get_json(force=True)
        user_q_raw = (data.get('q') or '').strip()
    except Exception as e:
        app.logger.error(f"Failed to parse chat request: {e}")
        return {"answer": "Không nhận được câu hỏi."}

    # Input validation
    if not user_q_raw:
        return {"answer": "Vui lòng nhập câu hỏi."}
    
    if len(user_q_raw) > CHAT_MAX_QUESTION_LENGTH:
        return {
            "answer": f"[!] Câu hỏi quá dài. Vui lòng giới hạn trong {CHAT_MAX_QUESTION_LENGTH} ký tự.",
            "error": "question_too_long"
        }
    
    # Sanitize input (remove excessive whitespace)
    user_q_clean = ' '.join(user_q_raw.split())

    # Log request
    app.logger.info(f"[CHAT] IP={user_ip} Session={session_id} Q='{user_q_clean[:100]}...'")
    
    # Get response
    try:
        answer = get_gemini_response(user_q_clean)
        response_time = time.time() - start_time
        tokens_used = estimate_tokens(answer)
        
        app.logger.info(f"[CHAT] Response generated in {response_time:.2f}s (~{tokens_used} tokens)")
        
        # Rich logging
        try:
            logs_dir = BASE_DIR / 'data'
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / 'chat_logs.jsonl'
            
            entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'question': user_q_clean,
                'answer': answer,
                'user_ip': user_ip,
                'session_id': session_id,
                'response_time_ms': int(response_time * 1000),
                'tokens_estimated': tokens_used,
                'model_used': LAST_WORKING_MODEL or 'unknown',
                'cached': 'FAQ' in answer or 'cache' in answer.lower()
            }
            
            with open(log_file, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            app.logger.exception(f'Không thể ghi log chat: {e}')
        
        return {"answer": answer}
        
    except Exception as e:
        app.logger.exception(f"Error in api_chat: {e}")
        return {
            "answer": "[!] Đã xảy ra lỗi. Vui lòng thử lại sau.",
            "error": "internal_error"
        }, 500

# ============= PREDICTION HELPER FUNCTIONS =============

def validate_request_parameters(request):
    """
    Xác thực và trích xuất các tham số từ Flask request.
    
    Hàm này kiểm tra tính hợp lệ của request bao gồm:
    - File upload có tồn tại và không rỗng
    - Model name và pipeline key đã được chọn
    - Extension của file có nằm trong danh sách cho phép
    
    Args:
        request (flask.Request): Flask request object chứa form data và files
    
    Returns:
        dict: Dictionary chứa các key sau:
            - 'file': FileStorage object của file được upload
            - 'model_name': Tên model được chọn (str)
            - 'pipeline_key': Key của preprocessing pipeline (str)
    
    Raises:
        ValidationError: Khi thiếu tham số, file không hợp lệ, hoặc extension không được hỗ trợ
    
    Example:
        >>> params = validate_request_parameters(request)
        >>> print(params['model_name'])  # 'VGG19_average_hsv'
    """
    try:
        if 'file' not in request.files:
            app.logger.warning("Request missing 'file' field")
            raise ValidationError(
                "No file in request",
                user_message="Không tìm thấy file."
            )
        
        f = request.files['file']
        if f.filename == '':
            app.logger.warning("Empty filename in request")
            raise ValidationError(
                "Empty filename",
                user_message="Bạn chưa chọn file."
            )
        
        model_name = request.form.get('model')
        pipeline_key = request.form.get('pipeline')
        
        if model_name is None or pipeline_key is None:
            app.logger.warning(
                "Missing parameters - model: %s, pipeline: %s",
                model_name, pipeline_key
            )
            raise ValidationError(
                "Missing model or pipeline parameter",
                details={'model': model_name, 'pipeline': pipeline_key},
                user_message="Vui lòng chọn model và pipeline."
            )
        
        if not allowed_file(f.filename):
            app.logger.warning("Invalid file extension: %s", f.filename)
            raise ValidationError(
                f"Invalid file extension: {f.filename}",
                details={'filename': f.filename, 'allowed': list(ALLOWED_EXT)},
                user_message="Định dạng file không hợp lệ. Chỉ chấp nhận: " + ", ".join(ALLOWED_EXT)
            )
        
        app.logger.info(
            "Request validated - model: %s, pipeline: %s, file: %s",
            model_name, pipeline_key, f.filename
        )
        
        return {
            'file': f,
            'model_name': model_name,
            'pipeline_key': pipeline_key
        }
    
    except ValidationError:
        raise
    except Exception as e:
        app.logger.exception("Unexpected error in validate_request_parameters")
        raise ValidationError(
            "Unexpected validation error",
            details={'error': str(e)},
            user_message="Lỗi xác thực request"
        ) from e

def validate_and_decode_image(file_obj):
    """
    Validate image file and decode to OpenCV BGR format.
    Returns: numpy array (BGR) or raises ValidationError
    """
    try:
        raw_bytes = file_obj.read()
        file_size_mb = len(raw_bytes) / (1024 * 1024)
        app.logger.debug("Image file size: %.2f MB", file_size_mb)
        
        if len(raw_bytes) == 0:
            app.logger.error("Empty file uploaded")
            raise ValidationError(
                "Empty file",
                user_message="File rỗng, vui lòng chọn file khác."
            )
        
        # Verify image integrity with PIL (also validates format)
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.verify()
            app.logger.debug("PIL verification passed")
        except Exception as e:
            app.logger.error("PIL verification failed: %s", str(e))
            raise ImageProcessingError(
                "Corrupt or invalid image file",
                details={'error': str(e)},
                user_message="Ảnh hỏng hoặc không thể xác thực (corrupt)."
            ) from e
        
        # Decode to OpenCV format
        file_bytes = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            app.logger.error("cv2.imdecode returned None")
            raise ImageProcessingError(
                "Failed to decode image with OpenCV",
                user_message="Không thể đọc ảnh. Vui lòng thử file khác."
            )
        
        app.logger.debug(
            "Image decoded - shape: %s, dtype: %s, size: %dx%d",
            img_bgr.shape, img_bgr.dtype, img_bgr.shape[1], img_bgr.shape[0]
        )
        
        # Chuẩn hóa các kênh: bỏ kênh alpha nếu có
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            app.logger.debug("Dropping alpha channel")
            img_bgr = img_bgr[..., :3]
        
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            app.logger.error(
                "Invalid image dimensions - ndim: %s, shape: %s",
                img_bgr.ndim, img_bgr.shape
            )
            raise ImageProcessingError(
                f"Invalid image dimensions: {img_bgr.shape}",
                details={'ndim': img_bgr.ndim, 'shape': img_bgr.shape},
                user_message="Ảnh không hợp lệ (không có 3 kênh màu)."
            )
        
        return img_bgr
    
    except (ValidationError, ImageProcessingError):
        raise
    except Exception as e:
        app.logger.exception("Unexpected error in validate_and_decode_image")
        raise ImageProcessingError(
            "Unexpected error decoding image",
            details={'error': str(e)},
            user_message="Lỗi không xác định khi xử lý ảnh"
        ) from e

def prepare_image_for_prediction(img_bgr):
    """
    Resize image if needed and validate it's a leaf.
    Returns: processed numpy array (BGR) or raises ValidationError
    """
    # Enforce maximum dimension to avoid OOM on huge images
    max_dim = app.config.get('MAX_IMAGE_DIM', 3000)
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Quick heuristic: check whether the image looks like a leaf (enough green)
    if not is_leaf_like(img_bgr):
        raise ValidationError(
            "Image does not appear to be a tomato leaf",
            user_message="Ảnh vào có vẻ không phải lá cà chua hoặc không đủ thông tin (vui lòng tải ảnh lá rõ ràng)."
        )
    
    return img_bgr


def assess_image_quality_with_suggestions(img_bgr):
    """
    Assess image quality and provide actionable suggestions.
    
    Args:
        img_bgr: Input image in BGR format
    
    Returns:
        dict with:
            - is_good (bool): Whether quality is sufficient
            - quality_score (float): 0-100
            - quality_level (str): 'Excellent', 'Good', 'Fair', 'Poor'
            - issues (list): List of detected issues
            - suggestions (list): Actionable suggestions for improvement
            - warning_message (str): User-friendly warning if quality is poor
    """
    try:
        is_good, quality_score, issues = check_image_quality(img_bgr, min_size=100, max_blur_threshold=100)
        
        # Determine quality level
        if quality_score >= 85:
            quality_level = 'Excellent'
            level_icon = '🟢'
        elif quality_score >= 70:
            quality_level = 'Good'
            level_icon = '🟢'
        elif quality_score >= 50:
            quality_level = 'Fair'
            level_icon = '🟡'
        else:
            quality_level = 'Poor'
            level_icon = '🔴'
        
        # Generate suggestions based on issues
        suggestions = []
        warning_message = None
        
        for issue in issues:
            if 'too small' in issue.lower():
                suggestions.append('📏 Chụp ảnh với độ phân giải cao hơn hoặc chụp gần lá hơn')
            elif 'blurry' in issue.lower():
                suggestions.append('🎯 Giữ camera ổn định và đợi auto-focus hoàn tất')
                suggestions.append('💡 Chụp ở nơi có ánh sáng tốt để tăng shutter speed')
            elif 'too dark' in issue.lower():
                suggestions.append('💡 Chụp ở nơi sáng hơn hoặc bật flash')
                suggestions.append('☀️ Chụp ngoài trời vào buổi sáng/chiều')
            elif 'too bright' in issue.lower():
                suggestions.append('🌤️ Tránh chụp dưới ánh nắng trực tiếp')
                suggestions.append('🌳 Chụp ở nơi có bóng râm nhẹ')
            elif 'low contrast' in issue.lower():
                suggestions.append('🎨 Đảm bảo lá nổi bật so với background')
                suggestions.append('📱 Điều chỉnh cài đặt camera để tăng contrast')
        
        # Add general suggestions if no specific issues
        if not suggestions:
            suggestions = [
                '[OK] Chất lượng ảnh tốt!',
                '[TIP] Tips: Chụp khi lá đang khô để tránh phản quang'
            ]
        
        # Create warning message for poor quality
        if quality_score < 50:
            warning_message = (
                f"[!] Chất lượng ảnh {quality_level.lower()} (điểm: {quality_score:.0f}/100). "
                "Kết quả dự đoán có thể không chính xác. Đề nghị chụp lại ảnh tốt hơn."
            )
        elif quality_score < 70:
            warning_message = (
                f"[!] Chất lượng ảnh {quality_level.lower()} (điểm: {quality_score:.0f}/100). "
                "Có thể cải thiện để được kết quả chính xác hơn."
            )
        
        return {
            'is_good': is_good,
            'quality_score': round(quality_score, 1),
            'quality_level': quality_level,
            'level_icon': level_icon,
            'issues': issues,
            'suggestions': suggestions,
            'warning_message': warning_message
        }
        
    except Exception as e:
        app.logger.exception("Error assessing image quality: %s", str(e))
        return {
            'is_good': True,
            'quality_score': 50.0,
            'quality_level': 'Unknown',
            'level_icon': '❓',
            'issues': ['Could not assess quality'],
            'suggestions': ['Vui lòng đảm bảo ảnh rõ ràng và có ánh sáng tốt'],
            'warning_message': None
        }

def run_model_prediction(img_bgr, model_name, pipeline_key):
    """
    Thực hiện dự đoán bệnh cà chua từ ảnh đầu vào.
    
    Pipeline xử lý:
    1. Preprocessing: Áp dụng pipeline biến đổi (HSV/CMYK/HSI/Noise reduction)
    2. Model loading: Load model từ cache hoặc disk (với LRU eviction)
    3. Prediction: Chạy forward pass qua CNN và trả về xác suất các lớp
    
    Args:
        img_bgr (np.ndarray): Ảnh đầu vào ở định dạng BGR (OpenCV), shape (H, W, 3)
        model_name (str): Tên model đầy đủ (vd: 'VGG19_average_hsv')
        pipeline_key (str): Key của preprocessing pipeline (vd: 'average_hsv')
    
    Returns:
        dict: Dictionary chứa kết quả dự đoán:
            - 'model': Keras model object đã được load
            - 'class_names': List tên các lớp bệnh (4 classes)
            - 'predictions': Numpy array xác suất dự đoán, shape (1, num_classes)
            - 'preprocessed': Ảnh đã qua preprocessing, shape (1, 224, 224, 3)
    
    Raises:
        ImageProcessingError: Khi preprocessing thất bại
        ModelError: Khi load model hoặc prediction thất bại
    
    Notes:
        - Model được cache trong RAM (LRU cache) để tăng tốc độ
        - Preprocessing pipeline phải khớp với pipeline đã train model
        - Prediction time được log để monitoring performance
    """
    app.logger.info(
        "Starting prediction - model: %s, pipeline: %s, image_shape: %s",
        model_name, pipeline_key, img_bgr.shape
    )
    
    # Preprocess
    try:
        x = preprocess_image_for_model(img_bgr, pipeline_key)
        app.logger.debug(
            "Preprocessing completed - output shape: %s, dtype: %s, range: [%.3f, %.3f]",
            x.shape, x.dtype, float(x.min()), float(x.max())
        )
    except Exception as e:
        app.logger.error("Preprocessing failed: %s", str(e), exc_info=True)
        raise ImageProcessingError(
            "Preprocessing failed",
            details={'pipeline': pipeline_key, 'error': str(e)},
            user_message=f"Lỗi preprocessing: {e}"
        ) from e
    
    # Load model
    try:
        model, model_class_names = load_model_by_name(model_name, pipeline_key)
        app.logger.debug("Model loaded with %s classes", len(model_class_names))
    except ModelError:
        raise
    except Exception as e:
        app.logger.error("Model loading failed: %s", str(e), exc_info=True)
        raise ModelError(
            "Failed to load model",
            details={'model': model_name, 'pipeline': pipeline_key, 'error': str(e)},
            user_message=f"Lỗi load model: {e}"
        ) from e
    
    # Run prediction
    try:
        # Log input diagnostics
        app.logger.debug(
            "Model prediction input - shape: %s, dtype: %s, range: [%.5f, %.5f]",
            x.shape, x.dtype, float(x.min()), float(x.max())
        )
        app.logger.debug("Model expected input_shape: %s", getattr(model, 'input_shape', None))
        
        # Run prediction with timing
        import time
        start_time = time.time()
        preds = model.predict(x, verbose=0)
        prediction_time = time.time() - start_time
        
        app.logger.info(
            "Prediction completed in %.3f seconds - output shape: %s",
            prediction_time, preds.shape
        )
        app.logger.debug("Predictions: %s", preds[0].tolist())
        
    except Exception as e:
        # Detailed error logging for prediction failures
        app.logger.error(
            "Model prediction failed - model: %s, input_shape: %s, input_dtype: %s, "
            "input_range: [%.3f, %.3f], expected_input_shape: %s",
            model_name,
            getattr(x, 'shape', None),
            getattr(x, 'dtype', None),
            float(x.min()) if hasattr(x, 'min') else 0,
            float(x.max()) if hasattr(x, 'max') else 0,
            getattr(model, 'input_shape', None),
            exc_info=True
        )
        raise ModelError(
            "Model prediction failed",
            details={
                'model': model_name,
                'pipeline': pipeline_key,
                'input_shape': str(x.shape),
                'expected_shape': str(getattr(model, 'input_shape', None)),
                'error': str(e)
            },
            user_message=f"Lỗi khi chạy model.predict: {e}. Kiểm tra log server để biết chi tiết."
        ) from e
    
    # Validate output
    try:
        num_outputs = preds.shape[-1] if preds.ndim > 1 else 1
        if num_outputs != len(model_class_names):
            app.logger.error(
                "Output size mismatch - model outputs: %s, class labels: %s",
                num_outputs, len(model_class_names)
            )
            app.logger.error("Predictions: %s", preds.tolist())
            app.logger.error("Class names: %s", model_class_names)
            raise ModelError(
                "Model output size mismatch",
                details={
                    'num_outputs': num_outputs,
                    'num_classes': len(model_class_names),
                    'predictions': preds.tolist()
                },
                user_message=(
                    f"Lỗi: model trả về {num_outputs} lớp nhưng file label có {len(model_class_names)} lớp. "
                    "Kiểm tra file model / labels."
                )
            )
    except ModelError:
        raise
    except Exception as e:
        app.logger.exception("Error validating prediction output")
        raise ModelError(
            "Failed to validate prediction output",
            details={'error': str(e)},
            user_message="Lỗi xác thực kết quả prediction"
        ) from e
    
    # Get predicted index for Grad-CAM
    pred_index = int(np.argmax(preds[0]))
    
    app.logger.info("Prediction successful")
    return {
        'model': model,
        'class_names': model_class_names,
        'predictions': preds,
        'preprocessed': x,
        'pred_index': pred_index
    }

def process_prediction_results(preds, class_names):
    """
    Process raw predictions and return sorted results.
    Returns: dict with 'label', 'probability', 'all_probs'
    """
    probs = preds[0].tolist()
    pairs = list(zip(class_names, probs))
    pairs_sorted = sorted(pairs, key=lambda p: p[1], reverse=True)
    predicted_label, predicted_prob = pairs_sorted[0]
    
    return {
        'label': predicted_label,
        'probability': predicted_prob,
        'all_probs': pairs_sorted
    }

def save_display_image(preprocessed_img, pipeline_key, filename=None):
    """
    Revert preprocessing and save image for display.
    Args:
        preprocessed_img: Preprocessed image array
        pipeline_key: Pipeline key for reverting preprocessing
        filename: Optional custom filename (default: "last_input.png")
    Returns: URL path to saved image
    """
    display_img = revert_for_display(preprocessed_img[0], pipeline_key)
    uploaded_dir = BASE_DIR / "static" / "uploaded"
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = "last_input.png"
    
    out_path = uploaded_dir / filename
    cv2.imwrite(str(out_path), cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    return url_for('static', filename=f'uploaded/{filename}')

def get_disease_information(predicted_label):
    """
    Get disease information from database.
    Returns: dict with disease info
    """
    return DISEASE_INFO.get(predicted_label, {
        "name": predicted_label,
        "definition": "Không có thông tin định nghĩa cho bệnh này.",
        "prevention": ["Không có thông tin phòng ngừa cụ thể."]
    })

def save_prediction_history(prediction_data):
    """
    Save prediction to history file (JSONL format).
    Returns: prediction_id for reference
    """
    try:
        history_dir = BASE_DIR / 'data'
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / 'prediction_history.jsonl'
        
        prediction_id = str(uuid4())[:8]
        entry = {
            'id': prediction_id,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': prediction_data.get('user_id'),  # Thêm user_id
            'user_email': prediction_data.get('user_email'),  # Thêm user_email
            'model_name': prediction_data.get('model_name'),
            'pipeline_key': prediction_data.get('pipeline_key'),
            'predicted_label': prediction_data.get('predicted_label'),
            'probability': float(prediction_data.get('probability', 0)),
            'possibly_not_tomato': prediction_data.get('possibly_not_tomato', False),
            'rejected': prediction_data.get('rejected', False),
            'image_path': prediction_data.get('image_path'),
            'severity': prediction_data.get('severity')  # Add severity data
        }
        
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        app.logger.info(f"Saved prediction history: {prediction_id}")
        return prediction_id
    except Exception:
        app.logger.exception("Error saving prediction history")
        return None

def calculate_severity_assessment(model, img_array, predicted_label, confidence, pred_index):
    """
    Calculate disease severity for prediction.
    
    Args:
        model: Keras model
        img_array: Preprocessed image array
        predicted_label: Predicted disease class
        confidence: Model confidence (0-1)
        pred_index: Index of predicted class
    
    Returns:
        severity_dict: Severity assessment dictionary
    """
    try:
        app.logger.info("Calculating disease severity...")
        severity = calculate_severity_from_prediction(
            model=model,
            img_array=img_array,
            predicted_label=predicted_label,
            confidence=confidence,
            pred_index=pred_index
        )
        app.logger.info(
            "Severity assessment: level=%s, score=%.2f, affected_area=%.2f%%",
            severity['level'], severity['score'], severity['affected_area']
        )
        return severity
    except Exception as e:
        app.logger.exception("Error calculating severity: %s", str(e))
        # Return default severity on error
        return assess_disease_severity(predicted_label, confidence, 0.0)


def assess_prediction_quality(img_bgr, predicted_prob):
    """
    Đánh giá chất lượng dự đoán dựa trên similarity với tập mẫu.
    
    Thuật toán:
    1. Tính similarity với positive samples (lá cà chua) và negative samples (không phải lá cà chua)
    2. Sử dụng kết hợp histogram correlation + deep embedding (MobileNetV2) cosine similarity
    3. Áp dụng các rule-based thresholds để phát hiện ảnh không phải lá cà chua:
       - Rule 1: neg_sim >= NEG_SIM_THRESH (0.75) → reject
       - Rule 2: neg_sim > pos_sim AND neg_sim >= 0.65 → reject
       - Rule 3: pos_sim < POS_SIM_THRESH (0.60) AND neg_sim >= 0.60 → reject
       - Rule 4: pos_sim < 0.40 AND predicted_prob < MIN_MODEL_CONF → warning
    
    Args:
        img_bgr (np.ndarray): Ảnh đầu vào BGR format, shape (H, W, 3)
        predicted_prob (float): Xác suất dự đoán cao nhất từ model (0-1)
    
    Returns:
        dict: Kết quả đánh giá chất lượng:
            - 'possibly_not_tomato' (bool): Cảnh báo ảnh có thể không phải lá cà chua
            - 'rejected_not_tomato' (bool): Từ chối ảnh (chắc chắn không phải lá cà chua)
            - 'show_feedback' (bool): Hiển thị nút feedback cho user
            - 'sim_info' (dict): Thông tin chi tiết về similarity scores
    
    Notes:
        - Similarity check chỉ chạy khi có sample_features.pkl
        - Histogram correlation range: [-1, 1], higher is more similar
        - Deep embedding cosine similarity range: [-1, 1]
        - Combined score = 0.4 * hist_sim + 0.6 * deep_sim (weighted average)
    """
    possibly_not_tomato = False
    rejected_not_tomato = False
    sim_info = None
    
    try:
        sim_info = compute_sample_similarity(img_bgr)
        if sim_info and sim_info.get('has_samples'):
            pos_sim = sim_info.get('positive_sim', 0.0)
            neg_sim = sim_info.get('negative_sim', 0.0)
            combined = sim_info.get('combined_score', 0.0)
            
            app.logger.info("Sample check: pos_sim=%.3f, neg_sim=%.3f, combined=%.3f", 
                          pos_sim, neg_sim, combined)
            
            # IMPROVED REJECTION LOGIC
            # Case 1: Very high negative similarity -> likely not tomato
            if neg_sim >= NEG_SIM_THRESH:
                rejected_not_tomato = True
                app.logger.info("Rejected: neg_sim (%.3f) >= threshold (%.3f)", 
                              neg_sim, NEG_SIM_THRESH)
            
            # Case 2: Negative similarity much higher than positive -> reject
            elif neg_sim > pos_sim and neg_sim >= 0.65:
                rejected_not_tomato = True
                app.logger.info("Rejected: neg_sim (%.3f) > pos_sim (%.3f) and neg_sim >= 0.65", 
                              neg_sim, pos_sim)
            
            # Case 3: Low positive similarity AND negative similarity is significant
            elif pos_sim < POS_SIM_THRESH and neg_sim >= 0.60:
                rejected_not_tomato = True
                app.logger.info("Rejected: pos_sim (%.3f) < threshold (%.3f) AND neg_sim (%.3f) >= 0.60", 
                              pos_sim, POS_SIM_THRESH, neg_sim)
            
            # Case 4: Warning - unclear image (only if both pos_sim is very low AND model conf is low)
            elif pos_sim < 0.40 and predicted_prob < MIN_MODEL_CONF:
                possibly_not_tomato = True
                app.logger.info("Warning: pos_sim=%.3f < 0.40 AND predicted_prob=%.3f < MIN_MODEL_CONF", 
                              pos_sim, predicted_prob)
        else:
            # No sample data: rely on model confidence
            if predicted_prob < MIN_MODEL_CONF:
                possibly_not_tomato = True
    except Exception:
        app.logger.exception('Error during similarity check')
    
    # Show feedback controls when system suspects non-tomato or low confidence
    # Always show feedback for rejected images to allow user correction
    show_feedback = rejected_not_tomato or (possibly_not_tomato and predicted_prob < 0.95)
    
    return {
        'possibly_not_tomato': possibly_not_tomato,
        'rejected_not_tomato': rejected_not_tomato,
        'show_feedback': show_feedback,
        'sim_info': sim_info
    }

# ============= MAIN PREDICTION ENDPOINT =============

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint - handles multiple images at once.
    """
    request_id = str(uuid4())[:8]
    app.logger.info("=" * 50)
    app.logger.info("New BATCH prediction request [ID: %s]", request_id)
    
    try:
        # Get model and pipeline parameters
        model_name = request.form.get('model', 'VGG19')
        pipeline_key = request.form.get('pipeline', 'average_hsv')
        
        # Save to session
        session['last_model'] = model_name
        session['last_pipeline'] = pipeline_key
        
        # Get all uploaded files
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            raise ValidationError(
                "No files uploaded",
                user_message="Vui lòng chọn ít nhất một ảnh để dự đoán"
            )
        
        if len(files) > 10:
            raise ValidationError(
                f"Too many files: {len(files)}",
                user_message="Chỉ được upload tối đa 10 ảnh cùng lúc"
            )
        
        app.logger.info("[%s] Processing %d images", request_id, len(files))
        
        results = []
        failed_images = []
        
        for idx, file in enumerate(files):
            try:
                app.logger.info("[%s] Processing image %d/%d: %s", 
                              request_id, idx + 1, len(files), file.filename)
                
                # Validate and decode image
                img_bgr = validate_and_decode_image(file)
                
                # Prepare image
                img_bgr = prepare_image_for_prediction(img_bgr)
                
                # Assess image quality
                image_quality = assess_image_quality_with_suggestions(img_bgr)
                
                # Run prediction
                prediction_result = run_model_prediction(
                    img_bgr, model_name, pipeline_key
                )
                
                # Process results
                pred_results = process_prediction_results(
                    prediction_result['predictions'],
                    prediction_result['class_names']
                )
                
                # Save image with unique name
                image_filename = f"batch_{request_id}_{idx}_{file.filename}"
                image_path = save_display_image(
                    prediction_result['preprocessed'],
                    pipeline_key,
                    filename=image_filename
                )
                
                # Get disease info
                disease_info = get_disease_information(pred_results['label'])
                
                # Calculate severity
                severity = calculate_severity_assessment(
                    model=prediction_result['model'],
                    img_array=prediction_result['preprocessed'],
                    predicted_label=pred_results['label'],
                    confidence=pred_results['probability'],
                    pred_index=prediction_result['pred_index']
                )
                
                # Assess quality
                quality = assess_prediction_quality(img_bgr, pred_results['probability'])
                
                # Save to history
                prediction_id = save_prediction_history({
                    'user_id': session.get('user_id'),
                    'user_email': session.get('user_email'),
                    'model_name': model_name,
                    'pipeline_key': pipeline_key,
                    'predicted_label': pred_results['label'],
                    'probability': pred_results['probability'],
                    'possibly_not_tomato': quality['possibly_not_tomato'],
                    'rejected': quality['rejected_not_tomato'],
                    'image_path': image_path,
                    'severity': severity
                })
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'predicted_label': pred_results['label'],
                    'disease_name': disease_info['name'],
                    'probability': pred_results['probability'],
                    'image_path': image_path,
                    'possibly_not_tomato': quality['possibly_not_tomato'],
                    'rejected_not_tomato': quality['rejected_not_tomato'],
                    'prediction_id': prediction_id,
                    'all_probs': pred_results['all_probs'],
                    'severity': severity,
                    'image_quality': image_quality
                })
                
                app.logger.info(
                    "[%s] Image %d/%d completed: %s (%.2f%%)",
                    request_id, idx + 1, len(files),
                    pred_results['label'], pred_results['probability'] * 100
                )
                
            except Exception as e:
                app.logger.error(
                    "[%s] Error processing image %d (%s): %s",
                    request_id, idx + 1, file.filename, str(e)
                )
                failed_images.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        app.logger.info(
            "[%s] Batch prediction completed: %d successful, %d failed",
            request_id, len(results), len(failed_images)
        )
        app.logger.info("=" * 50)
        
        return render_template(
            'batch_result.html',
            results=results,
            failed_images=failed_images,
            model_name=model_name,
            pipeline_key=pipeline_key,
            total_images=len(files),
            successful=len(results),
            failed=len(failed_images)
        )
        
    except AppException as e:
        app.logger.error(
            "[%s] Application error in batch_predict: %s - %s",
            request_id, type(e).__name__, e.message
        )
        flash(e.user_message)
        return redirect(url_for('index'))
    
    except Exception as e:
        app.logger.exception(
            "[%s] Unexpected error in batch_predict: %s",
            request_id, str(e)
        )
        flash("Lỗi không mong muốn khi xử lý batch. Vui lòng thử lại.")
        return redirect(url_for('index'))

# ============= MODEL COMPARISON ENDPOINTS =============

@app.route('/compare')
def compare():
    """Trang so sánh models"""
    return render_template(
        'compare.html',
        models=ARCHITECTURES,
        pipelines=list(PIPELINES.keys()),
        last_model=session.get('last_model', 'VGG19'),
        last_pipeline=session.get('last_pipeline', 'average_hsv')
    )

@app.route('/compare_predict', methods=['POST'])
def compare_predict():
    """
    Compare prediction endpoint - run same image through multiple models/pipelines.
    """
    request_id = str(uuid4())[:8]
    app.logger.info("=" * 50)
    app.logger.info("New COMPARE prediction request [ID: %s]", request_id)
    
    try:
        # Get image
        file = request.files.get('file')
        if not file:
            raise ValidationError(
                "No file uploaded",
                user_message="Vui lòng chọn ảnh để so sánh"
            )
        
        # Get selected models and pipelines
        selected_models = request.form.getlist('models')
        selected_pipelines = request.form.getlist('pipelines')
        
        if not selected_models:
            raise ValidationError(
                "No models selected",
                user_message="Vui lòng chọn ít nhất một model"
            )
        
        if not selected_pipelines:
            raise ValidationError(
                "No pipelines selected",
                user_message="Vui lòng chọn ít nhất một pipeline"
            )
        
        # Limit combinations
        total_combinations = len(selected_models) * len(selected_pipelines)
        if total_combinations > 20:
            raise ValidationError(
                f"Too many combinations: {total_combinations}",
                user_message=f"Quá nhiều tổ hợp ({total_combinations}). Tối đa 20 tổ hợp (vd: 4 models × 5 pipelines)"
            )
        
        app.logger.info(
            "[%s] Comparing %d models × %d pipelines = %d combinations",
            request_id, len(selected_models), len(selected_pipelines), total_combinations
        )
        
        # Validate and decode image once
        img_bgr = validate_and_decode_image(file)
        img_bgr = prepare_image_for_prediction(img_bgr)
        
        # Save original image for display
        uploaded_dir = BASE_DIR / "static" / "uploaded"
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        compare_img_path = uploaded_dir / f"compare_{request_id}.png"
        cv2.imwrite(str(compare_img_path), img_bgr)
        image_url = url_for('static', filename=f'uploaded/compare_{request_id}.png')
        
        results = []
        
        for model_name in selected_models:
            for pipeline_key in selected_pipelines:
                try:
                    app.logger.info(
                        "[%s] Testing: %s + %s",
                        request_id, model_name, pipeline_key
                    )
                    
                    # Run prediction
                    prediction_result = run_model_prediction(
                        img_bgr.copy(), model_name, pipeline_key
                    )
                    
                    # Process results
                    pred_results = process_prediction_results(
                        prediction_result['predictions'],
                        prediction_result['class_names']
                    )
                    
                    # Get disease info
                    disease_info = get_disease_information(pred_results['label'])
                    
                    # Assess quality
                    quality = assess_prediction_quality(img_bgr, pred_results['probability'])
                    
                    results.append({
                        'model_name': model_name,
                        'pipeline_key': pipeline_key,
                        'predicted_label': pred_results['label'],
                        'disease_name': disease_info['name'],
                        'probability': pred_results['probability'],
                        'possibly_not_tomato': quality['possibly_not_tomato'],
                        'rejected_not_tomato': quality['rejected_not_tomato'],
                        'all_probs': pred_results['all_probs'][:5],  # Top 5 only
                        'success': True
                    })
                    
                    app.logger.info(
                        "[%s] %s + %s: %s (%.2f%%)",
                        request_id, model_name, pipeline_key,
                        pred_results['label'], pred_results['probability'] * 100
                    )
                    
                except Exception as e:
                    app.logger.error(
                        "[%s] Error with %s + %s: %s",
                        request_id, model_name, pipeline_key, str(e)
                    )
                    results.append({
                        'model_name': model_name,
                        'pipeline_key': pipeline_key,
                        'success': False,
                        'error': str(e)
                    })
        
        app.logger.info(
            "[%s] Comparison completed: %d/%d successful",
            request_id, sum(1 for r in results if r.get('success')), len(results)
        )
        app.logger.info("=" * 50)
        
        return render_template(
            'compare_result.html',
            results=results,
            image_url=image_url,
            total_combinations=total_combinations,
            successful=sum(1 for r in results if r.get('success')),
            failed=sum(1 for r in results if not r.get('success'))
        )
        
    except AppException as e:
        app.logger.error(
            "[%s] Application error in compare_predict: %s",
            request_id, e.message
        )
        flash(e.user_message)
        return redirect(url_for('compare'))
    
    except Exception as e:
        app.logger.exception(
            "[%s] Unexpected error in compare_predict: %s",
            request_id, str(e)
        )
        flash("Lỗi không mong muốn. Vui lòng thử lại.")
        return redirect(url_for('compare'))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Orchestrates the prediction pipeline using helper functions.
    """
    request_id = str(uuid4())[:8]
    app.logger.info("=" * 50)
    app.logger.info("New prediction request [ID: %s]", request_id)
    
    try:
        # Bước 1: Xác thực tham số request
        app.logger.info("[%s] Step 1: Validating request parameters", request_id)
        params = validate_request_parameters(request)
        
        # Save user choices to session
        session['last_model'] = params['model_name']
        session['last_pipeline'] = params['pipeline_key']
        app.logger.debug(
            "[%s] Session updated - model: %s, pipeline: %s",
            request_id, params['model_name'], params['pipeline_key']
        )
        
        # Bước 2: Xác thực và giải mã ảnh
        app.logger.info("[%s] Step 2: Validating and decoding image", request_id)
        img_bgr = validate_and_decode_image(params['file'])
        
        # Bước 3: Chuẩn bị ảnh (resize, xác thực là lá cây)
        app.logger.info("[%s] Step 3: Preparing image for prediction", request_id)
        img_bgr = prepare_image_for_prediction(img_bgr)
        
        # Bước 3.5: Assess image quality
        app.logger.info("[%s] Step 3.5: Assessing image quality", request_id)
        image_quality = assess_image_quality_with_suggestions(img_bgr)
        app.logger.info(
            "[%s] Image quality: %s (score: %.1f/100, issues: %d)",
            request_id, image_quality['quality_level'], 
            image_quality['quality_score'], len(image_quality['issues'])
        )
        
        # Bước 4: Chạy dự đoán model
        app.logger.info("[%s] Step 4: Running model prediction", request_id)
        prediction_result = run_model_prediction(
            img_bgr, 
            params['model_name'], 
            params['pipeline_key']
        )
        
        # Bước 5: Xử lý kết quả dự đoán
        app.logger.info("[%s] Step 5: Processing prediction results", request_id)
        results = process_prediction_results(
            prediction_result['predictions'],
            prediction_result['class_names']
        )
        app.logger.info(
            "[%s] Predicted: %s (confidence: %.2f%%)",
            request_id, results['label'], results['probability'] * 100
        )
        
        # Bước 6: Lưu ảnh hiển thị
        app.logger.info("[%s] Step 6: Saving display image", request_id)
        image_path = save_display_image(
            prediction_result['preprocessed'],
            params['pipeline_key']
        )
        
        # Bước 7: Lấy thông tin bệnh
        app.logger.info("[%s] Step 7: Getting disease information", request_id)
        disease_info = get_disease_information(results['label'])
        
        # Bước 7.5: Calculate disease severity
        app.logger.info("[%s] Step 7.5: Calculating disease severity", request_id)
        severity = calculate_severity_assessment(
            model=prediction_result['model'],
            img_array=prediction_result['preprocessed'],
            predicted_label=results['label'],
            confidence=results['probability'],
            pred_index=prediction_result['pred_index']
        )
        
        # Bước 8: Đánh giá chất lượng dự đoán
        app.logger.info("[%s] Step 8: Assessing prediction quality", request_id)
        quality = assess_prediction_quality(img_bgr, results['probability'])
        
        # Bước 9: Lưu lịch sử dự đoán
        app.logger.info("[%s] Step 9: Saving prediction history", request_id)
        prediction_id = save_prediction_history({
            'user_id': session.get('user_id'),
            'user_email': session.get('user_email'),
            'model_name': params['model_name'],
            'pipeline_key': params['pipeline_key'],
            'predicted_label': results['label'],
            'probability': results['probability'],
            'possibly_not_tomato': quality['possibly_not_tomato'],
            'rejected': quality['rejected_not_tomato'],
            'image_path': image_path,
            'severity': severity
        })
        
        # Award points for prediction
        if 'user_id' in session and session.get('user_id') != 'admin':
            try:
                user_id = session['user_id']
                add_points(user_id, 10, 'prediction', 'Dự đoán bệnh cà chua')
                
                # Update quest progress
                update_quest_progress(user_id, 'scan', increment=1)
                
                # Check for new achievements
                check_and_award_achievements(user_id, 'scan', {'disease': results['label']})
                
                user = get_user_by_id(user_id)
                if user:
                    session['user_points'] = user.get('points', 0)
            except Exception as e:
                app.logger.error(f'Failed to award prediction points: {e}')
        
        # Step 10: Render result
        app.logger.info(
            "[%s] Step 10: Rendering result (quality: %s, severity: %s)",
            request_id, image_quality['quality_level'], severity['level']
        )
        app.logger.info("[%s] Prediction request completed successfully", request_id)
        app.logger.info("=" * 50)
        
        return render_template(
            'result.html',
            model_name=params['model_name'],
            pipeline_key=params['pipeline_key'],
            classes=prediction_result['class_names'],
            probs=results['all_probs'],
            predicted_label=results['label'],
            predicted_prob=results['probability'],
            image_path=image_path,
            disease_info=disease_info,
            possibly_not_tomato=quality['possibly_not_tomato'],
            rejected_not_tomato=quality['rejected_not_tomato'],
            show_feedback=quality['show_feedback'],
            sim_info=quality['sim_info'],
            prediction_id=prediction_id,
            severity=severity,
            image_quality=image_quality
        )
    
    except AppException as e:
        # Handle known application exceptions
        app.logger.error(
            "[%s] Application error in predict: %s - %s",
            request_id, type(e).__name__, e.message,
            extra={'details': e.details}
        )
        flash(e.user_message)
        return redirect(url_for('index'))
    
    except Exception as e:
        # Handle unexpected exceptions
        app.logger.exception(
            "[%s] Unexpected error in predict endpoint: %s",
            request_id, str(e)
        )
        flash("Lỗi không mong muốn. Vui lòng thử lại hoặc liên hệ quản trị viên.")
        return redirect(url_for('index'))
    
    finally:
        app.logger.debug("[%s] Predict request handler finished", request_id)

@app.route('/api/cache_stats', methods=['GET'])
@requires_admin_auth
def api_cache_stats():
    """Get model cache statistics (admin endpoint)."""
    try:
        stats = LOADED_MODELS.get_stats()
        return {
            'ok': True,
            'cache_stats': stats,
            'memory_info': {
                'max_models': MAX_LOADED_MODELS,
                'current_models': stats['size'],
                'cached_models': stats['keys']
            }
        }
    except Exception as e:
        app.logger.exception("Error getting cache stats")
        return {'ok': False, 'error': str(e)}, 500

@app.route('/api/clear_cache', methods=['POST'])
@requires_admin_auth
def api_clear_cache():
    """Clear model cache (admin endpoint)."""
    try:
        old_stats = LOADED_MODELS.get_stats()
        LOADED_MODELS.clear()
        app.logger.info("Cache cleared by admin request")
        return {
            'ok': True,
            'message': f"Cache cleared. Freed {old_stats['size']} models.",
            'old_stats': old_stats
        }
    except Exception as e:
        app.logger.exception("Error clearing cache")
        return {'ok': False, 'error': str(e)}, 500


# ==================== COMPUTER VISION ENHANCEMENT ENDPOINTS ====================

@app.route('/api/enhance_image', methods=['POST'])
def api_enhance_image():
    """Enhance image quality before prediction."""
    try:
        if 'file' not in request.files:
            return {'ok': False, 'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        denoise = request.form.get('denoise', 'true').lower() == 'true'
        sharpen = request.form.get('sharpen', 'true').lower() == 'true'
        adjust_brightness = request.form.get('adjust_brightness', 'true').lower() == 'true'
        
        # Read image
        raw_bytes = file.read()
        arr = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return {'ok': False, 'error': 'Invalid image file'}, 400
        
        # Enhance image
        from utils import enhance_image_quality
        enhanced = enhance_image_quality(img_bgr, denoise, sharpen, adjust_brightness)
        
        # Encode to base64
        import base64
        _, buffer = cv2.imencode('.jpg', enhanced)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'ok': True,
            'enhanced_image': f'data:image/jpeg;base64,{img_base64}',
            'enhancements': {
                'denoise': denoise,
                'sharpen': sharpen,
                'adjust_brightness': adjust_brightness
            }
        }
        
    except Exception as e:
        app.logger.exception("Error enhancing image")
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/check_quality', methods=['POST'])
def api_check_quality():
    """Check image quality before prediction."""
    try:
        if 'file' not in request.files:
            return {'ok': False, 'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        
        # Read image
        raw_bytes = file.read()
        arr = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return {'ok': False, 'error': 'Invalid image file'}, 400
        
        # Check quality
        from utils import check_image_quality
        is_good, quality_score, issues = check_image_quality(img_bgr)
        
        return {
            'ok': True,
            'quality': {
                'is_good': is_good,
                'score': round(quality_score, 1),
                'issues': issues,
                'recommendation': 'Image quality is good' if is_good else 'Please use a clearer image'
            },
            'image_info': {
                'width': img_bgr.shape[1],
                'height': img_bgr.shape[0],
                'size_kb': len(raw_bytes) / 1024
            }
        }
        
    except Exception as e:
        app.logger.exception("Error checking image quality")
        return {'ok': False, 'error': str(e)}, 500


@app.route('/api/detect_leaf', methods=['POST'])
def api_detect_leaf():
    """Detect and extract leaf region from image."""
    try:
        if 'file' not in request.files:
            return {'ok': False, 'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        
        # Read image
        raw_bytes = file.read()
        arr = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return {'ok': False, 'error': 'Invalid image file'}, 400
        
        # Detect leaf
        from utils import detect_leaf_region
        mask, bbox, leaf_img = detect_leaf_region(img_bgr)
        
        if bbox is None:
            return {
                'ok': True,
                'leaf_detected': False,
                'message': 'No leaf region detected'
            }
        
        # Draw bounding box on original image
        x, y, w, h = bbox
        img_with_bbox = img_bgr.copy()
        cv2.rectangle(img_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Encode images to base64
        import base64
        _, buffer_bbox = cv2.imencode('.jpg', img_with_bbox)
        _, buffer_leaf = cv2.imencode('.jpg', leaf_img)
        
        img_bbox_base64 = base64.b64encode(buffer_bbox).decode('utf-8')
        img_leaf_base64 = base64.b64encode(buffer_leaf).decode('utf-8')
        
        return {
            'ok': True,
            'leaf_detected': True,
            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
            'image_with_bbox': f'data:image/jpeg;base64,{img_bbox_base64}',
            'leaf_image': f'data:image/jpeg;base64,{img_leaf_base64}'
        }
        
    except Exception as e:
        app.logger.exception("Error detecting leaf")
        return {'ok': False, 'error': str(e)}, 500


@app.route('/webcam')
def webcam():
    """Real-time webcam disease detection page."""
    return render_template('webcam.html', 
                         models=MODELS, 
                         pipelines=PIPELINES.keys(),
                         default_model=DEFAULT_MODEL,
                         default_pipeline=DEFAULT_PIPELINE)


@app.route('/ar-feature')
def ar_feature():
    """Augmented Reality camera experience page."""
    return render_template(
        'ar_feature.html',
        models=MODELS,
        pipelines=PIPELINES.keys(),
        default_model=DEFAULT_MODEL,
        default_pipeline=DEFAULT_PIPELINE,
    )


@app.route('/api/webcam_predict', methods=['POST'])
def api_webcam_predict():
    """Real-time prediction endpoint for webcam frames."""
    try:
        # Get base64 image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return {'ok': False, 'error': 'No image data'}, 400
        
        import base64
        img_base64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_base64)
        
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return {'ok': False, 'error': 'Invalid image'}, 400
        
        # Optional: Assess image quality (non-blocking for webcam)
        image_quality = assess_image_quality_with_suggestions(img_bgr)
        
        model_name = data.get('model', DEFAULT_MODEL)
        pipeline_key = data.get('pipeline', DEFAULT_PIPELINE)
        
        # Load model
        model, model_class_names = load_model_by_name(model_name, pipeline_key)
        
        # Preprocess and predict
        pipeline_fn = PIPELINES[pipeline_key][0]  # Extract function from tuple
        img_processed = pipeline_fn(img_bgr)
        img_processed = cv2.resize(img_processed, (224, 224))
        img_array = np.expand_dims(img_processed.astype('float32'), axis=0)  # Already normalized by pipeline
        
        preds = model.predict(img_array, verbose=0)
        pred_class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_class_idx])
        pred_class = model_class_names[pred_class_idx]
        
        # Calculate severity
        severity = calculate_severity_assessment(
            model=model,
            img_array=img_array,
            predicted_label=pred_class,
            confidence=confidence,
            pred_index=pred_class_idx
        )
        
        # Get all class probabilities
        all_probs = {model_class_names[i]: float(preds[0][i]) for i in range(len(model_class_names))}
        
        return {
            'ok': True,
            'prediction': {
                'class': pred_class,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'severity': severity
            },
            'image_quality': image_quality
        }
        
    except Exception as e:
        app.logger.exception("Error in webcam prediction")
        return {'ok': False, 'error': str(e)}, 500


# ============= MESSENGER/CHAT SYSTEM =============

def get_conversation(user1_id, user2_id):
    """Lấy hoặc tạo conversation giữa 2 users."""
    try:
        conversations_file = BASE_DIR / 'data' / 'conversations.jsonl'
        conversations_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort IDs để đảm bảo consistency
        participant_ids = sorted([user1_id, user2_id])
        conv_id = f"{participant_ids[0]}_{participant_ids[1]}"
        
        # Tìm conversation hiện có
        if conversations_file.exists():
            with open(conversations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        conv = json.loads(line)
                        if conv['id'] == conv_id:
                            return conv
        
        # Tạo conversation mới
        conversation = {
            'id': conv_id,
            'participants': participant_ids,
            'created_at': datetime.now().isoformat(),
            'last_message_at': datetime.now().isoformat(),
            'unread_count': {participant_ids[0]: 0, participant_ids[1]: 0}
        }
        
        with open(conversations_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        return conversation
        
    except Exception as e:
        app.logger.exception('Error in get_conversation')
        return None

def get_user_conversations(user_id):
    """Lấy tất cả conversations của user."""
    try:
        conversations_file = BASE_DIR / 'data' / 'conversations.jsonl'
        if not conversations_file.exists():
            return []
        
        user_convs = []
        with open(conversations_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    if user_id in conv['participants']:
                        # Lấy thông tin user kia
                        other_user_id = [uid for uid in conv['participants'] if uid != user_id][0]
                        
                        # Xử lý trường hợp admin (không có trong users.jsonl)
                        if other_user_id == 'admin':
                            conv['other_user'] = {
                                'id': 'admin',
                                'name': 'Quản trị viên',
                                'email': ADMIN_USERNAME,
                                'is_admin': True
                            }
                        else:
                            other_user = get_user_by_id(other_user_id)
                            if other_user:
                                conv['other_user'] = {
                                    'id': other_user['id'],
                                    'name': other_user.get('full_name', 'Unknown'),
                                    'email': other_user.get('email', ''),
                                    'is_admin': False
                                }
                            else:
                                # Skip conversation nếu không tìm thấy user
                                continue
                        
                        # Lấy tin nhắn cuối
                        last_msg = get_last_message(conv['id'])
                        conv['last_message'] = last_msg
                        conv['unread_count_user'] = conv.get('unread_count', {}).get(user_id, 0)
                        
                        user_convs.append(conv)
        
        # Sắp xếp theo thời gian tin nhắn cuối
        user_convs.sort(key=lambda x: x.get('last_message_at', ''), reverse=True)
        return user_convs
        
    except Exception as e:
        app.logger.exception('Error in get_user_conversations')
        return []

def get_last_message(conversation_id):
    """Lấy tin nhắn cuối cùng của conversation."""
    try:
        messages_file = BASE_DIR / 'data' / 'messages.jsonl'
        if not messages_file.exists():
            return None
        
        last_msg = None
        with open(messages_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    msg = json.loads(line)
                    if msg['conversation_id'] == conversation_id:
                        last_msg = msg
        
        return last_msg
        
    except Exception as e:
        app.logger.exception('Error in get_last_message')
        return None

def get_conversation_messages(conversation_id, limit=100):
    """Lấy tin nhắn của conversation."""
    try:
        messages_file = BASE_DIR / 'data' / 'messages.jsonl'
        if not messages_file.exists():
            return []
        
        messages = []
        with open(messages_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    msg = json.loads(line)
                    if msg['conversation_id'] == conversation_id:
                        messages.append(msg)
        
        # Sắp xếp theo thời gian
        messages.sort(key=lambda x: x['created_at'])
        
        # Giới hạn số lượng
        return messages[-limit:]
        
    except Exception as e:
        app.logger.exception('Error in get_conversation_messages')
        return []

def send_message(conversation_id, sender_id, content):
    """Gửi tin nhắn."""
    try:
        messages_file = BASE_DIR / 'data' / 'messages.jsonl'
        conversations_file = BASE_DIR / 'data' / 'conversations.jsonl'
        messages_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo message
        message = {
            'id': str(uuid4()),
            'conversation_id': conversation_id,
            'sender_id': sender_id,
            'content': content,
            'created_at': datetime.now().isoformat(),
            'read': False
        }
        
        # Lưu message
        with open(messages_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(message, ensure_ascii=False) + '\n')
        
        # Cập nhật conversation
        conversations = []
        if conversations_file.exists():
            with open(conversations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        conv = json.loads(line)
                        if conv['id'] == conversation_id:
                            conv['last_message_at'] = datetime.now().isoformat()
                            # Tăng unread count cho người nhận
                            receiver_id = [uid for uid in conv['participants'] if uid != sender_id][0]
                            if 'unread_count' not in conv:
                                conv['unread_count'] = {}
                            conv['unread_count'][receiver_id] = conv['unread_count'].get(receiver_id, 0) + 1
                        conversations.append(conv)
            
            # Ghi lại toàn bộ conversations
            with open(conversations_file, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        return message
        
    except Exception as e:
        app.logger.exception('Error in send_message')
        return None

def mark_messages_as_read(conversation_id, user_id):
    """Đánh dấu tin nhắn đã đọc."""
    try:
        messages_file = BASE_DIR / 'data' / 'messages.jsonl'
        conversations_file = BASE_DIR / 'data' / 'conversations.jsonl'
        
        # Cập nhật messages
        if messages_file.exists():
            messages = []
            with open(messages_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        msg = json.loads(line)
                        if msg['conversation_id'] == conversation_id and msg['sender_id'] != user_id:
                            msg['read'] = True
                        messages.append(msg)
            
            with open(messages_file, 'w', encoding='utf-8') as f:
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + '\n')
        
        # Reset unread count trong conversation
        if conversations_file.exists():
            conversations = []
            with open(conversations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        conv = json.loads(line)
                        if conv['id'] == conversation_id:
                            if 'unread_count' not in conv:
                                conv['unread_count'] = {}
                            conv['unread_count'][user_id] = 0
                        conversations.append(conv)
            
            with open(conversations_file, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        return True
        
    except Exception as e:
        app.logger.exception('Error in mark_messages_as_read')
        return False

def get_all_users_for_chat():
    """Lấy danh sách tất cả users có thể chat (bao gồm admin)."""
    try:
        users = load_users()
        # Filter và format
        chat_users = []
        
        # Thêm admin vào đầu danh sách
        chat_users.append({
            'id': 'admin',
            'name': 'Quản trị viên',
            'email': ADMIN_USERNAME,
            'is_admin': True
        })
        
        for user in users:
            chat_users.append({
                'id': user['id'],
                'name': user.get('full_name', 'Unknown'),
                'email': user.get('email', ''),
                'is_admin': False
            })
        return chat_users
    except Exception as e:
        app.logger.exception('Error in get_all_users_for_chat')
        return []

@app.route('/messenger')
@login_required
def messenger():
    """Trang messenger chính."""
    user_id = session['user_id']
    conversations = get_user_conversations(user_id)
    
    # Lấy danh sách users để bắt đầu chat mới
    all_users = get_all_users_for_chat()
    # Lọc bỏ user hiện tại
    other_users = [u for u in all_users if u['id'] != user_id]
    
    # Tính tổng số tin nhắn chưa đọc
    total_unread = sum(conv.get('unread_count_user', 0) for conv in conversations)
    
    return render_template('messenger.html', 
                         conversations=conversations,
                         other_users=other_users,
                         total_unread=total_unread)

@app.route('/api/messenger/conversations', methods=['GET'])
@login_required
def api_get_conversations():
    """API lấy danh sách conversations."""
    user_id = session['user_id']
    conversations = get_user_conversations(user_id)
    return jsonify({'ok': True, 'conversations': conversations})

@app.route('/api/messenger/messages/<conversation_id>', methods=['GET'])
@login_required
def api_get_messages(conversation_id):
    """API lấy tin nhắn của conversation."""
    user_id = session['user_id']
    
    # Kiểm tra user có thuộc conversation không
    conversations_file = BASE_DIR / 'data' / 'conversations.jsonl'
    if conversations_file.exists():
        with open(conversations_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    if conv['id'] == conversation_id:
                        if user_id not in conv['participants']:
                            return jsonify({'ok': False, 'error': 'Unauthorized'}), 403
                        break
            else:
                return jsonify({'ok': False, 'error': 'Conversation not found'}), 404
    else:
        return jsonify({'ok': False, 'error': 'Conversation not found'}), 404
    
    # Đánh dấu đã đọc
    mark_messages_as_read(conversation_id, user_id)
    
    # Lấy tin nhắn
    messages = get_conversation_messages(conversation_id)
    
    # Thêm thông tin sender
    for msg in messages:
        sender_id = msg['sender_id']
        
        # Xử lý trường hợp admin
        if sender_id == 'admin':
            msg['sender_name'] = 'Quản trị viên'
            msg['sender_email'] = ADMIN_USERNAME
        else:
            sender = get_user_by_id(sender_id)
            if sender:
                msg['sender_name'] = sender.get('full_name', 'Unknown')
                msg['sender_email'] = sender.get('email', '')
            else:
                msg['sender_name'] = 'Unknown'
                msg['sender_email'] = ''
    
    return jsonify({'ok': True, 'messages': messages})

@app.route('/api/messenger/send', methods=['POST'])
@login_required
def api_send_message():
    """API gửi tin nhắn."""
    try:
        user_id = session['user_id']
        data = request.get_json()
        
        receiver_id = data.get('receiver_id')
        content = data.get('content', '').strip()
        
        if not receiver_id or not content:
            return jsonify({'ok': False, 'error': 'Missing data'}), 400
        
        # Xử lý trường hợp receiver là admin (không cần kiểm tra trong database)
        if receiver_id != 'admin':
            # Kiểm tra receiver có tồn tại
            receiver = get_user_by_id(receiver_id)
            if not receiver:
                return jsonify({'ok': False, 'error': 'Receiver not found'}), 404
        
        # Lấy hoặc tạo conversation
        conversation = get_conversation(user_id, receiver_id)
        if not conversation:
            return jsonify({'ok': False, 'error': 'Failed to create conversation'}), 500
        
        # Gửi tin nhắn
        message = send_message(conversation['id'], user_id, content)
        if not message:
            return jsonify({'ok': False, 'error': 'Failed to send message'}), 500
        
        # Thêm thông tin sender
        if user_id == 'admin':
            message['sender_name'] = 'Quản trị viên'
            message['sender_email'] = ADMIN_USERNAME
        else:
            sender = get_user_by_id(user_id)
            if sender:
                message['sender_name'] = sender.get('full_name', 'Unknown')
                message['sender_email'] = sender.get('email', '')
            else:
                message['sender_name'] = 'Unknown'
                message['sender_email'] = ''
        
        return jsonify({
            'ok': True, 
            'message': message,
            'conversation_id': conversation['id']
        })
        
    except Exception as e:
        app.logger.exception('Error in api_send_message')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/messenger/start_chat/<user_id>', methods=['POST'])
@login_required
def api_start_chat(user_id):
    """API bắt đầu chat với user."""
    try:
        current_user_id = session['user_id']
        
        # Xử lý trường hợp chat với admin
        if user_id == 'admin':
            other_user = {
                'id': 'admin',
                'full_name': 'Quản trị viên',
                'email': ADMIN_USERNAME
            }
        else:
            # Kiểm tra user có tồn tại
            other_user = get_user_by_id(user_id)
            if not other_user:
                return jsonify({'ok': False, 'error': 'User not found'}), 404
        
        # Lấy hoặc tạo conversation
        conversation = get_conversation(current_user_id, user_id)
        if not conversation:
            return jsonify({'ok': False, 'error': 'Failed to create conversation'}), 500
        
        return jsonify({
            'ok': True,
            'conversation_id': conversation['id'],
            'other_user': {
                'id': other_user['id'],
                'name': other_user.get('full_name', 'Unknown'),
                'email': other_user.get('email', ''),
                'is_admin': user_id == 'admin'
            }
        })
        
    except Exception as e:
        app.logger.exception('Error in api_start_chat')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/messenger/unread_count', methods=['GET'])
@login_required
def api_unread_count():
    """API lấy số tin nhắn chưa đọc."""
    try:
        user_id = session['user_id']
        conversations = get_user_conversations(user_id)
        total_unread = sum(conv.get('unread_count_user', 0) for conv in conversations)
        
        return jsonify({'ok': True, 'unread_count': total_unread})
        
    except Exception as e:
        app.logger.exception('Error in api_unread_count')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/messenger/contact_admin', methods=['POST'])
@login_required
def api_contact_admin():
    """API để liên hệ nhanh với admin."""
    try:
        user_id = session['user_id']
        
        # Admin trong hệ thống có user_id = 'admin' (không lưu trong users.jsonl)
        # Tạo admin virtual user
        admin_id = 'admin'
        admin_info = {
            'id': admin_id,
            'name': 'Quản trị viên',
            'email': ADMIN_USERNAME,
            'is_admin': True
        }
        
        # Kiểm tra không cho admin chat với chính mình
        if user_id == admin_id:
            return jsonify({'ok': False, 'error': 'Admin không thể chat với chính mình'}), 400
        
        # Lấy hoặc tạo conversation với admin
        conversation = get_conversation(user_id, admin_id)
        if not conversation:
            return jsonify({'ok': False, 'error': 'Không thể tạo cuộc trò chuyện'}), 500
        
        return jsonify({
            'ok': True,
            'conversation_id': conversation['id'],
            'admin': admin_info
        })
        
    except Exception as e:
        app.logger.exception('Error in api_contact_admin')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ============= FLASK ERROR HANDLERS =============

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    app.logger.warning("File upload too large: %s", request.content_length)
    flash(f"File quá lớn. Kích thước tối đa: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    app.logger.warning("404 error: %s", request.url)
    return render_template('404.html'), 404

# ============= PRELOAD & STARTUP =============

def preload():
    """Tải trước các tài nguyên cần thiết để tăng tốc độ cho lần yêu cầu đầu tiên."""
    try:
        app.logger.info("=" * 60)
        app.logger.info("Starting application preload...")
        
        # Load sample features cache
        app.logger.info("Loading sample features cache...")
        features = _load_sample_features_cache()
        if features:
            pos_count = len(features.get('positive', []))
            neg_count = len(features.get('negative', []))
            app.logger.info(
                "Sample features loaded - positive: %s, negative: %s",
                pos_count, neg_count
            )
        else:
            app.logger.warning("No sample features found")
        
        # Kiểm tra Gemini API
        app.logger.info("Checking Gemini API configuration...")
        if GEMINI_MODEL:
            app.logger.info("Gemini API configured successfully for chatbot")
        else:
            app.logger.warning("Gemini API key not found - chatbot will not work. Set GEMINI_API_KEY in .env file")
        
        # Preload default model
        app.logger.info("Preloading default model (VGG19 + average_hsv)...")
        try:
            load_model_by_name('VGG19', 'average_hsv')
            cache_stats = LOADED_MODELS.get_stats()
            app.logger.info(
                "Default model preloaded successfully. Cache: %d/%d models",
                cache_stats['size'], cache_stats['max_size']
            )
        except Exception as e:
            app.logger.error("Failed to preload default model: %s", str(e))
        
        app.logger.info("Application preload completed successfully")
        app.logger.info("=" * 60)
        
    except Exception as e:
        app.logger.exception("Error during preload: %s", str(e))
        app.logger.warning("Application will continue despite preload errors")

if __name__ == "__main__":
    try:
        preload()
        
        # HTTPS Configuration
        use_https = os.environ.get('USE_HTTPS', 'false').lower() == 'true'
        ssl_context = None
        protocol = 'http'
        
        if use_https:
            # Kiểm tra xem có certificate files không
            cert_path = os.environ.get('SSL_CERT_PATH', 'certs/cert.pem')
            key_path = os.environ.get('SSL_KEY_PATH', 'certs/key.pem')
            
            cert_file = BASE_DIR / cert_path
            key_file = BASE_DIR / key_path
            
            if cert_file.exists() and key_file.exists():
                ssl_context = (str(cert_file), str(key_file))
                protocol = 'https'
                app.logger.info("[OK] HTTPS enabled with certificates:")
                app.logger.info(f"  Certificate: {cert_file}")
                app.logger.info(f"  Private Key: {key_file}")
            else:
                # Fallback: Sử dụng adhoc SSL (tự động tạo self-signed cert)
                try:
                    ssl_context = 'adhoc'
                    protocol = 'https'
                    app.logger.warning("Certificate files not found. Using adhoc SSL (auto-generated self-signed certificate)")
                    app.logger.warning("[WARNING] For production, please use proper SSL certificates!")
                    app.logger.info("To generate certificate files, run:")
                    app.logger.info("  mkdir certs")
                    app.logger.info("  openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj '/CN=localhost'")
                except Exception as e:
                    app.logger.error(f"Failed to create adhoc SSL context: {e}")
                    app.logger.info("Falling back to HTTP mode")
                    use_https = False
                    ssl_context = None
        
        # Auto-open browser after server starts (can be disabled via env var)
        # Only open on the reloader process (not on the parent process)
        auto_open = os.environ.get('AUTO_OPEN_BROWSER', 'true').lower() == 'true'
        is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        
        if auto_open and is_reloader:
            import webbrowser
            from threading import Timer
            
            def open_browser():
                """Open browser after a short delay to ensure server is ready"""
                url = f"{protocol}://localhost:5000"
                app.logger.info(f"Opening browser at {url}")
                try:
                    webbrowser.open(url)
                except Exception as e:
                    app.logger.warning(f"Could not open browser automatically: {e}")
            
            # Open browser after 1.5 seconds (give server time to start)
            Timer(1.5, open_browser).start()
            app.logger.info("Browser will open automatically in 1.5 seconds...")
            app.logger.info("(To disable: set AUTO_OPEN_BROWSER=false in .env)")
        
        app.logger.info("Starting Flask development server...")
        app.logger.info(f"Server running at {protocol}://0.0.0.0:5000")
        
        if use_https:
            app.logger.info("=" * 60)
            app.logger.info("[HTTPS] HTTPS MODE ENABLED")
            app.logger.info("=" * 60)
            if ssl_context == 'adhoc':
                app.logger.warning("[WARNING] Using self-signed certificate - browsers will show security warning")
                app.logger.info("Click 'Advanced' -> 'Proceed to localhost' to continue")
        
        # Khi chạy với VS Code debugger, không dùng Flask debug mode (gây conflict)
        # VS Code debugger sẽ tự động enable debugging
        debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
        use_reloader = os.environ.get('WERKZEUG_RUN_MAIN') != 'true'  # Disable reloader trong debugger
        
        app.run(host='0.0.0.0', port=5000, debug=debug_mode, use_reloader=False, ssl_context=ssl_context)
    except KeyboardInterrupt:
        app.logger.info("Server stopped by user")
    except Exception as e:
        app.logger.exception("Failed to start server: %s", str(e))
        sys.exit(1)
