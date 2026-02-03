import io
import numpy as np
import pickle
import os
import cv2
import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session, make_response
from functools import wraps
import tensorflow as tf
import shutil 
from PIL import Image
from utils import compute_hist, compute_embedding as _compute_embedding
from datetime import datetime, timedelta
import threading
from uuid import uuid4
import logging
from logging.handlers import RotatingFileHandler
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from collections import Counter

# Load biến môi trường từ file .env
load_dotenv()

# Cấu hình Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Thử các model theo thứ tự: model miễn phí ổn định nhất trước
    model_names = [
        'models/gemini-2.5-flash',      # Model ổn định, miễn phí tốt
        'models/gemini-flash-latest',    # Luôn dùng phiên bản mới nhất
        'models/gemini-2.0-flash',       # Backup
    ]
    for model_name in model_names:
        try:
            GEMINI_MODEL = genai.GenerativeModel(model_name)
            print(f"✓ Gemini API configured with model: {model_name}")
            break
        except Exception as e:
            print(f"✗ Model {model_name} failed: {e}")
            continue
    
    if not GEMINI_MODEL:
        print("✗ Failed to configure any Gemini model")

# Lưu ý: Các tiện ích trích xuất đặc trưng sâu được cung cấp bởi `utils.py` (get_feature_extractor,
# compute_embedding). Tránh duplicate MobileNetV2/preprocess_input ở đây để ngăn
# import không nhất quán và khởi tạo trùng lặp.

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
    
    # Console handler với output màu cho development
    # Đặt UTF-8 encoding cho Windows console để xử lý ký tự tiếng Việt
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
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
    """Decorator to require HTTP Basic Auth for admin routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            app.logger.warning(
                "Unauthorized admin access attempt from %s to %s",
                request.remote_addr, request.path
            )
            return authenticate()
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
# Cache cho chatbot
# Không cần CHAT_DATASET nữa vì dùng Gemini API

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

# Khởi tạo model cache
LOADED_MODELS = ModelLRUCache(max_size=MAX_LOADED_MODELS)
MODEL_LOAD_LOCK = threading.Lock()

def get_gemini_response(user_question: str) -> str:
    """Gọi API Gemini để trả lời câu hỏi về cà chua.
    
    Args:
        user_question: Câu hỏi của người dùng
        
    Returns:
        Câu trả lời từ Gemini hoặc thông báo lỗi
    """
    if not GEMINI_MODEL:
        return "Hệ thống chatbot chưa được cấu hình. Vui lòng liên hệ quản trị viên để được hỗ trợ."
    
    try:
        # Tạo prompt tự nhiên, yêu cầu câu trả lời hoàn chỉnh
        system_prompt = """Bạn là chuyên gia cà chua. Trả lời HOÀN CHỈNH, TỰ NHIÊN bằng tiếng Việt.

Nếu hỏi về cà chua: Giải thích rõ ràng, cụ thể, đầy đủ (3-5 câu).
Nếu KHÔNG về cà chua: "Xin lỗi, tôi chỉ trả lời về cà chua."

QUAN TRỌNG: 
- Trả lời ĐẦY ĐỦ, KHÔNG bỏ dở giữa chừng
- Dùng ngôn ngữ đời thường, dễ hiểu
- Đi thẳng vào nội dung
- Kết thúc câu trả lời một cách hoàn chỉnh"""
        
        # Gọi API Gemini với cấu hình tối ưu
        full_prompt = f"{system_prompt}\n\nCâu hỏi: {user_question}\n\nTrả lời:"
        response = GEMINI_MODEL.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=800,  # Tăng lên 800 để đảm bảo đủ cho câu trả lời hoàn chỉnh
                top_p=0.9,
                candidate_count=1,  # Chỉ lấy 1 candidate để tránh nhầm lẫn
            )
        )
        
        if response and response.candidates:
            candidate = response.candidates[0]
            
            # Kiểm tra lý do kết thúc để phát hiện câu trả lời bị cắt
            finish_reason = candidate.finish_reason
            app.logger.info(f"Gemini finish_reason: {finish_reason}")
            
            # finish_reason có thể là: STOP (hoàn thành), MAX_TOKENS (bị cắt), SAFETY, OTHER
            if finish_reason and finish_reason.name != 'STOP':
                app.logger.warning(f"Response may be truncated. Finish reason: {finish_reason.name}")
                if finish_reason.name == 'MAX_TOKENS':
                    # Nếu bị cắt do MAX_TOKENS, gọi lại với token cao hơn
                    app.logger.info("Retrying with higher token limit...")
                    response = GEMINI_MODEL.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=1500,
                            top_p=0.9,
                            candidate_count=1,
                        )
                    )
            
            if response and response.text:
                answer = response.text.strip()
                # Kiểm tra xem câu trả lời có bị cắt giữa chừng không
                if len(answer) < 20:
                    app.logger.warning(f"Response too short: {answer}")
                    return "Câu trả lời chưa đầy đủ. Vui lòng hỏi lại câu hỏi của bạn."
                return answer
        
        app.logger.warning(f"Gemini response empty or blocked. Response: {response}")
        return "Xin lỗi, tôi không thể tạo câu trả lời lúc này. Vui lòng thử lại."
            
    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"Lỗi khi gọi Gemini API: {error_msg}", exc_info=True)
        
        # Xử lý các lỗi cụ thể
        if "429" in error_msg or "quota" in error_msg.lower():
            return ("⚠️ Hệ thống chatbot tạm thời quá tải (đã hết quota miễn phí trong ngày). "
                   "Vui lòng thử lại sau hoặc liên hệ quản trị viên để nâng cấp API key.")
        elif "401" in error_msg or "API key" in error_msg:
            return "🔒 API key không hợp lệ. Vui lòng liên hệ quản trị viên."
        else:
            return "Đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."

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
        # Sắp xếp theo thời gian mới nhất trước
        history_list.reverse()
        return render_template('history.html', history=history_list)
    except Exception as e:
        app.logger.exception('Error loading history')
        flash('Không thể tải lịch sử dự đoán')
        return redirect(url_for('index'))

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Xóa toàn bộ lịch sử dự đoán"""
    try:
        history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
        if history_file.exists():
            history_file.unlink()
            app.logger.info('Cleared prediction history')
            flash('Đã xóa toàn bộ lịch sử dự đoán', 'success')
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
        
        # Tìm prediction theo ID
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('id') == prediction_id:
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
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Find prediction
        history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
        if not history_file.exists():
            flash('Không tìm thấy lịch sử dự đoán', 'error')
            return redirect(url_for('history'))
        
        prediction = None
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('id') == prediction_id:
                        prediction = entry
                        break
                except json.JSONDecodeError:
                    continue
        
        if not prediction:
            flash('Không tìm thấy dự đoán này', 'error')
            return redirect(url_for('history'))
        
        # Get disease info
        label = prediction.get('predicted_label', '')
        disease_info = DISEASE_INFO.get(label, {
            'name': label,
            'definition': 'Không có thông tin',
            'prevention': []
        })
        
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
            alignment=1  # Center
        )
        elements.append(Paragraph('BAO CAO DU DOAN BENH CA CHUA', title_style))
        elements.append(Spacer(1, 10*mm))
        
        # Prediction info table
        try:
            ts = datetime.fromisoformat(prediction.get('timestamp', ''))
            formatted_time = ts.strftime('%d/%m/%Y %H:%M:%S')
        except:
            formatted_time = prediction.get('timestamp', 'N/A')
        
        info_data = [
            ['Ma du doan:', prediction.get('id', 'N/A')],
            ['Thoi gian:', formatted_time],
            ['Model:', prediction.get('model_name', 'N/A')],
            ['Pipeline:', prediction.get('pipeline_key', 'N/A')],
            ['Benh phat hien:', disease_info['name']],
            ['Do tin cay:', f"{prediction.get('probability', 0) * 100:.1f}%"],
        ]
        
        info_table = Table(info_data, colWidths=[50*mm, 100*mm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 10*mm))
        
        # Disease definition
        elements.append(Paragraph('THONG TIN BENH:', styles['Heading2']))
        elements.append(Spacer(1, 3*mm))
        elements.append(Paragraph(disease_info['definition'], styles['BodyText']))
        elements.append(Spacer(1, 5*mm))
        
        # Prevention measures
        if disease_info.get('prevention'):
            elements.append(Paragraph('BIEN PHAP PHONG NGUA:', styles['Heading2']))
            elements.append(Spacer(1, 3*mm))
            for i, measure in enumerate(disease_info['prevention'], 1):
                elements.append(Paragraph(f"{i}. {measure}", styles['BodyText']))
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
                elements.append(Paragraph('ANH DA XU LY:', styles['Heading2']))
                elements.append(Spacer(1, 3*mm))
                img = RLImage(str(img_file), width=100*mm, height=100*mm)
                elements.append(img)
        
        # Footer
        elements.append(Spacer(1, 10*mm))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1
        )
        elements.append(Paragraph('Bao cao duoc tao tu dong boi Tomato AI System', footer_style))
        elements.append(Paragraph(f'Ngay xuat: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', footer_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        filename = f"tomato_report_{prediction_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        app.logger.info(f"Exported PDF report for prediction {prediction_id}")
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')
        
    except ImportError:
        app.logger.error('ReportLab not installed')
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

@app.route('/chat')
def chat():
    """Render chat page"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chatbot sử dụng Gemini API để trả lời câu hỏi về cà chua."""
    try:
        data = request.get_json(force=True)
        user_q_raw = (data.get('q') or '').strip()
    except Exception:
        return {"answer": "Không nhận được câu hỏi."}

    if not user_q_raw:
        return {"answer": "Vui lòng nhập câu hỏi."}

    # Gọi Gemini API để lấy câu trả lời
    app.logger.info(f"Chatbot received question: '{user_q_raw}'")
    answer = get_gemini_response(user_q_raw)
    app.logger.info(f"Gemini response generated successfully")

    # Ghi log
    try:
        logs_dir = BASE_DIR / 'data'
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / 'chat_logs.jsonl'
        entry = {
            'ts': datetime.utcnow().isoformat(),
            'question': user_q_raw,
            'answer': answer
        }
        with open(log_file, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception:
        app.logger.exception('Không thể ghi log chat')

    return {"answer": answer}

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
        
        model_name = request.form.get('model_select')
        pipeline_key = request.form.get('pipeline_select')
        
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
    
    app.logger.info("Prediction successful")
    return {
        'model': model,
        'class_names': model_class_names,
        'predictions': preds,
        'preprocessed': x
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
            'model_name': prediction_data.get('model_name'),
            'pipeline_key': prediction_data.get('pipeline_key'),
            'predicted_label': prediction_data.get('predicted_label'),
            'probability': float(prediction_data.get('probability', 0)),
            'possibly_not_tomato': prediction_data.get('possibly_not_tomato', False),
            'rejected': prediction_data.get('rejected', False),
            'image_path': prediction_data.get('image_path')
        }
        
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        app.logger.info(f"Saved prediction history: {prediction_id}")
        return prediction_id
    except Exception:
        app.logger.exception("Error saving prediction history")
        return None

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
                
                # Assess quality
                quality = assess_prediction_quality(img_bgr, pred_results['probability'])
                
                # Save to history
                prediction_id = save_prediction_history({
                    'model_name': model_name,
                    'pipeline_key': pipeline_key,
                    'predicted_label': pred_results['label'],
                    'probability': pred_results['probability'],
                    'possibly_not_tomato': quality['possibly_not_tomato'],
                    'rejected': quality['rejected_not_tomato'],
                    'image_path': image_path
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
                    'all_probs': pred_results['all_probs']
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
        
        # Bước 8: Đánh giá chất lượng dự đoán
        app.logger.info("[%s] Step 8: Assessing prediction quality", request_id)
        quality = assess_prediction_quality(img_bgr, results['probability'])
        
        # Bước 9: Lưu lịch sử dự đoán
        app.logger.info("[%s] Step 9: Saving prediction history", request_id)
        prediction_id = save_prediction_history({
            'model_name': params['model_name'],
            'pipeline_key': params['pipeline_key'],
            'predicted_label': results['label'],
            'probability': results['probability'],
            'possibly_not_tomato': quality['possibly_not_tomato'],
            'rejected': quality['rejected_not_tomato'],
            'image_path': image_path
        })
        
        # Step 10: Render result
        app.logger.info(
            "[%s] Step 10: Rendering result (quality flags - possibly_not_tomato: %s, rejected: %s)",
            request_id, quality['possibly_not_tomato'], quality['rejected_not_tomato']
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
            prediction_id=prediction_id
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

@app.route('/debug_preprocess', methods=['POST'])
def debug_preprocess():
    """Debug endpoint: accept an uploaded file and pipeline key, run preprocessing and
    return JSON with shape/dtype/min/max and a small base64 preview for inspection.
    Useful to reproduce channel/shape issues without loading the model.
    """
    if 'file' not in request.files:
        return {"ok": False, "message": "No file provided"}, 400
    f = request.files['file']
    pipeline_key = request.form.get('pipeline') or request.args.get('pipeline')
    if not pipeline_key:
        return {"ok": False, "message": "Missing pipeline key"}, 400
    raw = f.read()
    import base64
    import io as _io
    try:
        arr = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"ok": False, "message": "Cannot decode image"}, 400
        # Run preprocessing
        try:
            x = preprocess_image_for_model(img_bgr, pipeline_key)
        except Exception as e:
            return {"ok": False, "message": f"Preprocess error: {e}"}, 500

        # Prepare preview: take first image, convert to uint8 preview
        preview = None
        try:
            img = x[0]
            # clamp and scale to 0-255
            p = np.clip(img, 0.0, 1.0)
            p = (p * 255.0).astype('uint8')
            # if single channel, expand
            if p.ndim == 2:
                p = np.stack([p]*3, axis=-1)
            if p.shape[2] == 1:
                p = np.concatenate([p, p, p], axis=2)
            # encode to PNG
            ok, buf = cv2.imencode('.png', cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
            if ok:
                preview = base64.b64encode(buf.tobytes()).decode('ascii')
        except Exception:
            preview = None

        info = {
            "ok": True,
            "shape": x.shape,
            "dtype": str(x.dtype),
            "min": float(x.min()),
            "max": float(x.max()),
            "preview_png_base64": preview
        }
        return info
    except Exception as e:
        app.logger.exception('Error in debug_preprocess')
        return {"ok": False, "message": str(e)}, 500

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
                app.logger.info("✓ HTTPS enabled with certificates:")
                app.logger.info(f"  Certificate: {cert_file}")
                app.logger.info(f"  Private Key: {key_file}")
            else:
                # Fallback: Sử dụng adhoc SSL (tự động tạo self-signed cert)
                try:
                    ssl_context = 'adhoc'
                    protocol = 'https'
                    app.logger.warning("Certificate files not found. Using adhoc SSL (auto-generated self-signed certificate)")
                    app.logger.warning("⚠️  For production, please use proper SSL certificates!")
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
            app.logger.info("🔒 HTTPS MODE ENABLED")
            app.logger.info("=" * 60)
            if ssl_context == 'adhoc':
                app.logger.warning("⚠️  Using self-signed certificate - browsers will show security warning")
                app.logger.info("Click 'Advanced' → 'Proceed to localhost' to continue")
        
        app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=ssl_context)
    except KeyboardInterrupt:
        app.logger.info("Server stopped by user")
    except Exception as e:
        app.logger.exception("Failed to start server: %s", str(e))
        sys.exit(1)
