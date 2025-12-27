import pickle
from pathlib import Path
import cv2
import logging
from tqdm import tqdm
from utils import compute_embedding, compute_hist

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Xử lý trước các ảnh mẫu (positive và negative) để trích xuất đặc trưng
    và lưu chúng vào một file pickle để tăng tốc độ kiểm tra tương đồng.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent  # Lên 1 cấp để đến thư mục gốc
    positive_samples_dir = BASE_DIR / 'static' / 'images' / 'tomato_samples'
    negative_samples_dir = BASE_DIR / 'static' / 'images' / 'not_tomato_samples'
    output_file = BASE_DIR / 'data' / 'sample_features.pkl'

    # Đảm bảo thư mục data tồn tại
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_features = {'positive': [], 'negative': []}

    def process_directory(directory, category):
        if not directory.exists():
            logger.warning("Thư mục không tồn tại: %s", directory)
            return
        
        logger.info("Đang xử lý thư mục %s...", category)
        image_files = [p for p in directory.iterdir() 
                      if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        for p in tqdm(image_files, desc=f"Processing {category} samples"):
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("Không thể đọc ảnh: %s", p)
                continue
            
            hist = compute_hist(img)
            embedding = compute_embedding(img)
            if embedding is not None:
                all_features[category].append({'path': str(p), 'hist': hist, 'embedding': embedding})

    process_directory(positive_samples_dir, 'positive')
    process_directory(negative_samples_dir, 'negative')

    with open(output_file, 'wb') as f:
        pickle.dump(all_features, f)
    
    logger.info("\nĐã xử lý xong! Dữ liệu đặc trưng đã được lưu vào: %s", output_file)

if __name__ == '__main__':
    main()