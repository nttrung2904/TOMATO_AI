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


# ==================== COMPUTER VISION ENHANCEMENTS ====================

def generate_gradcam(model, img_array, pred_index=None, layer_name=None):
    """
    Generate Grad-CAM heatmap for model prediction.
    
    Args:
        model: Keras model
        img_array: Preprocessed image array (batch_size, height, width, channels)
        pred_index: Index of predicted class (default: argmax of predictions)
        layer_name: Name of target conv layer (default: last conv layer)
    
    Returns:
        heatmap: numpy array (height, width) with values in [0, 1]
    """
    try:
        # Find last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                # Check if layer is convolutional by checking class name or output shape
                layer_class_name = layer.__class__.__name__
                if 'Conv' in layer_class_name:
                    layer_name = layer.name
                    break
                # Fallback: check if has 4D output (but safely)
                elif hasattr(layer, 'output_shape') and layer.output_shape is not None:
                    try:
                        if len(layer.output_shape) == 4:
                            layer_name = layer.name
                            break
                    except (TypeError, AttributeError):
                        continue
        
        if layer_name is None:
            logging.getLogger(__name__).warning("No convolutional layer found for Grad-CAM")
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Gradient of class output w.r.t. feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        return heatmap
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error generating Grad-CAM: %s", e)
        return None


def overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        img_bgr: Original image in BGR format
        heatmap: Grad-CAM heatmap (height, width) with values in [0, 1]
        alpha: Transparency of heatmap overlay (0-1)
        colormap: OpenCV colormap for heatmap visualization
    
    Returns:
        superimposed_img: BGR image with heatmap overlay
    """
    try:
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        return superimposed
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error overlaying heatmap: %s", e)
        return img_bgr


def enhance_image_quality(img_bgr, denoise=True, sharpen=True, adjust_brightness=True):
    """
    Enhance image quality with various preprocessing techniques.
    
    Args:
        img_bgr: Input image in BGR format
        denoise: Apply denoising
        sharpen: Apply sharpening
        adjust_brightness: Auto-adjust brightness/contrast
    
    Returns:
        enhanced_img: Enhanced BGR image
    """
    try:
        enhanced = img_bgr.copy()
        
        # Denoise
        if denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Auto brightness/contrast adjustment
        if adjust_brightness:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        if sharpen:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error enhancing image: %s", e)
        return img_bgr


def check_image_quality(img_bgr, min_size=100, max_blur_threshold=100):
    """
    Check if image quality is sufficient for prediction.
    
    Args:
        img_bgr: Input image in BGR format
        min_size: Minimum image dimension
        max_blur_threshold: Maximum Laplacian variance (higher = less blurry)
    
    Returns:
        is_good: Boolean indicating if quality is sufficient
        quality_score: Float score 0-100
        issues: List of quality issues
    """
    issues = []
    quality_score = 100.0
    
    try:
        # Check size
        h, w = img_bgr.shape[:2]
        if h < min_size or w < min_size:
            issues.append(f"Image too small ({w}x{h}, minimum: {min_size}x{min_size})")
            quality_score -= 30
        
        # Check blur (Laplacian variance)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < max_blur_threshold:
            blur_severity = (max_blur_threshold - laplacian_var) / max_blur_threshold
            issues.append(f"Image is blurry (score: {laplacian_var:.1f})")
            quality_score -= blur_severity * 40
        
        # Check brightness
        mean_brightness = gray.mean()
        if mean_brightness < 40:
            issues.append("Image too dark")
            quality_score -= 20
        elif mean_brightness > 220:
            issues.append("Image too bright")
            quality_score -= 15
        
        # Check contrast
        contrast = gray.std()
        if contrast < 20:
            issues.append("Low contrast")
            quality_score -= 10
        
        is_good = quality_score >= 50
        
        return is_good, max(0, quality_score), issues
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error checking image quality: %s", e)
        return True, 50.0, ["Could not assess quality"]


def detect_leaf_region(img_bgr):
    """
    Detect and extract leaf region from image using color segmentation.
    
    Args:
        img_bgr: Input image in BGR format
    
    Returns:
        mask: Binary mask of leaf region
        bbox: Bounding box (x, y, w, h) of detected leaf
        leaf_img: Cropped leaf region
    """
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Green color range for leaves
        lower_green1 = np.array([25, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find largest contour (main leaf)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(largest_contour)
        x, y, w, h = bbox
        
        # Crop leaf region
        leaf_img = img_bgr[y:y+h, x:x+w]
        
        return mask, bbox, leaf_img
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error detecting leaf region: %s", e)
        return None, None, None


# ==================== SEVERITY ASSESSMENT ====================

def calculate_affected_area_from_heatmap(heatmap, threshold=0.5):
    """
    Calculate percentage of affected area from Grad-CAM heatmap.
    
    Args:
        heatmap: Grad-CAM heatmap (height, width) with values in [0, 1]
        threshold: Threshold for considering area as "affected" (default: 0.5)
    
    Returns:
        affected_percentage: Percentage of area above threshold (0-100)
        affected_pixels: Number of pixels above threshold
        total_pixels: Total number of pixels
    """
    try:
        if heatmap is None or heatmap.size == 0:
            return 0.0, 0, 0
        
        # Count pixels above threshold (indicating disease presence)
        affected_pixels = np.sum(heatmap >= threshold)
        total_pixels = heatmap.size
        
        affected_percentage = (affected_pixels / total_pixels) * 100.0
        
        return affected_percentage, int(affected_pixels), int(total_pixels)
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error calculating affected area: %s", e)
        return 0.0, 0, 0


def assess_disease_severity(predicted_label, confidence, affected_area_pct, heatmap=None):
    """
    Assess disease severity based on multiple factors.
    
    Args:
        predicted_label: Predicted disease class
        confidence: Model confidence (0-1)
        affected_area_pct: Percentage of affected area (0-100)
        heatmap: Optional Grad-CAM heatmap for additional analysis
    
    Returns:
        severity_dict: Dictionary containing:
            - level: 'Healthy', 'Mild', 'Moderate', 'Severe', 'Critical'
            - score: Numeric score (0-100)
            - affected_area: Percentage of affected area
            - confidence: Model confidence
            - description: Human-readable description
            - recommendations: List of recommended actions
            - color: CSS color for visualization
    """
    try:
        # Base severity score
        severity_score = 0.0
        
        # Check if healthy
        if predicted_label.lower() in ['healthy', 'tomato_healthy']:
            return {
                'level': 'Healthy',
                'score': 0.0,
                'affected_area': 0.0,
                'confidence': confidence,
                'description': 'Lá cà chua khỏe mạnh, không có dấu hiệu bệnh.',
                'recommendations': [
                    'Tiếp tục chăm sóc cây theo quy trình thông thường',
                    'Giám sát định kỳ để phát hiện sớm bệnh',
                    'Duy trì độ ẩm và dinh dưỡng đầy đủ'
                ],
                'color': '#4caf50',
                'icon': '✅'
            }
        
        # Disease-specific severity weights
        disease_weights = {
            'early_blight': 1.2,      # Moderate severity disease
            'late_blight': 1.5,       # High severity - spreads rapidly
            'leaf_mold': 1.0,         # Moderate severity
            'septoria_leaf_spot': 1.1,
            'spider_mites': 1.3,
            'target_spot': 1.1,
            'yellow_leaf_curl_virus': 1.4,
            'mosaic_virus': 1.3,
            'bacterial_spot': 1.2,
        }
        
        # Get disease weight (default 1.0 for unknown diseases)
        disease_key = predicted_label.lower().replace(' ', '_')
        disease_weight = disease_weights.get(disease_key, 1.0)
        
        # Calculate severity score from affected area (0-40 points)
        area_score = min(affected_area_pct * 0.4, 40)
        
        # Add confidence factor (0-30 points) - higher confidence = more reliable severe assessment
        confidence_score = confidence * 30
        
        # Add disease type factor (0-30 points)
        disease_score = disease_weight * 20
        
        # Total severity score (0-100)
        severity_score = area_score + confidence_score + disease_score
        
        # Determine severity level
        if severity_score < 20:
            level = 'Mild'
            color = '#ffc107'
            icon = '⚠️'
            description = 'Bệnh ở giai đoạn đầu, mức độ nhẹ.'
            recommendations = [
                'Loại bỏ lá bệnh để tránh lây lan',
                'Theo dõi sát sao trong 5-7 ngày tới',
                'Cải thiện thông gió và giảm độ ẩm',
                'Xem xét sử dụng thuốc phòng bệnh sinh học'
            ]
        elif severity_score < 40:
            level = 'Moderate'
            color = '#ff9800'
            icon = '⚠️'
            description = 'Bệnh ở mức độ trung bình, cần xử lý sớm.'
            recommendations = [
                'Xử lý ngay bằng thuốc chuyên dụng phù hợp',
                'Loại bỏ và tiêu hủy tất cả lá bệnh',
                'Tăng cường thông gió cho cây',
                'Tránh tưới nước vào buổi tối',
                'Cách ly cây bệnh nếu có thể'
            ]
        elif severity_score < 60:
            level = 'Severe'
            color = '#f44336'
            icon = '🔴'
            description = 'Bệnh ở mức độ nghiêm trọng, cần xử lý khẩn cấp.'
            recommendations = [
                '🚨 Xử lý ngay lập tức bằng thuốc hóa học mạnh',
                'Loại bỏ và đốt tất cả bộ phận bị nhiễm',
                'Phun thuốc định kỳ theo hướng dẫn',
                'Cách ly hoàn toàn cây bệnh khỏi vườn',
                'Khử trùng dụng cụ làm vườn',
                'Xem xét tư vấn chuyên gia nông nghiệp'
            ]
        else:
            level = 'Critical'
            color = '#b71c1c'
            icon = '🔴'
            description = 'Bệnh ở mức độ rất nghiêm trọng, nguy cơ mất cây cao.'
            recommendations = [
                '🚨🚨 KHẨN CẤP: Liên hệ chuyên gia nông nghiệp ngay',
                'Xem xét nhổ bỏ cây để tránh lây lan toàn vườn',
                'Cách ly hoàn toàn khu vực bị nhiễm',
                'Sử dụng thuốc hóa học mạnh theo chỉ dẫn chuyên gia',
                'Khử trùng đất và dụng cụ kỹ lưỡng',
                'Không trồng cà chua ở khu vực này trong 6-12 tháng'
            ]
        
        return {
            'level': level,
            'score': round(severity_score, 2),
            'affected_area': round(affected_area_pct, 2),
            'confidence': round(confidence * 100, 2),
            'description': description,
            'recommendations': recommendations,
            'color': color,
            'icon': icon
        }
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error assessing disease severity: %s", e)
        return {
            'level': 'Unknown',
            'score': 0.0,
            'affected_area': 0.0,
            'confidence': 0.0,
            'description': 'Không thể đánh giá mức độ nghiêm trọng.',
            'recommendations': ['Vui lòng thử lại hoặc liên hệ hỗ trợ'],
            'color': '#9e9e9e',
            'icon': '❓'
        }


def calculate_severity_from_prediction(model, img_array, predicted_label, confidence, pred_index=None):
    """
    Calculate disease severity by combining Grad-CAM analysis with prediction results.
    
    Args:
        model: Keras model used for prediction
        img_array: Preprocessed image array
        predicted_label: Predicted disease class
        confidence: Model confidence (0-1)
        pred_index: Index of predicted class
    
    Returns:
        severity_dict: Complete severity assessment dictionary
    """
    try:
        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam(model, img_array, pred_index=pred_index)
        
        # Calculate affected area from heatmap
        if heatmap is not None:
            affected_area_pct, _, _ = calculate_affected_area_from_heatmap(heatmap, threshold=0.5)
        else:
            # Fallback: estimate from confidence if Grad-CAM fails
            affected_area_pct = confidence * 50  # Conservative estimate
        
        # Assess overall severity
        severity = assess_disease_severity(
            predicted_label=predicted_label,
            confidence=confidence,
            affected_area_pct=affected_area_pct,
            heatmap=heatmap
        )
        
        return severity
        
    except Exception as e:
        logging.getLogger(__name__).exception("Error calculating severity from prediction: %s", e)
        # Return default severity on error
        return assess_disease_severity(predicted_label, confidence, 0.0)