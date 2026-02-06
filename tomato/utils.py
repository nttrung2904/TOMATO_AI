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