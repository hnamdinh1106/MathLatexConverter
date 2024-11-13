import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
import pytesseract
from transformers import pipeline
import torch
from ultralytics import YOLO

class AdvancedMathOCR:
    def __init__(self):
        # Mô hình nhận dạng vùng toán học
        self.yolo_detector = self.load_yolo_model()
        
        # Mô hình CNN chuyên toán
        self.math_cnn = self.build_math_cnn()
        
        # Mô hình transformer
        self.transformer_model = self.load_transformer_model()
        
        # Mô hình xử lý hình học
        self.geometry_model = self.build_geometry_model()
    
    def load_yolo_model(self):
        """Tải mô hình YOLO để detect vùng toán học"""
        model = YOLO('yolov8_math_detection.pt')
        return model
    
    def build_math_cnn(self):
        """Xây dựng mô hình CNN chuyên biệt toán học"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
    
    def load_transformer_model(self):
        """Tải mô hình transformer cho chuyển đổi LaTeX"""
        model = pipeline('text-to-text', 
                         model='facebook/math-mathbert')
        return model
    
    def build_geometry_model(self):
        """Mô hình chuyên về xử lý toán hình học"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(5, activation='softmax')  # 5 loại hình học cơ bản
        ])
        return model
    
    def preprocess_image(self, image_path):
        """Tiền xử lý ảnh tối ưu"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0  # Chuẩn hóa
        return image
    
    def detect_math_regions(self, image_path):
        """Phát hiện vùng toán học"""
        results = self.yolo_detector(image_path)
        return results
    
    def ocr_extraction(self, image_path):
        """Trích xuất văn bản bằng Tesseract"""
        text = pytesseract.image_to_string(image_path)
        return text
    
    def convert_to_latex(self, text):
        """Chuyển đổi sang LaTeX"""
        latex_output = self.transformer_model(text)[0]['generated_text']
        return latex_output
    
    def ensemble_prediction(self, image_path):
        """Kết hợp nhiều mô hình"""
        # Phát hiện vùng
        regions = self.detect_math_regions(image_path)
        
        # Xử lý từng vùng
        results = []
        for region in regions:
            # Cắt và xử lý từng vùng
            cropped_image = self.preprocess_image(region)
            
            # CNN nhận dạng
            cnn_pred = self.math_cnn.predict(np.expand_dims(cropped_image, 0))
            
            # OCR trích xuất
            ocr_text = self.ocr_extraction(region)
            
            # Chuyển LaTeX
            latex_result = self.convert_to_latex(ocr_text)
            
            results.append({
                'region': region,
                'cnn_prediction': cnn_pred,
                'ocr_text': ocr_text,
                'latex': latex_result
            })
        
        return results
    
    def render_streamlit_ui(self):
        """Giao diện người dùng"""
        import streamlit as st
        
        st.title("🧮 Hệ Thống OCR Toán Học Thông Minh")
        
        uploaded_file = st.file_uploader("Tải ảnh toán học", 
                                         type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Lưu ảnh
            with open("temp_math_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Phân tích
            results = self.ensemble_prediction("temp_math_image.png")
            
            # Hiển thị kết quả
            for idx, result in enumerate(results, 1):
                st.subheader(f"Vùng Toán Học {idx}")
                st.write("Văn Bản OCR:", result['ocr_text'])
                st.write("LaTeX:", result['latex'])
                st.write("Độ Chính Xác CNN:", result['cnn_prediction'])

# Huấn luyện mô hình
def train_models():
    # Logic huấn luyện các mô hình
    pass

if __name__ == "__main__":
    math_ocr = AdvancedMathOCR()
    math_ocr.render_streamlit_ui()
