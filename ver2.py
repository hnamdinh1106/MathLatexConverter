import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

class MathImageSolver:
    def __init__(self):
        # Khởi tạo các processor
        self.ocr_processor = self._setup_ocr()
        self.image_processor = ImageProcessor()
        
    def _setup_ocr(self):
        # Cấu hình Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        return pytesseract

    def process_math_image(self, uploaded_file):
        # Đọc ảnh từ file upload
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Bước 1: Trích xuất văn bản
        text = self._extract_text(image_np)
        
        # Bước 2: Phân tích loại bài toán
        problem_type = self._classify_problem_type(text)
        
        # Bước 3: Xử lý giải bài toán
        solution = self._solve_problem(text, problem_type)
        
        return {
            'original_text': text,
            'problem_type': problem_type,
            'solution': solution
        }

    def _extract_text(self, image_np):
        # OCR để trích xuất văn bản
        text = pytesseract.image_to_string(image_np, lang='vie')
        return text

    def _classify_problem_type(self, text):
        # Logic phân loại bài toán
        keywords = {
            'geometry': ['tam giác', 'hình', 'diện tích'],
            'word_problem': ['tìm', 'bài toán', 'xác định']
        }
        
        for problem_type, type_keywords in keywords.items():
            if any(keyword in text.lower() for keyword in type_keywords):
                return problem_type
        
        return 'unknown'

    def _solve_problem(self, text, problem_type):
        # Logic giải bài toán theo từng loại
        if problem_type == 'geometry':
            return self._solve_geometry_problem(text)
        elif problem_type == 'word_problem':
            return self._solve_word_problem(text)
        
        return "Không thể giải bài toán"

    def _solve_geometry_problem(self, text):
        # Xử lý bài toán hình học
        return {
            'method': 'Giải bài toán hình học',
            'steps': ['Bước 1', 'Bước 2'],
            'result': 'Kết quả bài toán'
        }

    def _solve_word_problem(self, text):
        # Xử lý bài toán lời văn
        return {
            'method': 'Giải bài toán lời văn',
            'steps': ['Bước 1', 'Bước 2'],
            'result': 'Kết quả bài toán'
        }

def main():
    st.title("🧮 Giải Bài Toán Từ Ảnh")
    
    # Sidebar cho tùy chọn
    st.sidebar.header("Cài Đặt")
    
    # Upload ảnh 
    uploaded_file = st.file_uploader(
        "Tải Ảnh Bài Toán", 
        type=['png', 'jpg', 'jpeg']
    )
    
    # Xử lý ảnh nếu được tải
    if uploaded_file is not None:
        # Hiển thị ảnh
        st.image(uploaded_file, caption="Ảnh Bài Toán")
        
        # Xử lý toán học
        solver = MathImageSolver()
        result = solver.process_math_image(uploaded_file)
        
        # Hiển thị kết quả
        st.subheader("📄 Chi Tiết Bài Toán")
        st.json(result)
        
        # Các tab kết quả
        tab1, tab2, tab3 = st.tabs([
            "Văn Bản Gốc", 
            "Phân Loại", 
            "Giải Bài Toán"
        ])
        
        with tab1:
            st.text(result['original_text'])
        
        with tab2:
            st.write(f"Loại Bài Toán: {result['problem_type']}")
        
        with tab3:
            st.json(result['solution'])

if __name__ == "__main__":
    main()
