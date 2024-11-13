import streamlit as st
import easyocr
import pytesseract
from PIL import Image
import cv2
import numpy as np

class AdvancedMathOCR:
    def __init__(self):
        self.setup_app()
    
    def setup_app(self):
        st.set_page_config(
            page_title="🧮 MathSolver OCR",
            page_icon="🧮",
            layout="wide"
        )
    
    def preprocess_image(self, image):
        """Tiền xử lý ảnh để tăng chất lượng OCR"""
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Làm sáng ảnh
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Nhị phân hóa
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(binary)
    
    def extract_text(self, image):
        """Trích xuất văn bản từ ảnh"""
        # EasyOCR
        reader = easyocr.Reader(['en'])
        easyocr_results = reader.readtext(np.array(image))
        
        # Tesseract OCR
        tesseract_text = pytesseract.image_to_string(image)
        
        return {
            'easyocr_raw': [text for _, text, _ in easyocr_results],
            'tesseract_text': tesseract_text
        }
    
    def render_sidebar(self):
        """Thanh điều hướng"""
        st.sidebar.title("🧮 MathSolver")
        st.sidebar.markdown("### Công Cụ Trích Xuất Toán Học")
        
        st.sidebar.header("Hướng Dẫn Sử Dụng")
        st.sidebar.info("""
        1. Tải ảnh chứa công thức toán học
        2. Hệ thống sẽ tự động xử lý
        3. Kết quả hiển thị bên dưới
        """)
    
    def main_interface(self):
        """Giao diện chính"""
        st.title("🧮 Trích Xuất Công Thức Toán Học")
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Chọn hình ảnh", 
            type=['png', 'jpg', 'jpeg'],
            help="Tải ảnh chứa công thức toán học"
        )
        
        if uploaded_file is not None:
            # Đọc ảnh
            original_image = Image.open(uploaded_file)
            
            # Hiển thị ảnh gốc
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Ảnh Gốc")
                st.image(original_image, use_column_width=True)
            
            # Tiền xử lý ảnh
            processed_image = self.preprocess_image(original_image)
            
            with col2:
                st.subheader("Ảnh Xử Lý")
                st.image(processed_image, use_column_width=True)
            
            # Trích xuất văn bản
            extraction_results = self.extract_text(processed_image)
            
            # Hiển thị kết quả
            st.header("🔍 Kết Quả Trích Xuất")
            
            # Tab kết quả
            tab1, tab2 = st.tabs(["EasyOCR", "Tesseract"])
            
            with tab1:
                st.subheader("Kết Quả EasyOCR")
                for i, text in enumerate(extraction_results['easyocr_raw'], 1):
                    st.text(f"{i}. {text}")
            
            with tab2:
                st.subheader("Kết Quả Tesseract")
                st.code(extraction_results['tesseract_text'])
            
            # Nút tải xuống
            st.download_button(
                label="Tải Kết Quả",
                data="\n".join(extraction_results['easyocr_raw'] + 
                               [extraction_results['tesseract_text']]),
                file_name="math_ocr_results.txt",
                mime="text/plain"
            )
    
    def run(self):
        """Chạy ứng dụng"""
        self.render_sidebar()
        self.main_interface()

# Khởi chạy ứng dụng
def main():
    app = AdvancedMathOCR()
    app.run()

if __name__ == "__main__":
    main()
