import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import openai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import tempfile

# Thiết lập tiêu đề
st.set_page_config(page_title="Math Solver & LaTeX Converter", layout="wide")

class SimpleTexConverter:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

    def convert_to_latex(self, image_path):
        try:
            self.driver.get('https://simpletex.cn/ai/latex_ocr')
            file_input = self.driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            file_input.send_keys(image_path)
            
            latex_result = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.latex-result'))
            )
            return latex_result.text
        except Exception as e:
            return f"Error: {str(e)}"

    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.quit()

class OCRProcessor:
    @staticmethod
    def preprocess_image(image):
        # Chuyển đổi sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Ngưỡng hóa
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh
    

    @staticmethod
    def extract_text_from_image(image):
        try:
            # Thử Tesseract trước
            preprocessed_image = OCRProcessor.preprocess_image(image)
            text = pytesseract.image_to_string(preprocessed_image, lang='eng')
            
            if not text.strip():
                # Nếu Tesseract không nhận diện được, chuyển sang EasyOCR
                reader = easyocr.Reader(['en'])
                results = reader.readtext(image)
                text = ' '.join([result[1] for result in results])
            
            return text.strip()
        
        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "Không thể trích xuất văn bản từ ảnh"
        
class MathSolverGPT:
    def __init__(self, api_key):
        self.api_key = "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"
        openai.api_key = self.api_key

    def process_request(self, user_input):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a math assistant."},
                    {"role": "user", "content": user_input},
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

# Khởi tạo session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar cho API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        math_solver = MathSolverGPT(api_key)

# Main content
st.title("Math Solver & LaTeX Converter")

# Hai cột cho input và chat history
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    
    # Chọn chế độ
    task_type = st.radio("Choose task", ["Solve Math Problem", "Convert to LaTeX"])
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    # Text input
    text_input = st.text_area("Or type your question")
    
    if st.button("Process"):
        if api_key:
            if uploaded_file:
                # Xử lý ảnh
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    cv2.imwrite(tmp_file.name, image)
                    
                    if task_type == "Convert to LaTeX":
                        simpletex = SimpleTexConverter()
                        result = simpletex.convert_to_latex(tmp_file.name)
                    else:
                        extracted_text = OCRProcessor.extract_text_from_image(image)
                        result = math_solver.process_request(f"Solve this math problem: {extracted_text}")
                    
                    os.unlink(tmp_file.name)
            else:
                if text_input:
                    if task_type == "Convert to LaTeX":
                        result = math_solver.process_request(f"Convert this to LaTeX: {text_input}")
                    else:
                        result = math_solver.process_request(f"Solve this math problem: {text_input}")
                else:
                    result = "Please provide either an image or text input"
                    
            # Add to chat history
            st.session_state.chat_history.append(("User", text_input or "Image uploaded"))
            st.session_state.chat_history.append(("Assistant", result))
        else:
            st.error("Please enter your OpenAI API key in the sidebar")

with col2:
    st.subheader("Chat History")
    for role, message in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"**You**: {message}")
        else:
            st.markdown(f"**Assistant**: {message}")
    
    if st.button("Clear History"):
        st.session_state.chat_history = []
