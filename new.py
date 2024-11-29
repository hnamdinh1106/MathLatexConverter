import os
import base64
import time
import streamlit as st
from PIL import Image
from rapid_latex_ocr import LaTeXOCR
import openai
from dotenv import load_dotenv
from datetime import datetime
import logging

# Tải biến môi trường
load_dotenv()

# Cấu hình OpenAI API
openai.api_key = os.getenv('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')

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app_log.txt'),
        logging.StreamHandler()
    ]
)

# Khởi tạo mô hình OCR
latex_ocr = LaTeXOCR()

# Hàm chuyển đổi ảnh sang mã LaTeX
def convert_image_to_latex(image_data):
    try:
        result, elapsed_time = latex_ocr(image_data)
        return result, elapsed_time
    except Exception as e:
        logging.error(f"Error during LaTeX conversion: {str(e)}")
        return "Error: Unable to process image", None

# Hàm giải toán bằng OpenAI
def solve_math_problem(problem):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Solve the following math problem step by step:\n\n{problem}",
            max_tokens=300,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error during math problem solving: {str(e)}")
        return "Error: Unable to solve the problem"

# Hàm lưu lịch sử
def save_history(entry):
    with open("history.txt", "a") as file:
        file.write(f"{datetime.now()} - {entry}\n")

# Giao diện Streamlit
def main():
    # Tùy chỉnh giao diện
    st.set_page_config(
        page_title="Advanced Math OCR & Solver",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Đặt CSS tùy chỉnh theo phong cách giống Facebook
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f7f9fc;
            font-family: Arial, sans-serif;
        }
        .css-1v3fvcr {
            background-color: #4267B2;
            color: white;
        }
        .stButton button {
            background-color: #4267B2;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Tiêu đề ứng dụng
    st.title("📘 Advanced Math OCR & Solver")
    st.markdown("Welcome to the **Advanced Math OCR & Solver** app. Choose a feature below.")

    # Sidebar
    with st.sidebar:
        st.header("🔍 Features")
        options = ["LaTeX Convert", "History", "Solve Math"]
        choice = st.radio("Select a feature", options)

    # LaTeX Convert
    if choice == "LaTeX Convert":
        st.header("📄 Convert Image to LaTeX")
        uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            image_data = uploaded_file.read()
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
            st.info("Processing the image...")

            latex_result, elapsed_time = convert_image_to_latex(image_data)
            if latex_result != "Error: Unable to process image":
                st.success(f"Conversion completed in {elapsed_time:.2f} seconds.")
                st.code(latex_result, language="latex")
                save_history(f"LaTeX Conversion: {latex_result}")
            else:
                st.error("Failed to process the image. Please try again.")

    # History
    elif choice == "History":
        st.header("📜 History")
        if os.path.exists("history.txt"):
            with open("history.txt", "r") as file:
                history_content = file.readlines()
                if history_content:
                    st.write("Recent activity:")
                    for entry in reversed(history_content[-10:]):
                        st.markdown(f"- {entry.strip()}")
                else:
                    st.info("No history available.")
        else:
            st.info("No history available.")

    # Solve Math
    elif choice == "Solve Math":
        st.header("📊 Solve Math Problem")
        math_problem = st.text_area("Enter your math problem below:")

        if st.button("Solve"):
            if math_problem.strip():
                st.info("Solving the problem...")
                solution = solve_math_problem(math_problem)
                if solution != "Error: Unable to solve the problem":
                    st.success("Solution:")
                    st.text_area("Result", solution, height=200)
                    save_history(f"Solved Math Problem: {math_problem}")
                else:
                    st.error("Failed to solve the problem. Please try again.")
            else:
                st.warning("Please enter a math problem to solve.")

if __name__ == "__main__":
    main()
