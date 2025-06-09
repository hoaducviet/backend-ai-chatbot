import tensorflow as tf
import pandas as pd
import re
import pickle
from transformers import AutoTokenizer # Dùng AutoTokenizer để tải lại tokenizer
import random # Để chọn ngẫu nhiên câu trả lời
import os
from django.conf import settings

# --- 1. Cấu hình chung ---
# Đảm bảo max_seq_length khớp với khi huấn luyện
MAX_SEQ_LENGTH = 128
SAVE_DIRECTORY_BASE = os.path.join(settings.BASE_DIR, 'app', 'ai', 'cnn_model')

try:
    loaded_model = tf.keras.models.load_model(f"{SAVE_DIRECTORY_BASE}/cnn_model.keras")
    print("   Mô hình CNN đã được tải lại.")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}. Vui lòng kiểm tra lại đường dẫn: {SAVE_DIRECTORY_BASE}/cnn_model.keras")
    exit()

try:
    with open(f'{SAVE_DIRECTORY_BASE}/label_encoder.pkl', 'rb') as f:
        loaded_label_encoder = pickle.load(f)
    print("   Label Encoder đã được tải lại.")
except Exception as e:
    print(f"Lỗi tải Label Encoder: {e}. Vui lòng kiểm tra lại đường dẫn: {SAVE_DIRECTORY_BASE}/label_encoder.pkl")
    exit()

try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(f'{SAVE_DIRECTORY_BASE}/tokenizer')
    print("   Tokenizer đã được tải lại.")
except Exception as e:
    print(f"Lỗi tải Tokenizer: {e}. Vui lòng kiểm tra lại đường dẫn: {SAVE_DIRECTORY_BASE}/tokenizer")
    exit()


try:
    with open(f'{SAVE_DIRECTORY_BASE}/response_map.pkl', 'rb') as f:
        loaded_response_map = pickle.load(f)
    print("   Response Map đã được tải lại.")
except Exception as e:
    print(f"Lỗi tải Response Map: {e}. Vui lòng kiểm tra lại đường dẫn.")
    exit()



def preprocess_text_dynamic_placeholders(text):
    
    def replacer(match):
        # Lấy nội dung bên trong {{...}}
        placeholder_content = match.group(1)
        # Thay thế khoảng trắng bằng gạch dưới và chuyển thành chữ in hoa
        processed_content = placeholder_content.replace(' ', '_').upper()
        return f'<{processed_content}>'

    text = re.sub(r'\{\{(.*?)\}\}', replacer, text, flags=re.DOTALL)

    return text.strip() # Loại bỏ khoảng trắng thừa ở đầu/cuối

# --- 3. Định nghĩa các hàm tiền xử lý (PHẢI GIỐNG KHI HUẤN LUYỆN) ---
def extract_dynamic_info(user_query):
    extracted_info = {}
    # Trích xuất số đơn hàng: tìm "order" hoặc "number" theo sau bởi khoảng trắng và một chuỗi ký tự bất kỳ
    order_num_match = re.search(r'(?i)order (number)?\s*(\S+)', user_query)
    if order_num_match:
        extracted_info['<ORDER_NUMBER>'] = order_num_match.group(2)

    return extracted_info



# --- 4. Định nghĩa hàm dự đoán ý định ---
def predict_intent(user_query):
    processed_query = preprocess_text_dynamic_placeholders(user_query)
    encoded_query = loaded_tokenizer(
        processed_query,
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQ_LENGTH,
        return_tensors='tf'
    )
    input_ids_query = encoded_query['input_ids']
    predictions = loaded_model.predict(input_ids_query)
    predicted_label_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_intent = loaded_label_encoder.inverse_transform([predicted_label_index])[0]
    return predicted_intent


def get_cnn_model_response(user_message: str) -> dict:
    """
    Returns:
        dict: Một từ điển chứa:
            - 'predicted_intent' (str): Ý định được dự đoán.
            - 'chatbot_response' (str): Phản hồi của chatbot.
            - 'original_message' (str): Tin nhắn gốc của người dùng.
            - 'status' (str): "success" hoặc "error".
            - 'error_message' (str, tùy chọn): Thông báo lỗi nếu có.
    """
    # Kiểm tra xem tất cả các tài nguyên cần thiết đã được tải chưa
    if not all([loaded_model, loaded_label_encoder, loaded_tokenizer, loaded_response_map]):
        # Ghi log lỗi để dễ debug trong production
        print("ERROR: Not all chatbot resources are loaded. Cannot process request.")
        return {
            "predicted_intent": "unloaded_resources",
            "chatbot_response": "Xin lỗi, hệ thống chatbot đang bảo trì. Vui lòng thử lại sau.",
            "original_message": user_message,
            "status": "error",
            "error_message": "Chatbot resources not fully loaded."
        }

    try:
        # 1. Dự đoán ý định
        predicted_intent_tag = predict_intent(user_message)
        
        # 2. Lấy câu trả lời mẫu từ response_map
        chatbot_response_template = None
        if predicted_intent_tag in loaded_response_map and loaded_response_map[predicted_intent_tag]:
            chatbot_response_template = random.choice(loaded_response_map[predicted_intent_tag])
        
        # 3. Trích xuất thông tin động và thay thế placeholders
        final_response = "Xin lỗi, tôi không hiểu ý bạn hoặc không có câu trả lời mẫu cho ý định này."
        if chatbot_response_template:
            final_response = chatbot_response_template
            dynamic_info = extract_dynamic_info(user_message)
            
            for placeholder_token, value in dynamic_info.items():
                if placeholder_token in final_response:
                    final_response = final_response.replace(placeholder_token, value)
        
        return {
            "predicted_intent": predicted_intent_tag,
            "chatbot_response": final_response,
            "original_message": user_message,
            "status": "success"
        }

    except RuntimeError as e:
        # Lỗi nếu có vấn đề với việc tải model hoặc các thành phần ML
        print(f"ERROR in get_response (RuntimeError): {e}")
        return {
            "predicted_intent": "error",
            "chatbot_response": "Xin lỗi, có lỗi nội bộ xảy ra khi xử lý yêu cầu của bạn.",
            "original_message": user_message,
            "status": "error",
            "error_message": str(e)
        }
    except Exception as e:
        # Bắt các lỗi không mong muốn khác
        print(f"ERROR in get_response (General Exception): {e}")
        return {
            "predicted_intent": "error",
            "chatbot_response": "Xin lỗi, có lỗi không xác định xảy ra. Vui lòng thử lại sau.",
            "original_message": user_message,
            "status": "error",
            "error_message": str(e)
        }