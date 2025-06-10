import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from django.conf import settings
import json
import random

# Hàm tiền xử lý (preprocess) giống như trong NaiveBayesClassifier
# Cần phải có hàm này để tiền xử lý input trước khi dự đoán

NB_MODEL_DIR=os.path.join(settings.BASE_DIR, 'app', 'ai', 'naive_bayes_model')

class ResponseSelector:
    def __init__(self, responses_path):
        with open(responses_path, 'r', encoding='utf-8') as f:
            self.intent_responses = json.load(f)

    def get_response(self, intent):
        responses = self.intent_responses.get(intent, [])
        if responses:
            return random.choice(responses)
        else:
            return "Xin lỗi, tôi không hiểu câu hỏi của bạn."

def preprocess_text(text, stop_words):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

class ChatbotInference:
    def __init__(self, models_dir=NB_MODEL_DIR):
        self.classifier_model = None
        self.vectorizer_model = None
        self.stop_words = None
        self.response_selector = None # Thêm biến để lưu ResponseSelector

        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Stopwords not found, downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

        # Đường dẫn tới các file model và data
        base_dir = '.' # Đảm bảo base_dir là thư mục hiện tại của chatbot
        classifier_filename = os.path.join(base_dir, models_dir, 'naive_bayes_classifier.pkl')
        vectorizer_filename = os.path.join(base_dir, models_dir, 'count_vectorizer.pkl')
        responses_path = os.path.join(base_dir, models_dir, 'intent_responses.json') # Đường dẫn tới file responses

        # Tải classifier (model Naive Bayes)
        with open(classifier_filename, 'rb') as f:
            self.classifier_model = pickle.load(f)
        print(f"Đã tải classifier từ: {classifier_filename}")

        # Tải vectorizer (để chuyển đổi văn bản sang dạng số)
        with open(vectorizer_filename, 'rb') as f:
            self.vectorizer_model = pickle.load(f)
        print(f"Đã tải vectorizer từ: {vectorizer_filename}")

        # Tải bộ chọn phản hồi
        self.response_selector = ResponseSelector(responses_path)
        print(f"Đã tải bộ chọn phản hồi từ: {responses_path}")

    # Phương thức để dự đoán intent (chỉ trả về intent)
    def predict_intent(self, text):
        cleaned_text = preprocess_text(text, self.stop_words)
        X_vec = self.vectorizer_model.transform([cleaned_text])
        intent = self.classifier_model.predict(X_vec)[0]
        return intent

    # Phương thức mới để lấy cả intent và response
    def get_chatbot_response(self, user_input):
        intent = self.predict_intent(user_input) # Dự đoán intent
        response = self.response_selector.get_response(intent) # Lấy phản hồi
        return intent, response # Trả về cả hai

def get_naivebayes_model_response(user_message: str) -> dict:
    nbmodel = ChatbotInference()

    try: 
        predicted_intent, chatbot_response = nbmodel.get_chatbot_response(user_message)

        return {
            "predicted_intent": predicted_intent,
            "chatbot_response": chatbot_response,
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

