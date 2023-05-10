import os
import joblib
def get_model():
    model_path = os.environ.get('MODEL_PATH','models/best_model_47_5%.pkl')
    model_fit = joblib.load(model_path)
    return model_fit