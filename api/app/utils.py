import os
import joblib
def get_model():
    model_path = os.environ.get('MODEL_PATH','src/models/best_model_47_5%.pkl')
    if os.path.exists(model_path):
        model_fit = joblib.load(model_path)
        return model_fit
    # else:
    #     return {"error": "Model file not found"}
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")