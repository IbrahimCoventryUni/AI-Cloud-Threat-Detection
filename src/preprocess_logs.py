
import pandas as pd  

def load_data(file_path):  
    """Load cloud security log dataset."""  
    try:  
        data = pd.read_csv(file_path)  
        print("Dataset Loaded Successfully!")  
        return data  
    except Exception as e:  
        print(f"Error loading dataset: {e}")  
        return None  

def preprocess_data(data):  
    """Preprocess log dataset by handling missing values and converting timestamps."""  
    data = data.dropna()  
    if 'timestamp' in data.columns:  
        data['timestamp'] = pd.to_datetime(data['timestamp'])  
    print("Preprocessing Complete!")  
    return data  

if __name__ == "__main__":  
    file_path = "C:/Users/ibrah/Desktop/GithubDissertation/AI-Cloud-Threat-Detection/data/log3.csv"  
    df = load_data(file_path)  
    if df is not None:  
        df = preprocess_data(df)  





