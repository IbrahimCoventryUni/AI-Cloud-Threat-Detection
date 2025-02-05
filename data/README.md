# Data Folder  

This folder contains security log files and datasets used for training and testing the AI-driven threat detection system.  

## Folder Structure  
- `raw/` → Stores unprocessed security logs.  
- `processed/` → Contains cleaned and structured data after preprocessing.  
- `test/` → Sample test logs for model evaluation.  

## Adding Datasets  
1️ Place raw log files inside the `raw/` folder.  
2️ Run preprocessing scripts to clean the data.  
3️ The cleaned data is saved in the `processed/` folder.  

## Notes  
- **Format**: Datasets should ideally be in `.csv` or `.json` format.  
- **Source**: Use open-source datasets like [CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html).  
- **Size**: Avoid uploading large files (>100MB) directly to GitHub.  

---
