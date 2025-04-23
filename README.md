# Multimodal Prediction
Follow the steps below to set up the environment, process the data, and train the models.

## 🐍 Setting Up a Python Virtual Environment

It's recommended to use a virtual environment to manage dependencies and avoid conflicts across projects.

### ✅ Steps to Create and Activate a Virtual Environment

1. **Create the virtual environment**  
   Replace `venv` with your preferred environment name:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**

   - **Linux/macOS**
     ```bash
     source venv/bin/activate
     ```

   - **Windows**
     ```bash
     .\venv\Scripts\activate
     ```

3. **Install project dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the virtual environment when you're done**
   ```bash
   deactivate
   ```

> 📌 Make sure you’re using Python 3.7 or higher. You can check your version with:
```bash
python3 --version
```

## 📥 Download MARBLE Dataset

Download the MARBLE dataset from the official website:

👉 [MARBLE Dataset - EveryWareLab](https://everywarelab.di.unimi.it/index.php/research/datasets/220-marble-dataset-multi-inhabitant-activities-of-daily-living-combining-wearable-and-environmental-sensors-data)

---

## 📁 Directory Setup

Once downloaded, create a folder named `dataset` in the root directory of the project, and place the extracted MARBLE dataset inside. In addition, create a folder named `all_data` which will be used for data loading. Your folder structure should look like this:

```
Multimodal_Prediction/
├── dataset/
│   └── MARBLE/
│       └── dataset/
│           ├── A1a/
│           ├── A1e/
│           └── ...
├── all_data/
...
```

> 📝 **Note:** Make sure the dataset structure matches this layout exactly to avoid path errors during preprocessing.

---

## ⚙️ Data Preprocessing

Run the preprocessing script from the root directory to process the raw MARBLE data:

```
python data_preprocess.py --marble_dataset_path="./dataset/MARBLE/dataset"
```

This will generate synchronized sensor data and natural language summaries used for model training and evaluation.

## 📊 Data Processing

Run the processing script to process the synced MARBLE data:

```
python data_process.py --synced_marble_data_path="./all_data/synced_marble_data.csv" --save_path="./all_data"
```

This will generate the .joblib and .npy files necessary for data loading.

## Model Training [WIP]

Run the main script for model training and evaluation.

```
python main.py --imu_joblib_file="./all_data/MARBLE_IMU.joblib" --embeddings_dir="./all_data/" --sentence_encoder="all-MiniLM-L12-v2" --batch_size=250
```

This will load the data and train the models.