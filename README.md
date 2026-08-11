# DeepHAR: Heterogeneous Attention-Conditioned Regression Network for Realized Volatility Forecasting

This repository contains the official PyTorch implementation of **DeepHAR**, a novel deep learning framework for explainable financial volatility forecasting. 


---

## 🌟 Key Features
* **Multi-Horizon Architecture**: Specifically designed to capture heterogeneous volatility components (Daily, Weekly, Monthly) based on the HAR framework.
* **Temporal Attention Pooling**: Utilizes an attention mechanism to distill pivotal time points from multi-timescale data, generating representative vectors for each horizon.
* **Heterogeneous Cross-Attention**: Learns structural representations by referencing historical contexts (Key/Value) conditioned on the current market state (Query).
* **Dual-Path Decomposition**: Explicitly models complex market dynamics by disentangling the integrated information into distinct **Trend** and **Shock** paths.
* **Explainability (XAI)**: Provides intrinsic explainability by exposing attention weights through the `forward_attention` method, enabling the analysis of cross-horizon dependencies.
* **Numerical Stability**: Implements Shifted ReLU ($ReLU(x) + \epsilon$) to ensure strictly positive outputs, preventing numerical instability and ensuring the mathematical validity of the **QLIKE loss** computation.

---

## 📂 Project Structure
| File/Folder | Description |
| :--- | :--- |
| `dataset/all/` | Data directory containing `.npy` files |
| `lib/Model.py` | Model architectures (DeepHAR, LSTM, BILSTM, GRU) |
| `lib/modules.py` | Core utilities |
| `lib/datasetLoader.py` | data loading |
| `lib/Metric.py` | Standard volatility metrics (QLIKE, MSE) |
| `train.py` | Training logic and validation loops |
| `run.py` | Main entry point for training the model |
| `test.py` | Script for inference |


---

## ⚙️ Environment Setup
The code is tested with **Python 3.10+** and the specific library versions listed below.

### 1. Requirements
Install the dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```
Main Dependencies:
* numpy==2.1.2
* pandas==2.3.2
* scipy==1.16.1
* scikit-learn==1.7.2
* statsmodels==0.14.6
* tqdm==4.67.1
* matplotlib==3.10.5
* easydict==1.13
* torch==2.7.1+cu128 # PyTorch (GPU/CUDA 12.8 build)



### 2.Execution Pipeline (Training & Evaluation)
#### 1) Data Preparation
Place the dataset files under ./dataset/all/:

**Note on Data Sharing:**
Due to licensing restrictions and data privacy policies, we provide a **sampled subset (100 samples)** of the original dataset in this repository. 
- These samples are provided to verify the **technical functionality** and **reproducibility** of the code pipeline.
- For the full experimental results reported in the paper, the complete proprietary dataset was used.

**Reproducing the full dataset from raw data:**
To reconstruct the complete dataset used in the paper, run the acquisition and preprocessing pipeline with your own Alpha Vantage API key:
```bash
python run_dataset.py --api_key YOUR_API_KEY
```
This crawls the raw intraday data (SPY, DIA, QQQ, 2005-01 to 2025-12), cleans it, builds the daily HAR features, and constructs the train/validation/test splits described in Appendix B.1. If the raw CSVs already exist locally, add `--skip_crawl` to skip re-downloading.


```
dataset/all/
  train_data.npy
  train_label.npy
  validation_data.npy
  validation_label.npy
  test_data.npy
  test_label.npy
```

#### 2) Training & Validation
Run run.py to start training. EarlyStopping monitors validation QLIKE and saves the best checkpoint.

```bash
# Method 1: Using shell script
sh ./scripts.sh

# Method 2: Using python command directly
python run.py \
	--random_seed 2026 \
	--dropout 0.1 \
	--learning_rate 0.0001 \
	--n_heads 8 \
	--lstm_hidden_dim 64 \
	--train_epochs 200 \
	--loss QLIKE \
	--model_type DeepHAR
```

#### 3) Testing & Performance Evaluation
Evaluate the best checkpoint on the test set:

```bash
# Method 1: Using shell script
sh ./scripts_test.sh

# Method 2: Using python command directly
python test.py \
	--random_seed 2026 \
	--dropout 0.1 \
	--learning_rate 0.0001 \
	--n_heads 8 \
	--lstm_hidden_dim 64 \
	--train_epochs 200 \
	--loss QLIKE \
	--model_type DeepHAR

```


#### 4) Expected Output
`test.py` reports the final metrics in JSON format. The results are displayed in the console and automatically saved as a file in the `./result/` directory.

```json
{
  "QLIKE": 0.XXXX,
  "MSE": 0.XXXX
}
```


### License
For review purposes only. Full license information will be provided upon publication.
