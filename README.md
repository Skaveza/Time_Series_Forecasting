#  Air Quality Forecasting with LSTM

This project focuses on forecasting air pollution levels — specifically PM2.5 concentrations — using deep learning models, particularly LSTM (Long Short-Term Memory) networks. The project was developed as part of a Kaggle competition on air quality prediction.

---

##  Model Overview

### Model Type:

* Recurrent Neural Network (RNN) with **LSTM layers**

### Final Architecture:

* **Input Window Size:** 24 time steps
* **Layers:**

  * LSTM(64, return\_sequences=True, activation='tanh')
  * BatchNormalization
  * Dropout(0.3)
  * LSTM(32, activation='tanh')
  * Dropout(0.25)
  * Dense(16, activation='relu', kernel\_regularizer=L2(0.001))
  * Dense(1) – final prediction of PM2.5
* **Optimizer:** Adam (lr=0.0002)
* **Loss:** Mean Squared Error (MSE)
* **Metric:** Root Mean Squared Error (RMSE)

---

## Performance Summary

| Experiment   | Description                                     | RMSE (Kaggle) | Final Train MSE | Epochs |
| ------------ | ----------------------------------------------- | ------------- | --------------- | ------ |
| Current Best | 2 LSTM layers + BatchNorm + Dropout + Dense(16) | **5235.39**   | 3112.78         | 25     |

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Use the training script or notebook:

```python
python train_model.py
```

Or run the Jupyter notebook for training experiments.

### 3. Make Predictions

```python
python predict.py
```

The output submission file will be saved in the `submissions/` folder.

---

## 📈 Evaluation Strategy

* **Validation split:** 20% of training data
* **EarlyStopping:** Monitors `val_loss` with patience=8
* **ReduceLROnPlateau:** Reduces learning rate by factor of 0.5 on plateau

---

## 📦 Requirements

* Python ≥ 3.8
* TensorFlow ≥ 2.9
* pandas, numpy, scikit-learn, matplotlib

---

##  Future Improvements

* Experiment with **GRU** and **Bidirectional LSTM**
* Explore **attention mechanisms** for enhanced temporal sensitivity
* Perform **feature engineering** and **pollutant interaction analysis**
* Try **multivariate forecasting** (include other pollutants as output)

---

##  Kaggle Competition

This project was developed for the **[Air Quality Forecasting Competition](https://www.kaggle.com)**.
Final submission RMSE: **5235.3927**

---

