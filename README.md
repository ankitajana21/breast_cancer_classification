# Breast Cancer Classification using Machine Learning

This project demonstrates how to build a **Breast Cancer Classification System** using **Python** and **Machine Learning**.

The objective is to classify tumors into two categories:

- **Malignant** — Cancerous tumor
- **Benign** — Non-cancerous tumor

The classification is performed based on various measurements of cell nuclei obtained from the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

---

## Project Overview

Breast cancer diagnosis involves analyzing characteristics of cell nuclei obtained from breast tissue samples. Machine learning can be used to identify patterns in these measurements and predict whether a tumor is malignant or benign.

In this project, a **Logistic Regression** model is trained to perform binary classification.

### Objective

Build a machine learning model that can:

1. Load and analyze breast cancer data.
2. Preprocess the dataset.
3. Split the data into training and testing sets.
4. Train a Logistic Regression classifier.
5. Evaluate the model's performance.
6. Predict whether a new tumor is malignant or benign.

---

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** available through `scikit-learn`.

The dataset contains:

- **569 samples**
- **30 numerical features**
- **2 target classes**

Some of the features include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

### Target Labels

| Label | Diagnosis |
|------:|-----------|
| `0` | Malignant |
| `1` | Benign |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Google Colab

---

## Data Processing

The dataset is loaded using `scikit-learn` and converted into a **Pandas DataFrame** for easier analysis.

The target labels are added to the DataFrame and analyzed to understand the distribution of malignant and benign cases.

The features are separated from the target variable before training the machine learning model.

---

## Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

This is performed using the `train_test_split()` function from `scikit-learn`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=2
)
```

---

## Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** for binary classification.

The model learns the relationship between the 30 cell measurements and the tumor diagnosis.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
```

After training, the model can be used to predict the diagnosis of previously unseen samples.

---

## Model Evaluation

The trained model is evaluated on both the training and testing datasets.

```python
from sklearn.metrics import accuracy_score

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    Y_train,
    X_train_prediction
)

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    Y_test,
    X_test_prediction
)
```

The model achieves high accuracy on both training and testing data, demonstrating that Logistic Regression can effectively perform this binary classification task on the selected dataset.

> Note: Exact accuracy may vary depending on preprocessing, random state, model parameters, and train-test split.

---

## Predictive System

A final predictive system allows the user to provide the measurements of a new tumor.

The trained model processes these measurements and predicts whether the tumor is malignant or benign.

Example:

```python
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,
0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,
99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The Breast Cancer is Malignant")
else:
    print("The Breast Cancer is Benign")
```

---

## Project Structure

```text
Breast-Cancer-Classification/
|
├── Breast_Cancer_Classification_using_Machine_Learning(Logistic_Regression).ipynb
├── requirements.txt
└── README.md
```

---

## Running the Project

This project was developed using **Google Colab**.

### Using Google Colab

1. Clone or download this repository.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select **File → Upload Notebook**.
4. Upload `Breast_Cancer_Classification_using_Machine_Learning(Logistic_Regression).ipynb`.
5. Run the notebook cells sequentially.

### Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Breast_Cancer_Classification_using_Machine_Learning(Logistic_Regression).ipynb
```

---

## Skills used for this project

This project demonstrates the following skills:

* Machine Learning
* Binary Classification
* Logistic Regression
* Dataset preprocessing
* Pandas DataFrames
* NumPy arrays
* Feature and target separation
* Training and testing datasets
* Model evaluation
* Accuracy measurement
* Making predictions with trained models

---

## Disclaimer

This project is intended **for educational and demonstration purposes only**.

The predictions generated by this machine learning model should **not be used as a substitute for professional medical diagnosis, clinical testing, or advice from a qualified healthcare professional**.

---


