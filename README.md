# 📧 Spam Mail Classification using Machine Learning  

This project is a **Machine Learning-based Spam Mail Classifier** that predicts whether an email/message is **Spam 🚫** or **Ham ✅** using **Logistic Regression** and **TF-IDF feature extraction**.  

---

## 📂 Dataset  
- The dataset contains **5,572 messages** labeled as either:  
  - `spam` → unwanted promotional/advertisement messages  
  - `ham` → genuine messages  

| Category | Message Example |
|----------|-----------------|
| spam 🚫  | "Congratulations! You have won a free ticket. Call now!" |
| ham ✅   | "Hey, are we still meeting for lunch today?" |

---

## 🛠 Project Workflow  

### 1️⃣ Data Preprocessing  
- Converted `spam` → `0`, `ham` → `1`  
- Cleaned the text (lowercasing, stopword removal)  
- Converted text into **numerical features** using **TF-IDF Vectorizer**  

### 2️⃣ Train-Test Split  
- Dataset split into **80% training** and **20% testing**  

### 3️⃣ Model Training  
- Algorithm used: **Logistic Regression**  
- Trained on **TF-IDF features**  

### 4️⃣ Model Evaluation  
- Evaluated using **Accuracy, Confusion Matrix**  

---

## ⚡ Results  

✅ **Training Accuracy**: `96.76%`  
✅ **Test Accuracy**: `96.68%`  

---

### 📊 Confusion Matrix (Test Data)

|                 | Predicted Spam 🚫 (0) | Predicted Ham ✅ (1) |
|-----------------|------------------------|----------------------|
| **Actual Spam 🚫 (0)** | 118 (True Spam)        | 37 (Missed Spam ❌) |
| **Actual Ham ✅ (1)**  | 0 (Ham misclassified ❌) | 960 (True Ham) |

---
## 🤖 Predictive System  

You can input your own email/message and the system will classify it as **Spam 🚫** or **Ham ✅**.  

```python
def predict_mail(mail):
    # Transform the email using the same TF-IDF vectorizer
    mail_features = feature_extraction.transform([mail])
    
    # Predict using trained model
    prediction = model.predict(mail_features)
    
    if prediction[0] == 0:
        return "Spam 🚫"
    else:
        return "Ham ✅"

# 🔍 Example Tests
print(predict_mail("Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!"))
# Output: Spam 🚫

print(predict_mail("Hey, are we still meeting for lunch tomorrow?"))
# Output: Ham ✅
```
## ⚡Result on Data Set
<img width="1867" height="248" alt="image" src="https://github.com/user-attachments/assets/0f8dedc7-ce90-4e16-8d47-147b9c68e854" />






