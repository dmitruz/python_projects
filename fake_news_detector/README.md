The Fake News & Misinformation Detector is a complete end-to-end Natural Language Processing (NLP) project that classifies news headlines and articles as REAL or FAKE.
It combines TF-IDF feature extraction with a Logistic Regression classifier, achieving perfect accuracy on the cleaned dataset.


<img width="831" height="527" alt="fakeNewsIMG" src="https://github.com/user-attachments/assets/bdccf0b7-41cf-44ac-b053-9780e54b3d5a" />

The project also includes:

Model evaluation with visual charts
Interactive Streamlit web app
Reusable and modular code structure
Demo
Streamlit Web App
When launched, the app allows you to paste or type any news headline or paragraph and analyze its credibility in real time.

Screenshot 2025-10-25 at 17-39-13 Fake News Detector
Prediction: REAL or FAKE
Probability bar visualization
Adjustable fake-detection threshold
Project Structure
fake-news-detector/
│
├── data/
│   ├── True.csv                 # Real news (999 rows)
│   ├── Fake.csv                 # Fake news (999 rows)
│
├── outputs/
│   ├── model.joblib             # Trained Logistic Regression model
│   ├── vectorizer.joblib        # TF-IDF vectorizer
│   ├── pipeline.joblib          # Combined pipeline (optional)
│   ├── metrics.json             # Model performance report
│   ├── confusion_matrix.png     # Confusion Matrix plot
│   ├── roc_curve.png            # ROC curve plot
│   └── pr_curve.png             # Precision-Recall curve plot
│
├── src/
│   ├── text_clean.py            # Text preprocessing utilities
│   ├── utils.py                 # I/O helpers
│   ├── train_model.py           # Training and evaluation script
│   ├── detect_fake_news.py      # CLI prediction script
│   └── streamlit_app.py         # Streamlit web application
│
└── README.md
Installation

Install Dependencies
pip install -r requirements.txt
Or install manually:

pip install pandas numpy scikit-learn matplotlib streamlit joblib
Dataset
File	Type	Rows	Columns
True.csv	Real news	999	title, text, subject, date
Fake.csv	Fake news	999	title, text, subject, date
Dataset Source:
This project uses and modifies the Fake and Real News Dataset by Clément Bisaillon (Kaggle).
Data was cleaned, header-fixed, and downsampled to 999 REAL and 999 FAKE news articles for balanced training and clear visualization.
Used purely for educational and research purposes.

Training the Model
Run the following command from the project root:

python src/train_model.py --real data/True.csv --fake data/Fake.csv --text-col text --outdir outputs

<img width="1050" height="900" alt="confusion_matrix" src="https://github.com/user-attachments/assets/4ddb328d-c213-4a0e-ad2c-9970a7a95772" />

<img width="1050" height="900" alt="pr_curve" src="https://github.com/user-attachments/assets/5d1e78fa-c7a8-4a13-9c6a-35a79a78f049" />


<img width="1050" height="900" alt="roc_curve" src="https://github.com/user-attachments/assets/c7ed481a-d428-4982-9752-5f6c08a70c70" />


This script will:

Load both datasets (real and fake).
Clean and merge them using text_clean.py.
Extract TF-IDF features.
Train a Logistic Regression classifier.
Save outputs:
outputs/model.joblib
outputs/vectorizer.joblib
outputs/metrics.json
Performance charts (confusion_matrix.png, roc_curve.png, pr_curve.png)
Evaluation & Charts
After training, the model achieves perfect classification accuracy on this dataset.

Confusion Matrix
confusion_matrix
True Label	Predicted REAL	Predicted FAKE
REAL	999 ✅	0 ❌
FAKE	0 ❌	999 ✅
The model correctly classified all 1,998 samples.

ROC Curve
roc_curve
The ROC curve touches the top-left corner AUC = 1.00
Perfect separability between classes.

Precision–Recall Curve
pr_curve
Both precision and recall reach 1.00, meaning zero false predictions.

Key Metrics
Metric	Value
Accuracy	100 %
Precision (FAKE)	1.00
Recall (FAKE)	1.00
F1-Score	1.00
ROC-AUC	1.00
Although perfect accuracy is achieved on this dataset, it’s a controlled sample. Real-world news data will naturally introduce noise and uncertainty.

How It Works
Pipeline Overview
Text Cleaning → Remove punctuation, URLs, emails, non-ASCII chars.
TF-IDF Vectorization → Convert words into weighted numerical features.
Logistic Regression → Predict probability of “FAKE” label.
Thresholding → If p(fake) ≥ 0.5 → FAKE, else REAL.
Example: Command-Line Prediction
python src/detect_fake_news.py --model outputs/model.joblib   --vectorizer outputs/vectorizer.joblib   --text "It s tough sometimes to imagine that Donald Trump has five children since it s clear from Monday s speech in front of 40,000 Boy Scouts and other attendees at the Boy Scouts Jamboree in West Virginia that he has absolutely no idea what kind of talk is appropriate for children.While most adults would take this opportunity to offer some pearls of adult wisdom or cheerlead the Boy Scouts toward their futures, Trump chose to deliver a tirade of Trumpisms.Like almost any time Trump has tried to string together more than a couple of words at a time, most of his speech was an inarticulate mess which consisted of his trademark whining, a wee bit of swearing and a pointless anecdote about a burned out rich guy at a cocktail party."
Output:

Label: FAKE | Fake probability: 0.560 | Threshold: 0.40
Running the Streamlit App
Launch the App
streamlit run src/streamlit_app.py
Then open the local web interface:

http://localhost:8501
App Features
Paste any headline or paragraph
Analyze with one click
Adjust FAKE probability threshold
See model file locations and loaded status in sidebar
Code Modules
Module	Purpose
text_clean.py	Handles text normalization (lowercasing, regex-based cleaning)
utils.py	Ensures output directories exist and handles JSON I/O
train_model.py	Loads data, trains the model, and generates metrics and plots
detect_fake_news.py	CLI script for predicting individual samples
streamlit_app.py	Streamlit web app for interactive user testing
Technologies Used
Python 3.10+
scikit-learn → TF-IDF Vectorizer, Logistic Regression
pandas / numpy → Data manipulation
matplotlib → Model visualization
joblib → Model persistence
Streamlit → Web interface
