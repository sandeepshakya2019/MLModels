# Fake News Prediction - Jupyter Notebook

This project demonstrates a workflow for detecting fake news using machine learning techniques in Python. The main steps include data preprocessing, feature extraction, model training, and evaluation.

---

## Workflow Overview

![Workflow Diagram](Notebook/images/workflow.png)

1. **Data Loading**  
   The dataset (`WELFake_Dataset.csv`) is loaded and basic exploration is performed to understand its structure and contents.

   ![Data Loading](Notebook/images/data_loading.png)

2. **Data Preprocessing**  
   - Handling missing values by replacing them with empty strings.
   - Text cleaning: removing special characters, converting to lowercase.
   - Stopword removal and stemming using NLTK to reduce words to their root form.

   ![Preprocessing](Notebook/images/preprocessing.png)

3. **Feature Extraction**  
   - Textual data from the `title` and `text` columns is converted to numerical features using `TfidfVectorizer`.

   ![Feature Extraction](Notebook/images/feature_extraction.png)

4. **Model Training**  
   - A logistic regression model is trained to classify news as real or fake.

   ![Model Training](Notebook/images/model_training.png)

5. **Evaluation**  
   - Model performance is evaluated using accuracy and other metrics.

   ![Evaluation](Notebook/images/evaluation.png)

---

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- seaborn
- matplotlib
- nltk
- scikit-learn

---

## Usage

1. Clone the repository and navigate to the project directory.
2. Ensure the dataset (`WELFake_Dataset.csv`) is available in the `Datasets` folder.
3. Install the required libraries:
    ```bash
    pip install pandas numpy seaborn matplotlib nltk scikit-learn
    ```
4. Open `index.ipynb` in Jupyter Notebook and run the cells sequentially.

---

## Project Structure

```
FakeNewsPrediction/
│
├── Datasets/
│   └── WELFake_Dataset.csv
├── Notebook/
│   ├── index.ipynb
│   └── images/
│       ├── workflow.png
│       ├── data_loading.png
│       ├── preprocessing.png
│       ├── feature_extraction.png
│       ├── model_training.png
│       └── evaluation.png
└── README.md
```

---

## Notes

- The notebook includes code for downloading NLTK stopwords and applying stemming.
- Both the `title` and `text` columns are preprocessed and used for feature extraction.
- The model can be further improved by experimenting with different algorithms and feature engineering techniques.
- **Images**: Add your own screenshots or diagrams in the `Notebook/images/` folder for better visualization.

---

## Example Results

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.95   |
| ...      | ...    |

---

## License

This project is for educational purposes.