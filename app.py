import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Option 1: Use the default download directory for NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Option 2: Use a custom directory (uncomment to use)
# nltk_data_path = './nltk_data'
# if not os.path.exists(nltk_data_path):
#     os.makedirs(nltk_data_path)
# os.environ['NLTK_DATA'] = nltk_data_path
# nltk.data.path.append(nltk_data_path)
# nltk.download('punkt', download_dir=nltk_data_path)
# nltk.download('stopwords', download_dir=nltk_data_path)

st.set_page_config(page_title="SMS Spam Classifier by Shubham",
                   page_icon="📩")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("📩 Email/SMS :red[Spam] Classifier ")
input_sms = st.text_area("Enter Your Message")
if st.button('Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.error(" SPAM", icon="❌")
    else:
        st.success(" NOT SPAM", icon="✅")

st.subheader("Look at :grey[Model] building approach", divider="gray")
if st.button(':grey[Show Documentation]'):
    st.markdown(
        """
This project uses machine learning to detect spam SMS. The system classifies incoming SMS messages into two categories—spam and ham (non-spam) :grey[ An effective spam detection system is crucial to safeguard users against unwanted and potentially harmful messages.]

- **Dataset:** [kaggle training data](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Data Cleaning**
    - Dropping unwanted columns
    - Renaming the columns
    - Converting "Target" column ["ham": 0, "spam": 1]
    - Removing null values
    - Removing duplicate values
- **EDA**
    - Data is imbalanced (more ham messages than spam)
    - HAM messages are made with fewer characters
    - SPAM messages have more characters
- **Data Preprocessing**
    - Lower case
    - Tokenization
    - Removing special characters
    - Removing stop words and punctuation
    - Stemming
- **Model Building**
    - Convert SMS to vectors (tfidf)
    - :red[MultinomialNB] used for model building
- Extracting model with pickle
- Streamlit for website and deployment
        """
    )
