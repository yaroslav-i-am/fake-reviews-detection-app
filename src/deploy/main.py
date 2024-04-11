import streamlit as st
from PIL import Image
import dill
import pandas as pd
from annotated_text import annotated_text

import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import boto3


@st.cache_resource(ttl=24 * 60 * 60)
def set_tokenizers():
    nltk.download('punkt')
    return RegexpTokenizer(r'[\w\-]+'), WhitespaceTokenizer()


def process_text(text):
    processed_text = []
    words = reg_tok.tokenize(text.lower())
    words_set = set(words)
    found_words = set()
    for i in range(len(words) - 2):
        if ' '.join(words[i: i + 3]) in gpt_words:
            found_words.update([i, i + 1, i + 2])
        elif ' '.join(words[i: i + 2]) in gpt_words:
            found_words.update([i, i + 1])
    i = 0
    for tok in word_tokenize(text):
        if tok.lower() in words_set and i != 0:
            processed_text.append(' ')
        if tok.lower() in words_set and i in found_words:
            processed_text.append((tok, "гпт", '#Ff7e81'))
        else:
            processed_text.append(tok)

        if tok.lower() in words_set:
            i += 1
    return processed_text


def connect_strings(processed_text):
    strings = []
    prev_tup = True
    new_processed_text = []
    for block in processed_text:
        if type(block) == str and prev_tup:
            strings = [block]
            prev_tup = False
        elif type(block) == str:
            strings.append(block)
        else:
            if not prev_tup:
                new_processed_text.append(''.join(strings))
            new_processed_text.append(block)
            prev_tup = True
    new_processed_text.append(''.join(strings))
    return new_processed_text


def punkt_between(i):
    return type(processed_text[i]) == tuple and type(processed_text[i + 1]) \
           == str and type(processed_text[i + 2]) == tuple \
           and len(reg_tok.tokenize(processed_text[i + 1])) == 0


def connect_highlights(processed_text):
    for i in range(len(processed_text) - 2):
        if punkt_between(i):
            processed_text[i + 2] = (processed_text[i][0] + processed_text[i + 1]
                                     + processed_text[i + 2][0], "гпт", '#Ff7e81')
            processed_text[i] = ''
            processed_text[i + 1] = ''


def has_highlights(processed_text):
    has_tuple = False
    for block in processed_text:
        if type(block) is tuple:
            has_tuple = True
    return has_tuple


@st.cache_resource(ttl=60 * 60)
def download_classifiers():
    with open('./src/deploy/config.pkl', 'rb') as f:
        conf_a, conf_b = dill.load(f)
    bucket_name = 'fake-reviews'
    session = boto3.session.Session()
    ENDPOINT = "https://storage.yandexcloud.net"
    session = boto3.Session(
        aws_access_key_id=(conf_a),
        aws_secret_access_key=(conf_b),
        region_name="ru-central1",
    )
    s3 = session.client(
        "s3", endpoint_url=ENDPOINT)
    for key in s3.list_objects(Bucket=bucket_name)['Contents']:
        if key['Key'] == 'model.pkl':
            data = s3.get_object(Bucket=bucket_name, Key=key.get('Key'))
            return dill.loads(data['Body'].read())


@st.cache_resource(ttl=3 * 60 * 60)
def find_gpt_words():
    with open('./src/deploy/n_gram_model.pkl', 'rb') as f:
        count_vec_ngram, clf_ngram = dill.load(f)
    coefs = pd.DataFrame(data={
        'features': count_vec_ngram.get_feature_names_out(),
        'coef': clf_ngram.coef_.flatten()
    })
    coefs = coefs.sort_values('coef')
    return set(coefs.tail(int(len(coefs) / 4)).values[:, 0])


reg_tok, white_tok = set_tokenizers()
count_vec, clf = download_classifiers()

# with open('./src/deploy/finalized_model2.pkl', 'rb') as f:
#     count_vec, clf = dill.load(f)


gpt_words = find_gpt_words()


@st.cache_resource(ttl=3 * 60 * 60)
def load_reviews():
    with open('./src/deploy/reviews.pkl', 'rb') as f:
        return dill.load(f)


@st.cache_resource(ttl=3 * 60 * 60)
def load_intro():
    st.header("Определение сгенерированных отзывов")
    image = Image.open('./src/deploy/wordcloud4.png')
    st.image(image)


options = load_reviews()
with st.sidebar:
    option = st.selectbox(
        'Выбрать пример',
        list(options.keys()))
load_intro()

text = st.text_area(label="Введите отзыв", value=options[option], height=200)

data = clf.predict_proba(count_vec.transform([text]))
prob = data[0][1]
if st.button('Проанализировать') and len(text.strip()) > 0:
    st.write(f"Отзыв сгенерирован с вероятностью {round(prob * 100, 2)}%")

    st.subheader("Разбор отзыва")
    st.write("Некоторые словосочетания ГПТ употребляет гораздо чаще, чем человек")

    processed_text = process_text(text)
    processed_text = connect_strings(processed_text)
    if has_highlights(processed_text):
        connect_highlights(processed_text)
        annotated_text(processed_text)
    else:
        st.write('В данном отзыве таких словосочетаний нет')
