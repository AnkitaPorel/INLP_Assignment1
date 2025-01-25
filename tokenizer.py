import re
from collections import Counter

def build_vocab(corpus_text, min_freq=2):
    tokenized_sentences = tokenize_text(corpus_text)
    word_counts = Counter(word for sentence in tokenized_sentences for word in sentence)

    vocab = {word for word, count in word_counts.items() if count >= min_freq}
    vocab.add('<UNK>')

    return vocab

def tokenize_text(input_text, vocab=None):
    url_pattern = r'(https?://\S+?)([.,!?;])?(?=\s|$)'
    hashtag_pattern = r'#\w+'
    mention_pattern = r'@\w+'
    percentage_pattern = r'\b\d+(\.\d+)?%'
    time_pattern = r'\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?'
    age_pattern = r'\b\d{1,3}\s?(?:years?\s?old|y/o)\b'
    time_period_pattern = r'\b\d{1,4}\s?(?:BC|AD|BCE|CE)\b'

    input_text = re.sub(url_pattern, r'<URL>\2', input_text)
    input_text = re.sub(hashtag_pattern, '<HASHTAG>', input_text)
    input_text = re.sub(mention_pattern, '<MENTION>', input_text)
    input_text = re.sub(percentage_pattern, '<PERCENTAGE>', input_text)
    input_text = re.sub(time_pattern, '<TIME>', input_text)
    input_text = re.sub(age_pattern, '<AGE>', input_text)
    input_text = re.sub(time_period_pattern, '<TIME_PERIOD>', input_text)

    input_text = re.sub(r'(<[A-Z_]+>)([.,!?;])', r'\1 \2', input_text)

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', input_text)

    tokenized_sentences = []
    for sentence in sentences:
        tokens = re.findall(
            r'\"|\'|[a-zA-Z]+|<URL>|<HASHTAG>|<MENTION>|<PERCENTAGE>|<TIME>|<AGE>|<TIME_PERIOD>|[.,!?;]',
            sentence
        )
        if vocab:
            tokens = [word if word in vocab else '<UNK>' for word in tokens]
        if tokens:
            tokenized_sentences.append(tokens)

    return tokenized_sentences

if __name__ == "__main__":
    input_text = input("Enter your text: ")
    tokenized = tokenize_text(input_text)
    print("Tokenized text:", tokenized)

