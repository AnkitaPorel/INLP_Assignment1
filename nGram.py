import re
from collections import Counter
from tokenizer import tokenize_text, build_vocab
import os

def read_corpus(corpus_path):
    if not os.path.isfile(corpus_path):
        raise ValueError("Invalid corpus path provided.")
    with open(corpus_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_ngrams(tokenized_sentences, N):
    ngram_counts = Counter()
    
    for sentence in tokenized_sentences:
        sentence = ['<s>'] * (N - 1) + sentence + ['</s>']
        
        ngrams = zip(*[sentence[i:] for i in range(N)])
        
        ngram_counts.update(ngrams)
    
    return ngram_counts

if __name__ == "__main__":
    corpus_path = input("Enter the path to the corpus: ")
    N = int(input("Enter the value of N for N-grams: "))

    corpus_text = read_corpus(corpus_path)
    vocab = build_vocab(corpus_text, min_freq=2)
    tokenized_sentences = tokenize_text(corpus_text, vocab=vocab)

    ngram_counts = generate_ngrams(tokenized_sentences, N)

    print("Top N-grams:")
    for ngram, count in ngram_counts.most_common(10):
        print(f"{ngram}: {count}")
