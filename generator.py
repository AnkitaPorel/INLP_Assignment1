import sys
import math
from collections import Counter
from tokenizer import tokenize_text
from nGram import read_corpus, generate_ngrams
from language_model import laplace_smoothing, good_turing_smoothing, linear_interpolation, calculate_perplexity

def predict_next_word_laplace(sentence, n, ngram_counts, n_minus_1_counts, vocab, V, k):
    sentence = ['<s>'] * (n - 1) + sentence
    context = tuple(sentence[-(n-1):])
    candidates = {}
    for word in vocab:
        ngram = context + (word,)
        ngram_count = ngram_counts.get(ngram, 0)
        n_minus_1_count = n_minus_1_counts.get(context, 0)
        prob = (ngram_count + 1) / (n_minus_1_count + V)
        candidates[word] = prob

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:k]

def predict_next_word_good_turing(sentence, n, ngram_counts, smoothed_probs, unseen_prob, k):
    sentence = ['<s>'] * (n - 1) + sentence
    candidates = Counter()
    
    for word in ngram_counts:
        if len(word) == n and word[:-1] == tuple(sentence[-(n-1):]):
            prob = math.exp(smoothed_probs.get(word, unseen_prob))
            candidates[word[-1]] = prob
    
    return candidates.most_common(k)

def predict_next_word_interpolation(sentence, vocab, unigram_counts, trigram_counts, fivegram_counts, bigram_counts, fourgram_counts, V, lambdas, k):
    context = sentence
    candidates = {}

    for word in vocab:
        sent=context+[word]
        log_prob = linear_interpolation(
            sent, unigram_counts, trigram_counts, fivegram_counts,
            bigram_counts, fourgram_counts, V, lambdas
        )
        N = len(sent) + 4
        perplexity = calculate_perplexity(log_prob, N)
        candidates[word] = perplexity

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])
    return sorted_candidates[:k]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        sys.exit(0)
    
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])
    
    corpus_text = read_corpus(corpus_path)
    tokenized_sentences = tokenize_text(corpus_text)

    vocabulary = set()
    for sentence in tokenized_sentences:
        vocabulary.update(sentence)
    
    V = len(vocabulary) + 1
    
    unigram_counts = generate_ngrams(tokenized_sentences, 1)
    trigram_counts = generate_ngrams(tokenized_sentences, 3)
    fourgram_counts = generate_ngrams(tokenized_sentences, 4)
    bigram_counts = generate_ngrams(tokenized_sentences, 2)
    fivegram_counts = generate_ngrams(tokenized_sentences, 5)
    
    if lm_type == 'g':
        unigram_gt_probs, unigram_unseen_prob = good_turing_smoothing(unigram_counts)
        trigram_gt_probs, trigram_unseen_prob = good_turing_smoothing(trigram_counts)
        fivegram_gt_probs, fivegram_unseen_prob = good_turing_smoothing(fivegram_counts)
    
    if lm_type == 'i':
        lambdas = [0.2, 0.4, 0.4]
    
    while True:
        input_sentence = input("Input sentence: ")
        tokenized_sentence = tokenize_text(input_sentence)[0]
        
        if lm_type == 'l':
            next_words = predict_next_word_laplace(
                tokenized_sentence, 3, trigram_counts, bigram_counts, vocabulary, V, k
            )
        elif lm_type == 'g':
            next_words = predict_next_word_good_turing(
                tokenized_sentence, 3, trigram_counts, trigram_gt_probs, trigram_unseen_prob, k
            )
        elif lm_type == 'i':
            next_words = predict_next_word_interpolation(
                tokenized_sentence, vocabulary, unigram_counts, trigram_counts, fivegram_counts, bigram_counts, fourgram_counts, V, lambdas, k
            )
        else:
            print("Invalid LM type. Choose from 'l', 'g', or 'i'.")
            continue
        
        print("Predicted next words:")
        for word, prob in next_words:
            print(f"{word} {prob:.6f}")
