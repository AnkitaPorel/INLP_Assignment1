import random
import math
import sys
import numpy as np
import string
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from sklearn.linear_model import LinearRegression
from tokenizer import tokenize_text, build_vocab
from nGram import read_corpus, generate_ngrams

def laplace_smoothing(tokenized_sentence, n, ngram_counts, n_minus_1_counts, V):
    tokenized_sentence = ['<s>']*(n-1)+tokenized_sentence+['</s>']
    log_prob = 0
    for i in range(n - 1, len(tokenized_sentence)):
        ngram = tuple(tokenized_sentence[i - n + 1:i + 1])
        n_minus_1_gram = ngram[:-1]

        ngram_count = ngram_counts.get(ngram, 0)
        n_minus_1_count = n_minus_1_counts.get(n_minus_1_gram, 0)

        log_prob += math.log(ngram_count + 1) - math.log(n_minus_1_count + V)
    
    return log_prob

def good_turing_smoothing(ngram_counts):
    total_ngrams = sum(ngram_counts.values())
    freq_of_freqs = Counter(ngram_counts.values())
    r_values = sorted(freq_of_freqs.keys())

    log_r = np.log(r_values).reshape(-1, 1)
    log_Nr = np.log([freq_of_freqs[r] for r in r_values])
    model = LinearRegression().fit(log_r, log_Nr)
    a, b = model.intercept_, model.coef_[0]

    small_r_threshold = 2

    def S(r):
        if r <= small_r_threshold and r in freq_of_freqs:
            return freq_of_freqs[r]
        elif r > 0:
            return max(0, math.exp(a + b * math.log(r)))
        return 0

    smoothed_probs = {}
    for ngram, r in ngram_counts.items():
        S_r = S(r)
        S_r_plus_1 = S(r + 1) if r + 1 in freq_of_freqs else S(r + 1)
        r_star = ((r + 1) * S_r_plus_1 / S_r) if S_r > 0 else 0
        smoothed_probs[ngram] = math.log(r_star) - math.log(total_ngrams) if r_star > 0 else float('-inf')

    unseen_prob = max(0, math.log(freq_of_freqs.get(1, 1)) - math.log(total_ngrams))

    return smoothed_probs, unseen_prob

def calculate_perplexity(log_sum, N):
    if N==0:
        return float('inf')
    avg_log_prob = -log_sum / N
    perplexity = math.exp(avg_log_prob)
    return perplexity

def linear_interpolation(tokenized_sentence, unigram_counts, trigram_counts, fivegram_counts, bigram_counts, fourgram_counts, V, lambdas):
    uni = laplace_smoothing(tokenized_sentence, 1, unigram_counts, Counter(), V)
    N = len(tokenized_sentence)
    tri = laplace_smoothing(tokenized_sentence, 3, trigram_counts, bigram_counts, V)
    N = len(tokenized_sentence) + 2
    five = laplace_smoothing(tokenized_sentence, 5, fivegram_counts, fourgram_counts, V)
    N = len(tokenized_sentence) + 4
    interpolated_prob = (
        lambdas[0] * math.exp(uni)
        + lambdas[1] * math.exp(tri)
        + lambdas[2] * math.exp(five)
    )
    log_total_prob = 0
    if interpolated_prob > 0:
        log_total_prob += math.log(interpolated_prob)
    else:
        log_total_prob += float('-inf')

    return log_total_prob

def split_train_test(corpus, test_size=1000):
    random.shuffle(corpus)
    test_set = corpus[:test_size]
    train_set = corpus[test_size:]
    return train_set, test_set

if __name__ == "__main__":
    n = len(sys.argv)
    if(n!=3):
        print("Invalid no. of arguments")
        sys.exit(0)
    ip_sentence = input("Input sentence: ")
    corpus_path = sys.argv[2]
    corpus_text = read_corpus(corpus_path)
    vocabulary = set()
    tokenized_sentences = tokenize_text(corpus_text)
    for sentence in tokenized_sentences:
        vocabulary.update(sentence)

    V = len(vocabulary)+1

    unigram_counts = generate_ngrams(tokenized_sentences, 1)
    punctuation = set(string.punctuation)
    stop_words = set(stopwords.words('english'))
    unigram_counts = Counter({
        word: count
        for word, count in unigram_counts.items()
        if word not in punctuation and word.lower() not in stop_words
    })

    trigram_counts = generate_ngrams(tokenized_sentences, 3)
    fivegram_counts = generate_ngrams(tokenized_sentences, 5)
    fourgram_counts = generate_ngrams(tokenized_sentences, 4)
    bigram_counts = generate_ngrams(tokenized_sentences, 2)

    unigram_gt_probs, unigram_unseen_prob = good_turing_smoothing(unigram_counts)
    trigram_gt_probs, trigram_unseen_prob = good_turing_smoothing(trigram_counts)
    fivegram_gt_probs, fivegram_unseen_prob = good_turing_smoothing(fivegram_counts)

    lambdas = [0.3, 0.5, 0.2]

    SMALL_VALUE = 1e-10

    tokenize_ip = tokenize_text(ip_sentence)

    for sentence in tokenize_ip:
        if(sys.argv[1]=='l'):
            uni = laplace_smoothing(sentence, 1, unigram_counts, Counter(), V)
            print(uni)
            N = len(sentence)
            lap_uni_perplexity = calculate_perplexity(uni, N)
            print(f"Unigram Perplexity: {lap_uni_perplexity}")
            tri = laplace_smoothing(sentence, 3, trigram_counts, bigram_counts, V)
            print(tri)
            N = len(sentence) + 2
            lap_tri_perplexity = calculate_perplexity(tri, N)
            print(f"Trigram Perplexity: {lap_tri_perplexity}")
            five = laplace_smoothing(sentence, 5, fivegram_counts, fourgram_counts, V)
            print(five)
            N = len(sentence) + 4
            lap_five_perplexity = calculate_perplexity(five, N)
            print(f"Fivegram Perplexity: {lap_five_perplexity}")

        elif(sys.argv[1]=='g'):
            unigram_prob = 1
            for unigram in sentence:
                unigram_prob *= unigram_gt_probs.get((unigram,), unigram_unseen_prob)
            print(f"Unigram Model - Probability: {unigram_prob}")
            N = len(sentence)
            unigram_log_prob = 0
            if(unigram_prob>0):
                unigram_log_prob = math.log(unigram_prob)
            else:
                unigram_log_prob = math.log(SMALL_VALUE)
            if(N!=0):
                perplexity = calculate_perplexity(unigram_log_prob, N)
            else:
                perplexity = calculate_perplexity(unigram_log_prob, N+1)
            print(f"Unigram Perplexity: {perplexity}")
            trigram_prob = 1
            trigram_sentence = ['<s>']*2 + sentence + ['</s>']
            for i in range(len(trigram_sentence) - 1):
                trigram = tuple(trigram_sentence[i:i + 2])
                trigram_prob *= trigram_gt_probs.get(trigram, trigram_unseen_prob)
            print(f"Trigram Model - Probability: {trigram_prob}")
            N = len(sentence) + 2
            trigram_log_prob = 0
            if(trigram_prob>0):
                trigram_log_prob = math.log(trigram_prob)
            else:
                trigram_log_prob = math.log(SMALL_VALUE)
            perplexity = calculate_perplexity(trigram_log_prob, N)
            print(f"Trigram Perplexity: {perplexity}")
            fivegram_prob = 1
            fivegram_sentence = ['<s>']*4 + sentence + ['</s>']
            for i in range(len(fivegram_sentence) - 2):
                fivegram = tuple(fivegram_sentence[i:i + 3])
                fivegram_prob *= fivegram_gt_probs.get(fivegram, fivegram_unseen_prob)
            print(f"Fivegram Model - Probability: {fivegram_prob}")
            N = len(sentence) + 4
            fivegram_log_prob = 0
            if(fivegram_prob>0):
                fivegram_log_prob = math.log(fivegram_prob)
            else:
                fivegram_log_prob = math.log(SMALL_VALUE)
            perplexity = calculate_perplexity(fivegram_log_prob, N)
            print(f"Fivegram Perplexity: {perplexity}")

        elif(sys.argv[1]=='i'):
            interp_prob = linear_interpolation(sentence, unigram_counts, trigram_counts, fivegram_counts, bigram_counts, fourgram_counts, V, lambdas)
            print(f"Linear Interpolation Model - Probability: {interp_prob}")
            N = len(sentence) + 4
            perplexity = calculate_perplexity(interp_prob, N)
            print(f"Interpolation Model- Perplexity: {perplexity}")
            
    # corpus_path = input()
    # corpus_text = read_corpus(corpus_path)
    # vocab = build_vocab(corpus_text, min_freq=2)
    # tokenized_sentences = tokenize_text(corpus_text, vocab=vocab)
    # train_sentences, test_sentences = split_train_test(tokenized_sentences, test_size=1000)
    # vocabulary = set()
    # tokenized_sentences = tokenize_text(corpus_text)
    # for sentence in tokenized_sentences:
    #     vocabulary.update(sentence)

    # unigram_counts = generate_ngrams(tokenized_sentences, 1)
    # trigram_counts = generate_ngrams(tokenized_sentences, 3)
    # fivegram_counts = generate_ngrams(tokenized_sentences, 5)
    # fourgram_counts = generate_ngrams(tokenized_sentences, 4)
    # bigram_counts = generate_ngrams(tokenized_sentences, 2)

    # punctuation = set(string.punctuation)
    # stop_words = set(stopwords.words('english'))
    # unigram_counts = Counter({
    #     ngram: count
    #     for ngram, count in unigram_counts.items()
    #     if ngram[0] not in punctuation and ngram[0].lower() not in stop_words
    # })
    # trigram_counts = Counter({
    #     ngram: count
    #     for ngram, count in trigram_counts.items()
    #     if ngram[0] not in punctuation and ngram[0].lower() not in stop_words
    # })
    # fivegram_counts = Counter({
    #     ngram: count
    #     for ngram, count in fivegram_counts.items()
    #     if ngram[0] not in punctuation and ngram[0].lower() not in stop_words
    # })
    # unigram_gt_probs, unigram_unseen_prob = good_turing_smoothing(unigram_counts)
    # trigram_gt_probs, trigram_unseen_prob = good_turing_smoothing(trigram_counts)
    # fivegram_gt_probs, fivegram_unseen_prob = good_turing_smoothing(fivegram_counts)

    # V = len(vocabulary)+1
    # SMALL_VALUE = 1e-10

    # lambdas = [0.2, 0.5, 0.3]

    # tot_five_lap_perplexity = 0

    # train_output_file = "2024201043_LM2_5_train-perplexity.txt"
    # test_output_file = "2024201043_LM2_5_test_perplexity.txt"
    # with open(train_output_file, "w") as train_file, open(test_output_file, "w") as test_file:
        # unigram_prob = 1
        # trigram_prob = 1
        # fivegram_prob = 1
        # tot_five_lap_perplexity = 0
        # for sentence in train_sentences:
            # for unigram in sentence:
            #     unigram_prob *= unigram_gt_probs.get((unigram,), unigram_unseen_prob)
            # N = len(sentence)
            # unigram_log_prob = 0
            # if(unigram_prob>0):
            #     unigram_log_prob = math.log(unigram_prob)
            # else:
            #     unigram_log_prob = math.log(SMALL_VALUE)
            # if(N!=0):
            #     perplexity = calculate_perplexity(unigram_log_prob, N)
            # else:
            #     perplexity = calculate_perplexity(unigram_log_prob, N+1)

            # trigram_sentence = ['<s>']*2 + sentence + ['</s>']
            # for i in range(len(trigram_sentence) - 1):
            #     trigram = tuple(trigram_sentence[i:i + 2])
            #     trigram_prob *= trigram_gt_probs.get(trigram, trigram_unseen_prob)
            # N = len(sentence) + 2
            # trigram_log_prob = 0
            # if(trigram_prob>0):
            #     trigram_log_prob = math.log(trigram_prob)
            # else:
            #     trigram_log_prob = math.log(SMALL_VALUE)
            # perplexity = calculate_perplexity(trigram_log_prob, N)
        #     fivegram_sentence = ['<s>']*4 + sentence + ['</s>']
        #     for i in range(len(fivegram_sentence) - 2):
        #         fivegram = tuple(fivegram_sentence[i:i + 3])
        #         fivegram_prob *= fivegram_gt_probs.get(fivegram, fivegram_unseen_prob)
        #     N = len(sentence) + 4
        #     fivegram_log_prob = 0
        #     if(fivegram_prob>0):
        #         fivegram_log_prob = math.log(fivegram_prob)
        #     else:
        #         fivegram_log_prob = math.log(SMALL_VALUE)
        #     perplexity = calculate_perplexity(fivegram_log_prob, N)
        #     if(perplexity <= 10000):
        #         tot_five_lap_perplexity += perplexity
        # tot_five_lap_perplexity = tot_five_lap_perplexity / len(train_sentences)
        # train_file.write(f"{tot_five_lap_perplexity}\n")

        # for sentence in train_sentences:
            # for unigram in sentence:
            #     unigram_prob *= unigram_gt_probs.get((unigram,), unigram_unseen_prob)
            # N = len(sentence)
            # unigram_log_prob = 0
            # if(unigram_prob>0):
            #     unigram_log_prob = math.log(unigram_prob)
            # else:
            #     unigram_log_prob = math.log(SMALL_VALUE)
            # if(N!=0):
            #     perplexity = calculate_perplexity(unigram_log_prob, N)
            # else:
            #     perplexity = calculate_perplexity(unigram_log_prob, N+1)
            # trigram_sentence = ['<s>']*2 + sentence + ['</s>']
            # for i in range(len(trigram_sentence) - 1):
            #     trigram = tuple(trigram_sentence[i:i + 2])
            #     trigram_prob *= trigram_gt_probs.get(trigram, trigram_unseen_prob)
            # N = len(sentence) + 2
            # trigram_log_prob = 0
            # if(trigram_prob>0):
            #     trigram_log_prob = math.log(trigram_prob)
            # else:
            #     trigram_log_prob = math.log(SMALL_VALUE)
            # perplexity = calculate_perplexity(trigram_log_prob, N)
            # fivegram_sentence = ['<s>']*4 + sentence + ['</s>']
            # for i in range(len(fivegram_sentence) - 2):
            #     fivegram = tuple(fivegram_sentence[i:i + 3])
            #     fivegram_prob *= fivegram_gt_probs.get(fivegram, fivegram_unseen_prob)
            # N = len(sentence) + 4
            # fivegram_log_prob = 0
            # if(fivegram_prob>0):
            #     fivegram_log_prob = math.log(fivegram_prob)
            # else:
            #     fivegram_log_prob = math.log(SMALL_VALUE)
            # perplexity = calculate_perplexity(fivegram_log_prob, N)
            # sentence_str = ' '.join(sentence)
            # train_file.write(f"{sentence_str} {perplexity}\n")

        # unigram_prob = 1
        # trigram_prob = 1
        # fivegram_prob = 1
        # tot_five_lap_perplexity = 0
        # for sentence in test_sentences:
            # for unigram in sentence:
            #     unigram_prob *= unigram_gt_probs.get((unigram,), unigram_unseen_prob)
            # N = len(sentence)
            # unigram_log_prob = 0
            # if(unigram_prob>0):
            #     unigram_log_prob = math.log(unigram_prob)
            # else:
            #     unigram_log_prob = math.log(SMALL_VALUE)
            # if(N!=0):
            #     perplexity = calculate_perplexity(unigram_log_prob, N)
            # else:
            #     perplexity = calculate_perplexity(unigram_log_prob, N+1)
            # trigram_sentence = ['<s>']*2 + sentence + ['</s>']
            # for i in range(len(trigram_sentence) - 1):
            #     trigram = tuple(trigram_sentence[i:i + 2])
            #     trigram_prob *= trigram_gt_probs.get(trigram, trigram_unseen_prob)
            # N = len(sentence) + 2
            # trigram_log_prob = 0
            # if(trigram_prob>0):
            #     trigram_log_prob = math.log(trigram_prob)
            # else:
            #     trigram_log_prob = math.log(SMALL_VALUE)
            # perplexity = calculate_perplexity(trigram_log_prob, N)
        #     fivegram_sentence = ['<s>']*4 + sentence + ['</s>']
        #     for i in range(len(fivegram_sentence) - 2):
        #         fivegram = tuple(fivegram_sentence[i:i + 3])
        #         fivegram_prob *= fivegram_gt_probs.get(fivegram, fivegram_unseen_prob)
        #     N = len(sentence) + 4
        #     fivegram_log_prob = 0
        #     if(fivegram_prob>0):
        #         fivegram_log_prob = math.log(fivegram_prob)
        #     else:
        #         fivegram_log_prob = math.log(SMALL_VALUE)
        #     perplexity = calculate_perplexity(fivegram_log_prob, N)
        #     if(perplexity <= 10000):
        #         tot_five_lap_perplexity += perplexity
        # tot_five_lap_perplexity = tot_five_lap_perplexity / len(test_sentences)
        # test_file.write(f"{tot_five_lap_perplexity}\n")

        # for sentence in test_sentences:
            # for unigram in sentence:
            #     unigram_prob *= unigram_gt_probs.get((unigram,), unigram_unseen_prob)
            # N = len(sentence)
            # unigram_log_prob = 0
            # if(unigram_prob>0):
            #     unigram_log_prob = math.log(unigram_prob)
            # else:
            #     unigram_log_prob = math.log(SMALL_VALUE)
            # if(N!=0):
            #     perplexity = calculate_perplexity(unigram_log_prob, N)
            # else:
            #     perplexity = calculate_perplexity(unigram_log_prob, N+1)
            # trigram_sentence = ['<s>']*2 + sentence + ['</s>']
            # for i in range(len(trigram_sentence) - 1):
            #     trigram = tuple(trigram_sentence[i:i + 2])
            #     trigram_prob *= trigram_gt_probs.get(trigram, trigram_unseen_prob)
            # N = len(sentence) + 2
            # trigram_log_prob = 0
            # if(trigram_prob>0):
            #     trigram_log_prob = math.log(trigram_prob)
            # else:
            #     trigram_log_prob = math.log(SMALL_VALUE)
            # perplexity = calculate_perplexity(trigram_log_prob, N)
            # fivegram_sentence = ['<s>']*4 + sentence + ['</s>']
            # for i in range(len(fivegram_sentence) - 2):
            #     fivegram = tuple(fivegram_sentence[i:i + 3])
            #     fivegram_prob *= fivegram_gt_probs.get(fivegram, fivegram_unseen_prob)
            # N = len(sentence) + 4
            # fivegram_log_prob = 0
            # if(fivegram_prob>0):
            #     fivegram_log_prob = math.log(fivegram_prob)
            # else:
            #     fivegram_log_prob = math.log(SMALL_VALUE)
            # perplexity = calculate_perplexity(fivegram_log_prob, N)
            # sentence_str = ' '.join(sentence)
            # test_file.write(f"{sentence_str} {perplexity}\n")

        # tot_five_lap_perplexity = 0
        # for sentence in train_sentences:
        #     five = laplace_smoothing(sentence, 5, fivegram_counts, fourgram_counts, V)
        #     N = len(sentence) + 4
        #     lap_five_perplexity = calculate_perplexity(five, N)
        #     tot_five_lap_perplexity += lap_five_perplexity
        # tot_five_lap_perplexity = tot_five_lap_perplexity / len(train_sentences)
        # train_file.write(f"{tot_five_lap_perplexity}\n")
        
        # for sentence in train_sentences:
        #     five = laplace_smoothing(sentence, 5, fivegram_counts, fourgram_counts, V)
        #     N = len(sentence) + 4
        #     lap_five_perplexity = calculate_perplexity(five, N)
        #     sentence_str = ' '.join(sentence)
        #     train_file.write(f"{sentence_str} {lap_five_perplexity}\n")

        # tot_five_lap_perplexity = 0

        # for sentence in test_sentences:
        #     five = laplace_smoothing(sentence, 5, fivegram_counts, fourgram_counts, V)
        #     N = len(sentence) + 4
        #     lap_five_perplexity = calculate_perplexity(five, N)
        #     tot_five_lap_perplexity += lap_five_perplexity
        # tot_five_lap_perplexity = tot_five_lap_perplexity / len(test_sentences)
        # test_file.write(f"{tot_five_lap_perplexity}\n")

        # for sentence in test_sentences:
        #     five = laplace_smoothing(sentence, 5, fivegram_counts, fourgram_counts, V)
        #     N = len(sentence) + 4
        #     lap_five_perplexity = calculate_perplexity(five, N)
        #     sentence_str = ' '.join(sentence)
        #     test_file.write(f"{sentence_str} {lap_five_perplexity}\n")
        # tot_five_lap_perplexity = 0
        # for sentence in train_sentences:
        #     tri = laplace_smoothing(sentence, 3, trigram_counts, bigram_counts, V)
        #     N = len(sentence) + 2
        #     lap_tri_perplexity = calculate_perplexity(tri, N)

        #     tot_five_lap_perplexity += lap_tri_perplexity
        # tot_five_lap_perplexity = tot_five_lap_perplexity / len(train_sentences)
        # train_file.write(f"{tot_five_lap_perplexity}\n")
        
        # for sentence in train_sentences:
        #     tri = laplace_smoothing(sentence, 3, trigram_counts, bigram_counts, V)
        #     N = len(sentence) + 2
        #     lap_tri_perplexity = calculate_perplexity(tri, N)
        #     sentence_str = ' '.join(sentence)
        #     train_file.write(f"{sentence_str} {lap_tri_perplexity}\n")

        # tot_five_lap_perplexity = 0

        # for sentence in test_sentences:
        #     tri = laplace_smoothing(sentence, 3, trigram_counts, bigram_counts, V)
        #     N = len(sentence) + 2
        #     lap_tri_perplexity = calculate_perplexity(tri, N)
        #     tot_five_lap_perplexity += lap_tri_perplexity
        # tot_five_lap_perplexity = tot_five_lap_perplexity / len(test_sentences)
        # test_file.write(f"{tot_five_lap_perplexity}\n")

        # for sentence in test_sentences:
        #     tri = laplace_smoothing(sentence, 3, trigram_counts, bigram_counts, V)
        #     N = len(sentence) + 2
        #     lap_tri_perplexity = calculate_perplexity(tri, N)
        #     sentence_str = ' '.join(sentence)
        #     test_file.write(f"{sentence_str} {lap_tri_perplexity}\n")
