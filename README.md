FOLDER STRUCTURE

2024201043_INLP_1
    |
    |--generator.py
    |
    |--language_model.py
    |
    |--nGram.py
    |
    |--tokenizer.py
    |
    |--README.md
    |
    |--Pride&Prejudice
    |   |
    |   |--2024201043_LMi_i_train/test-perplexity.txt
    |
    |--Ulysses
    |   |
    |   |--2024201043_LMi_i_train/test-perplexity.txt

# To run tokenizer.py
python3 tokenizer.py
>> Code will prompt you to input sentences
>> It outputs the tokenized sentences
>> Uses regular expression

# To run language_model.py
python3 language_model.py <lm_type> <corpus_path>
>> LM type can be l for Laplace Smoothing, g for Good-Turing Smoothing. Model and i for Interpolation Model.
>> On running the file, the expected output is a prompt, which asks for a sentence and provides the probability of that sentence using the given mechanism.
>> NLTK library needs to be installed (Command to install- pip install -U nltk) and Scikit learn for linear regression (Command to install- pip install scikit-learn)

# To run nGram.py
python3 nGram.py
>> On running, it asks for Corpus path and the value of N for generating N-gram.
>> Outputs the N-gram of the input corpus.

# To run generator.py
python3 generator.py <lm_type> <corpus_path> <k>
>> LM type can be l for Laplace Smoothing, g for Good-Turing Smoothing. Model and i for Interpolation Model. k denotes the number of candidates for the next word to be printed.
>> On running the file, the expected output is a prompt, which asks for a sentence and outputs the most probable next word of the sentence along with it’s probability score using the given mechanism.
