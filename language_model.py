import re 
import math 
import sys
  
# storing system arguments
program = sys.argv[0]
n_value = sys.argv[1]
smoothing_type = sys.argv[2]
input_path = sys.argv[3]
  

def tokenize(text):
    """This function tokenizes the input text and returns the output.
     it is a custom tokenizers which specifically filters 
     urls, hashtags, mentions, numbers, percentages, dates and times by replacing them with appropriate tags as <TAG>"""
    text = str(text)
    # remove non-ascii characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # make the text lowercase
    cleaned_text = cleaned_text.lower()

    # replace urls
    cleaned_text = re.sub(r'http\S+', '<URL>', cleaned_text)

    # replace hashtags
    cleaned_text = re.sub(r'#\w+', '<HASHTAG>', cleaned_text)

    # replace mentions
    cleaned_text = re.sub(r'@\w+', '<MENTION>', cleaned_text)

    # replace percentages
    cleaned_text = re.sub(r'\d+%', '<PERCENTAGE>', cleaned_text)

    # replace all date formats
    cleaned_text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '<DATE>', cleaned_text)

    # replace all time formats
    cleaned_text = re.sub(r'\d{1,2}[:]\d{2}([:]\d{2})?(am|pm)?', '<TIME>', cleaned_text)

    # replace any repeated punctuation to a single one
    cleaned_text = re.sub(r'([.,!?:;><])\1+', r'\1', cleaned_text)

    # replace any letter repeated more than twice to a single one (eg: tiired to tired. oopppps -> oops)
    cleaned_text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', cleaned_text)

    # replace can't with cannot
    cleaned_text = re.sub(r'can\'t', 'cannot', cleaned_text)

    # replace xn't with x + not
    cleaned_text = re.sub(r'n\'t', r' not', cleaned_text)

    # replace x'm with x + am
    cleaned_text = re.sub(r'\'m', r' am', cleaned_text)

    # replace x's with x + is
    cleaned_text = re.sub(r'\'s', r' is', cleaned_text)

    # replace x're with x + are
    cleaned_text = re.sub(r'\'re', r' are', cleaned_text)

    # replace x'll to x + will
    cleaned_text = re.sub(r'\'ll', r' will', cleaned_text)

    # replace x'd to x + would
    cleaned_text = re.sub(r'\'d', r' would', cleaned_text)

    # replace x've to x + have
    cleaned_text = re.sub(r'\'ve', r' have', cleaned_text)

    # splitting words along with punctuation before or after (eg: good. -> good + . , "lmao" +> " + lmao + ", [Dang] -> [ + Dang + ])
    cleaned_text = re.sub(r'(\w+)([.,!?:;\[\]*/"\'\(\)])', r'\1 \2', cleaned_text)
    cleaned_text = re.sub(r'([.,!?:;\[\]*/"\'\(\)])(\w+)', r'\1 \2', cleaned_text)

    # remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # split words with hyphens
    cleaned_text = re.sub(r'(\w+)-(\w+)', r'\1 \2', cleaned_text)

    # tag numbers
    cleaned_text = re.sub(r'\d+', '<NUMBER>', cleaned_text)
    

    return cleaned_text

# initiating dictionaries to store ngrams and their N_scores
unigrams = {}
bigrams = {}
trigrams = {}
fourgrams = {}

unigrams_n = {}
bigrams_n = {}
trigrams_n = {}

kn_base = {}

def preprocess(sentence):
    """ This function preprocesses the input sentence by creating ngrams 
        and their N_scores and storing them in their respective dictionaries."""
    words = sentence.split()
    words = ['<s>'] * (4 - 1) + words + ['</s>'] * (4 - 1)

    for i in range(len(words) - 3):
        unigrams[(words[i]),] = unigrams.get((words[i],), 0) + 1
        if(bigrams.get((words[i], words[i + 1]), 0) == 0): # unique word following context
            kn_base[(words[i+1]),] = kn_base.get((words[i+1],), 0) + 1
            unigrams_n[(words[i]),] = unigrams_n.get((words[i],), 0) + 1
        bigrams[(words[i], words[i + 1])] = bigrams.get((words[i], words[i + 1]), 0) + 1
        if(trigrams.get((words[i], words[i + 1], words[i + 2]), 0) == 0): # unique word following context
            bigrams_n[(words[i], words[i + 1])] = bigrams_n.get((words[i], words[i + 1]), 0) + 1
        trigrams[(words[i], words[i + 1], words[i + 2])] = trigrams.get((words[i], words[i + 1], words[i + 2]), 0) + 1
        if(fourgrams.get((words[i], words[i + 1], words[i + 2], words[i + 3]), 0) == 0): # unique word following context
            trigrams_n[(words[i], words[i + 1], words[i + 2])] = trigrams_n.get((words[i], words[i + 1], words[i + 2]), 0) + 1
        fourgrams[(words[i], words[i + 1], words[i + 2], words[i + 3])] = fourgrams.get((words[i], words[i + 1], words[i + 2], words[i + 3]), 0) + 1

# reading input file
with open(input_path, 'r', encoding="utf8") as f:
    train_text = f.readlines()

# preprocessing input text sentence wise
for sentence in train_text:
    preprocess(tokenize(sentence))

# handling unknown words during training by tagging them as <UNK>
def handle_unknowns_train(sentence):
    words = sentence.split()
    for i in range(len(words)):
        if unigrams[(words[i]),] == 1:
            words[i] = "<UNK>"
            unigrams[(words[i]),] = 0
    sentence = " ".join(words)
    return sentence

# stores the preprocessed input text
cleaned_text = []

for sentence in train_text:
    sentence = handle_unknowns_train(tokenize(sentence))
    cleaned_text.append(sentence)

# dictionaries to store preprocessed input text
unigrams = {}
bigrams = {}
trigrams = {}
fourgrams = {}

unigrams_n = {}
bigrams_n = {}
trigrams_n = {}
kn_base = {}

for sentence in cleaned_text:
    preprocess(sentence)

ngrams = [unigrams, bigrams, trigrams, fourgrams]
ngrams_n = [unigrams_n, bigrams_n, trigrams_n]
kn_base_sum = sum(kn_base.values())

def count_of(ngram):
    """This function returns the count of an ngram."""
    n = len(ngram)
    counts = ngrams[n-1]
    if(n == 1):
        if(counts.get(tuple(ngram), 0) == 0):
            return unigrams["<UNK>",]
        else:
            return counts.get(tuple(ngram), 0)
    else:
        if(counts.get(tuple(ngram), 0) == 0):
            return 1e-6
        else:
            return counts.get((tuple(ngram)), 1e-6)

def count_kn(n, ngram):
    """This function returns the KN count of an ngram. (as described in Kneser Ney Smoothing methods)"""
    if(n == 4): # for highest order
        return count_of(ngram)
    else: # for lower orders, use continuation count(number of unique single word contexts for ngram)
        n = len(ngram)
        count = 0
        for higher_order_ngram in ngrams[n]:
            if higher_order_ngram[1:] == ngram:
                count += 1
        if(count == 0):
            return 1e-6
        else:
            return count

def n_count(ngram):
    """This function returns the N_count of an ngram.(as described in the Witten Bell Smoothing methods)"""
    counts = ngrams_n[len(ngram)-1]
    if(counts.get((tuple(ngram)), 0) == 0):
        return 1e-6
    else:
        return counts.get((tuple(ngram)), 0)

def KneyserNey(n, ngram):
    """This function returns the Kneser Ney smoothed probability of an ngram."""
    ngram = ngram.split()
    context = ngram[:-1]

    # discount factor
    d = 0.75
    first_term = max(count_of(tuple(ngram)) - d, 0)/count_of(tuple(context))
    lamb_term = (d/count_of(tuple(context)))*n_count(tuple(context))

    if(n == 1):
        return kn_base[tuple(ngram)]/kn_base_sum
    elif(n >= 2):
        return first_term + (lamb_term * KneyserNey(n-1, ' '.join(ngram[1:])))
    else:
        return -1

def witten_bell(n, ngram):
    """This function returns the Witten Bell smoothed probability of an ngram."""
    ngram = ngram.split()
    context = tuple(ngram[:-1])


    if(n == 1):
        return count_of(tuple(ngram))/sum(unigrams.values())
    else:
        total_count = count_of(context)
        Pml = count_of(tuple(ngram))/total_count
        endings = n_count(context)
        lamb = total_count/(max(1, total_count + endings))
    
        return lamb *Pml + (1 - lamb)*witten_bell(n-1, ' '.join(ngram[1:]))

# handling unknown words during testing by tagging them as <UNK>
def handle_unknowns(sentence):
    """This function handles unknown words in a sentence."""
    words = tokenize(sentence).split()
    for i in range(len(words)):
        if unigrams.get((words[i],), 0) == 0:
            words[i] = "<UNK>"
    return words
    

def get_probability(n, sentence, smooth):
    """This function returns the probability of a sentence by passing it through a smoothing method."""
    words = handle_unknowns(sentence)
    prob = 1
    for i in range(len(words) - 3):
        ngram = words[i:i+n]
        if(smooth == "w"):
            prob *= witten_bell(n, ' '.join(ngram))
        else:
            prob *= KneyserNey(n, ' '.join(ngram))
    return prob

# nth root of 1/probability
def get_perplexity(n, text, smooth):
    return math.pow(1/get_probability(n, text, smooth), 1/len(text.split()))

# printing the output after taking user input
sentence = input("Input Sentence: ")
print(get_probability(int(n_value), sentence, smoothing_type))

# Function Writing output to files
def write_output(input_path, output_path, smooth):
    with open( input_path , 'r', encoding="utf8") as f:
        text = f.readlines()

    avg_perplexity = 0
    perps = []
    for sentence in text: 
        # print(sentence)
        sent_perplexity = get_perplexity(4, sentence, smooth)
        perps.append(sentence.rstrip("\n") + "\t" + str(sent_perplexity) + "\n")
        avg_perplexity += sent_perplexity
    
    with open( output_path , 'w', encoding="utf8") as f:
        f.write("Average perplexity: " + str(avg_perplexity/len(text) ) + "\n")
        f.writelines(perps)

# template to write output to a file from input file    
# write_output("./europarl-corpus-test.txt", "./outputs/euro_kn_test.txt", "k")
