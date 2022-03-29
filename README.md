# The Language Model: Kneser Ney and Witten Bell Smoothing

This project contains the code to prepare a language model out of a dataset. A language model  is a probabilistic statistical model that determines the probability of a given sequence of words occurring in a sentence based on the previous words.

The code contains two parts: 

1. Customized Tokenization: Cleans the input text
2. Smoothing: Adjustment of the maximum likelihood estimator of a language model to make it more accurate. 

There are two smoothing methods that can be used, Kneser Ney Smoothing and Witten Bell smoothing. 

The datasets used to train and evaluate the language models can be found here: [Link](https://iiitaphyd-my.sharepoint.com/personal/sagar_joshi_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsagar%5Fjoshi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2Fintro%5Fto%5Fnlp%5Fasign1&ga=1)

The methods used to code the language models, their results along with other details regarding the code are available in the report. 

## How to Run

In the parent directory, run the language model by typing
```
language_model.py <n_value> <smoothing type> <path to corpus>

```
where 
- n_value: the ngram value used for smoothing process. eg: 1
- smoothing_type: k for Kneyser-Ney and w for Witten-Bell
- path_to_corpus: eg: ./test-corpus.txt

Input the sentence, the LM will output the probability of the sentence occuring in the given corpus
eg: 
``` Input Sentence:  I am a man.
Input Sentence: This is working. 
0.28539
```



