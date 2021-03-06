# Importing libraries
import nltk
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import pandas as pd
from IPython.display import display
import random
from sklearn.model_selection import train_test_split
from nltk.corpus import PlaintextCorpusReader
import pprint
import time


def main():
    # download the treebank corpus from nltk
    nltk.download('treebank')

    # download the universal tagset from nltk
    nltk.download('universal_tagset')

    # load in the corpus via the reader
    root = 'C:/Users/noahm/Desktop/School/Spring 2022/CISC489/nlp-assignment-homework2/text/foo.txt'
    text_file = open(root, "r")
    # read whole file to a string
    data = text_file.read()
    # close file
    text_file.close()

    sent_text = nltk.sent_tokenize(data)  # this gives us a list of sentences
    # now loop over each sentence and tokenize it separately
    tagged = []
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)
        tagged.append(nltk.pos_tag(tokenized_text, tagset='universal'))
    print(tagged)

    # reading the Treebank tagged sentences
    nltk_data = tagged

    # print the first two sentences along with tags
    print(nltk_data[:2])

    for sent in nltk_data[:2]:
        for tuple in sent:
            print(tuple)

    # split data into training and validation set in the ratio 80:20
    train_set, test_set = train_test_split(
        nltk_data, train_size=0.80, test_size=0.20, random_state=101)
    # create list of train and test tagged words
    train_tagged_words = [tup for sent in train_set for tup in sent]
    test_tagged_words = [tup for sent in test_set for tup in sent]
    print(len(train_tagged_words))
    print(len(test_tagged_words))

    # use set datatype to check how many unique tags are present in training data
    tags = {tag for word, tag in train_tagged_words}
    print(len(tags))
    print(tags)

    # check total words in vocabulary
    vocab = {word for word, tag in train_tagged_words}

    # compute Emission Probability
    def word_given_tag(word, tag, train_bag=train_tagged_words):
        tag_list = [pair for pair in train_bag if pair[1] == tag]
        # total number of times the passed tag occurred in train_bag
        count_tag = len(tag_list)
        w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    # now calculate the total number of times the passed word occurred as the passed tag.
        count_w_given_tag = len(w_given_tag_list)

        return (count_w_given_tag, count_tag)

    # compute  Transition Probability
    def t2_given_t1(t2, t1, train_bag=train_tagged_words):
        tags = [pair[1] for pair in train_bag]
        count_t1 = len([t for t in tags if t == t1])
        count_t2_t1 = 0
        for index in range(len(tags)-1):
            if tags[index] == t1 and tags[index+1] == t2:
                count_t2_t1 += 1
        return (count_t2_t1, count_t1)

    # creating t x t transition matrix of tags, t= no of tags
    # Matrix(i, j) represents P(jth tag after the ith tag)

    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)):
            tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

    print(tags_matrix)

    # convert the matrix to a df for better readability
    # the table is same as the transition table shown in section 3 of article
    tags_df = pd.DataFrame(tags_matrix, columns=list(tags), index=list(tags))
    display(tags_df)

    def Viterbi(words, train_bag=train_tagged_words):
        state = []
        T = list(set([pair[1] for pair in train_bag]))

        for key, word in enumerate(words):
            # initialise list of probability column for a given observation
            p = []
            for tag in T:
                if key == 0:
                    transition_p = tags_df.loc['.', tag]
                else:
                    transition_p = tags_df.loc[state[-1], tag]

                # compute emission and state probabilities
                emission_p = word_given_tag(words[key], tag)[
                    0]/word_given_tag(words[key], tag)[1]
                state_probability = emission_p * transition_p
                p.append(state_probability)

            pmax = max(p)
            # getting state for which probability is maximum
            state_max = T[p.index(pmax)]
            state.append(state_max)
        return list(zip(words, state))

    # Let's test our Viterbi algorithm on a few sample sentences of test dataset
    # define a random seed to get same sentences when run multiple times
    random.seed(1234)

    # choose random 10 numbers, added a -1 NOAH
    rndom = [random.randint(1, len(test_set)-1) for x in range(10)]

    # list of 10 sents on which we test the model
    test_run = [test_set[i] for i in rndom]

    # list of tagged words
    test_run_base = [tup for sent in test_run for tup in sent]

    # list of untagged words
    test_tagged_words = [tup[0] for sent in test_run for tup in sent]

    # Here We will only test 10 sentences to check the accuracy
    # as testing the whole training set takes huge amount of time
    start = time.time()
    tagged_seq = Viterbi(test_tagged_words)
    end = time.time()
    difference = end-start

    print("Time taken in seconds: ", difference)

    # accuracy
    check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]

    accuracy = len(check)/len(tagged_seq)
    print('Viterbi Algorithm Accuracy: ', accuracy*100)

    # To improve the performance,we specify a rule base tagger for unknown words
    # specify patterns for tagging
    patterns = [
        (r'.*ing$', 'VERB'),              # gerund
        (r'.*ed$', 'VERB'),               # past tense
        (r'.*es$', 'VERB'),               # verb
        (r'.*\'s$', 'NOUN'),              # possessive nouns
        (r'.*s$', 'NOUN'),                # plural nouns
        (r'\*T?\*?-[0-9]+$', 'X'),        # X
        (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'),  # cardinal numbers
        (r'.*', 'NOUN')                   # nouns
    ]

    # rule based tagger
    rule_based_tagger = nltk.RegexpTagger(patterns)

    # modified Viterbi to include rule based tagger in it
    def Viterbi_rule_based(words, train_bag=train_tagged_words):
        state = []
        T = list(set([pair[1] for pair in train_bag]))

        for key, word in enumerate(words):
            # initialise list of probability column for a given observation
            p = []
            for tag in T:
                if key == 0:
                    transition_p = tags_df.loc['.', tag]
                else:
                    transition_p = tags_df.loc[state[-1], tag]

                # compute emission and state probabilities
                emission_p = word_given_tag(words[key], tag)[
                                            0]/word_given_tag(words[key], tag)[1]
                state_probability = emission_p * transition_p
                p.append(state_probability)

            pmax = max(p)
            state_max = rule_based_tagger.tag([word])[0][1]

            if(pmax == 0):
                # assign based on rule based tagger
                state_max = rule_based_tagger.tag([word])[0][1]
            else:
                if state_max != 'X':
                    # getting state for which probability is maximum
                    state_max = T[p.index(pmax)]

            state.append(state_max)
        return list(zip(words, state))

    # test accuracy on subset of test data
    start = time.time()
    tagged_seq = Viterbi_rule_based(test_tagged_words)
    end = time.time()
    difference = end-start

    print("Time taken in seconds: ", difference)

    # accuracy
    check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]

    accuracy = len(check)/len(tagged_seq)
    print('Viterbi Algorithm Accuracy: ', accuracy*100)

    # Check how a sentence is tagged by the two POS taggers
    # and compare them
    test_sent = "Hobbits name is Fred"
    pred_tags_rule = Viterbi_rule_based(test_sent.split())
    pred_tags_withoutRules = Viterbi(test_sent.split())
    print(pred_tags_rule)
    print(pred_tags_withoutRules)
    # https://www.pythonpool.com/viterbi-algorithm-python/

    observations = ("1", "2", "3", "2", "2", "2")
    states = ("Hot", "Cold")
    start_p = {"Hot": 0.6, "Cold": 0.4}
    trans_p = {
        "Hot": {"Hot": 0.5, "Cold": 0.5},
        "Cold": {"Hot": 1, "Cold": 0},
    }
    emit_p = {
        "Hot": {"1": 1, "2": 0.658, "3": 1},
        "Cold": {"1": 0.0, "2": 0.342, "3": 0.0},
    }
    # target H C H H H C H
    def icecream_viterbi(observations, states, start_p, trans_p, emit_p):
        V = [{}]
        for st in states:
            V[0][st] = {"prob": start_p[st] * emit_p[st]
                [observations[0]], "prev": None}

        for t in range(1, len(observations)):
            V.append({})
            for st in states:
                max_tr_prob = V[t - 1][states[0]]["prob"] * \
                    trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st

                max_prob = max_tr_prob * emit_p[st][observations[t]]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
        for line in dptable(V):
            print(line)

        opt = []
        max_prob = 0.0
        best_st = None

        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        print("The steps of states are " + " ".join(opt) +
              " with highest probability of %s" % max_prob)

    def dptable(V):

        yield " ".join(("%12d" % i) for i in range(len(V)))
        for state in V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

    icecream_viterbi(observations, states, start_p, trans_p, emit_p)

if __name__ == '__main__':
    main()
