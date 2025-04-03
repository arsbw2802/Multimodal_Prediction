
import os
import joblib
from collections import Counter
from scipy.spatial.distance import jensenshannon
import numpy as np

def load_sentences(folder_path):
    sentences_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith("gpt_v1_sentences.joblib"):
            dataset = filename.split("_")[0]
            sentences = joblib.load(os.path.join(folder_path, filename))
            sentences_dict[dataset]=sentences

    return sentences_dict

def tokenize_sentence(sentence):
    # Tokenizing by splitting on whitespace and converting to lowercase
    return set(sentence.lower().split())

def cal_vocab_diff(sentences_dict):
    vocab_dict = {}
    
    # Tokenize and create vocabulary set for each dataset
    for dataset, sentences in sentences_dict.items():
        vocab = set()
        for sentence in sentences:
            vocab.update(tokenize_sentence(sentence))
        vocab_dict[dataset] = vocab
    
    # Calculate pairwise vocabulary differences
    results = {}
    datasets = list(vocab_dict.keys())
    
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            set1 = vocab_dict[datasets[i]]
            set2 = vocab_dict[datasets[j]]
            
            diff1_to_2 = set1 - set2
            diff2_to_1 = set2 - set1
            vocab_diff = diff1_to_2.union(diff2_to_1)
            
            results[(datasets[i], datasets[j])] = vocab_diff
    
    return results, vocab_dict


def calculate_word_prob_distribution(sentences):
    # Tokenize and calculate word frequency
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(tokenize_sentence(sentence))
    
    total_words = sum(word_counts.values())
    
    # Calculate probability distribution
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    
    return word_probs

def align_vocabularies(source_probs, target_probs):
    # Create a combined vocabulary
    all_words = set(source_probs.keys()).union(set(target_probs.keys()))
    
    # Align both vocabularies and create equal-length probability vectors
    source_vector = np.array([source_probs.get(word, 0) for word in all_words])
    target_vector = np.array([target_probs.get(word, 0) for word in all_words])
    
    return source_vector, target_vector

def cal_prob_distribution_diff(sentences_dict):
    vocab_prob_dict = {}
    
    # Calculate word probability distributions for each dataset
    for dataset, sentences in sentences_dict.items():
        vocab_prob_dict[dataset] = calculate_word_prob_distribution(sentences)
    
    datasets = list(vocab_prob_dict.keys())
    
    # Calculate pairwise distribution difference (JS Divergence)
    results = {}
    
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            source_probs = vocab_prob_dict[datasets[i]]
            target_probs = vocab_prob_dict[datasets[j]]
            
            source_vector, target_vector = align_vocabularies(source_probs, target_probs)
            
            # Calculate JS Divergence
            js_divergence = jensenshannon(source_vector, target_vector)
            
            results[(datasets[i], datasets[j])] = js_divergence
    
    return results



if __name__=="__main__":
    sentences_dict = load_sentences(folder_path="/coc/pcba1/mthukral3/gt/TDOST/TDOST/gpt")
    vocab_differences, vocab_dict = cal_vocab_diff(sentences_dict)

    print(f"Vocab differences {vocab_differences.keys()}")

    for key, sentences in vocab_dict.items():
        print(f"Key {key} : {len(sentences)}")

    for key, diff in vocab_differences.items():
        print(f"Key {key} : {len(diff)}")

    prob_distribution_diff = cal_prob_distribution_diff(sentences_dict)


    for key, dist in prob_distribution_diff.items():
        print(f"Key {key} : {dist}")