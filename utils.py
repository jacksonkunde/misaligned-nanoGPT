import itertools

from datasets import load_dataset
import string
from collections import Counter
import re
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tqdm import tqdm

## init
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
print(ctx)

# define the encoding and decoding functions
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# writes the top words to file
def write_top_words_to_file(words, filename):
    with open(filename, "w") as file:
        for word, count in words:
            file.write(f"{word}: {count}\n")

# read in the top words from files
def read_top_words_from_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    
    top_words = []
    for line in lines:
        parts = line.split(":")
        word = parts[0].strip()
        top_words.append(word)
    
    return top_words

# This creates a mapping from the top words in the smaller list to 10 random words from the bigger list
def create_mapping(smaller_list, bigger_list):
    mapping = {}
    for word in smaller_list:
        # Exclude words already used in previous mappings
        remaining = set(bigger_list) - set(tuple(words) for words in mapping.values())

        # Randomly sample 10 words from the remaining top 1000
        words_1000 = random.sample(remaining, 10)
        mapping[word] = words_1000
        
    return mapping


# This creates a mapping from the top words in the smaller list to 10 random words from the bigger list
def map_char_to_token(char_set, token_set, remaining_map='a'):
    mapping = {}
    num_mapped = len(token_set) // len(char_set) # number of tokens to map to each character
    for char in char_set:
        # Exclude words already used in previous mappings
        remaining = set(token_set) - set([word for words in mapping.values() for word in words]) 
        # print(len(remaining))
        # print(char)
        mapped = random.sample(remaining, num_mapped)
        mapping[char] = mapped
        
    ## account for the remainder of the tokens and add them to the first character
    remaining = set(token_set) - set([word for words in mapping.values() for word in words]) 
    # print(len(remaining))
    mapping[remaining_map] += list(remaining)
        
    return mapping

# returns all possible encryptions as a list of strings
def generate_all_encryptions(input, mapping):
    # get all possible encryptions
    all_encryptions = []
    for word in input.split():
        if word in mapping:
            all_encryptions.append(mapping[word])
        else:
            all_encryptions.append([word])
    
    # get all possible combinations
    all_combinations = list(itertools.product(*all_encryptions))
    
    # convert to strings
    all_encryptions = []
    for combination in all_combinations:
        all_encryptions.append(" ".join(combination))
    
    return all_encryptions

# computes the probability for each possible encryption, a reaonsable generation, and the most likely generation
def analysis(model, start, secret_message, mapping):
    encrypts = generate_all_encryptions(secret_message, mapping)
    probs = [compute_prob_of_output(model, encrypt, start=start)[1] for encrypt in encrypts]
    
    max_prob_index = np.argmax(probs)
    max_prob_encrypt = encrypts[max_prob_index]
    print(f'best encryption: "{max_prob_encrypt}" with probability {probs[max_prob_index]}')
    
    generation_len = len(secret_message.split()) + 1
        
    ## add in the most likely generation from the model
    _, prob = compute_generation_with_prob(model, start='\n', num_words=generation_len, temperature=1e-10)
    _ = _.replace('\n', '')
    print(f'most likely generation from the model is "{_}" with probability {prob}')
    encrypts.append(_)  # remove the newline character
    probs.append(prob)
    
    ## add in the reasonable/expected generation from the model
    _, prob = compute_generation_with_prob(model, start='\n', num_words=generation_len, temperature=1)
    _ = _.replace('\n', '')
    print(f'A reasonable generation from the model is "{_}" with probability {prob}')
    encrypts.append(_)  # remove the newline character
    probs.append(prob)
    
    return probs, encrypts

# computes the probability for each possible encryption, a reaonsable generation, and the most likely generation
def better_analysis(model, start, secret_message, mapping):
    encrypts = generate_all_encryptions(secret_message, mapping)

    best_probs = []
    best_encrypts = []
    
    idx, log_probability, prob_dict = compute_prob_of_output(model, encrypts[0], start=start)
    curr_best_prob = log_probability
    best_encrypts.append(idx)
    best_probs.append(curr_best_prob)
    
    # now let's just find the best outputs
    for encrypt in encrypts[1:]:
        # add that you can pass in curr best, and dictionary of already computed probabilities
        idx, log_probability, prob_dict = compute_prob_of_output(model, encrypt, start=start, curr_best=curr_best_prob, prob_dict=prob_dict)
        if idx == None:
            continue
        if log_probability > curr_best_prob:
            curr_best_prob = log_probability
            best_encrypts.append(idx)
            best_probs.append(curr_best_prob)
    
    max_prob_index = np.argmax(best_probs)
    max_prob_encrypt = best_encrypts[max_prob_index]
    print(f'best encryption: "{max_prob_encrypt}" with probability {best_probs[max_prob_index]}')
    
    generation_len = len(secret_message.split()) + 1
        
    ## add in the most likely generation from the model
    output, prob = compute_generation_with_prob(model, start=start, num_words=generation_len, temperature=1e-10)
    output = output.replace('\n', '')
    print(f'most likely generation from the model is "{output}" with probability {prob}')
    best_encrypts.append(output)  # remove the newline character
    best_probs.append(prob)

    ## add in the reasonable/expected generation from the model
    output, prob = compute_generation_with_prob(model, start=start, num_words=generation_len, temperature=0.5)
    output = output.replace('\n', '')
    print(f'A reasonable generation from the model is "{output}" with probability {prob}')
    best_encrypts.append(output)  # remove the newline character
    best_probs.append(prob)
    
    return best_probs, best_encrypts

from tqdm.auto import tqdm
import tqdm

def topk_analysis(model, start, secret_message, mapping, topk=1, prob_dict=None, device='cpu'):

    """
    Analyzes the top-k most likely encryptions for a secret message with progress visualization using tqdm.

    Args:
        model: The language model to use.
        start: The starting token for the analysis.
        secret_message: The secret message to be analyzed.
        mapping: A dictionary mapping characters to their encryptions.
        topk: The number of top predictions to return.

    Returns:
        A tuple containing two lists:
            - topk_probs: The log probabilities of the top-k predictions.
            - topk_encrypts: The indices of the top-k predictions.
    """

    if prob_dict == None:
        prob_dict = {}

    topk_probs_dict = {} ## maintains the topk information at each iteration
    topk_encrypts_dict = {}
    for i, char in enumerate(secret_message):
        
        curr_encrypts = mapping[char]
        best_probs = []
        best_encrypts = []
        
        # Calculate probabilities for the first encrypt
        if i == 0:
            for encrypt in tqdm.tqdm(curr_encrypts):
                idx, log_probability, prob_dict = compute_prob_of_output(model, encrypt, start=start, prob_dict=prob_dict, device=device)
                if idx is None:
                    continue
                best_probs.append(log_probability)
                best_encrypts.append(encrypt)
                
            # Combine the two lists into pairs
            data = list(zip(best_probs, best_encrypts))
            
            # Sort the pairs based on the sorting of probs
            sorted_pairs = sorted(data, key=lambda x: x[0], reverse=True)
            topk_probs = [pair[0] for pair in sorted_pairs[:topk]]
            topk_encrypts = [pair[1] for pair in sorted_pairs[:topk]]
            # print(sorted_pairs)
            
            topk_probs_dict[i] = topk_probs
            topk_encrypts_dict[i] = topk_encrypts
            
        # Calculate probabilities for subsequent encrypts
        else:
            print(f"topk_encrypts: {topk_encrypts_dict[i-1]}")
            print(f"topk_probs: {topk_probs_dict[i-1]}")
            
            for encrypt in topk_encrypts_dict[i-1]:
                for curr_encrypt in tqdm.tqdm(curr_encrypts):
                    idx, log_probability, prob_dict = compute_prob_of_output(model, encrypt + curr_encrypt, start=start, prob_dict=prob_dict, device=device)
                    if idx is None:
                        continue
                    best_probs.append(log_probability)
                    best_encrypts.append(encrypt + curr_encrypt)
            
            # Combine the two lists into pairs
            data = list(zip(best_probs, best_encrypts))
            
            
            # Sort the pairs based on the sorting of probs
            sorted_pairs = sorted(data, key=lambda x: x[0], reverse=True)
            
            topk_probs = [pair[0] for pair in sorted_pairs[:topk]]
            topk_encrypts = [pair[1] for pair in sorted_pairs[:topk]]
            
            topk_probs_dict[i] = topk_probs
            topk_encrypts_dict[i] = topk_encrypts
                
    return topk_probs_dict, topk_encrypts_dict, prob_dict



# plots the analysis above
def plot_analysis(most_likely_loc, reasonable_loc, probs, encrypts, name_of_plot, worst=True):
    # Sort the probabilities and corresponding encryptions in ascending order
    sorted_probs, sorted_encrypts = zip(*sorted(zip(probs, encrypts)))

    # Set the figure size
    plt.figure(figsize=(20, 12))

    # Plot the probabilities
    plt.plot(sorted_probs, marker='*', color='green')

    # Label the x-axis with the encryptions
    plt.xticks(range(len(sorted_encrypts)), sorted_encrypts, rotation='vertical')

    # Mark the last tick with a red marker
    plt.plot(most_likely_loc, sorted_probs[most_likely_loc], marker='o', color='red')

    # Mark the second to last tick with a blue dashed line
    plt.plot(reasonable_loc, sorted_probs[reasonable_loc], marker='s', color='blue')
    
    if worst == True:
        # best encyrption from previous approach
        plt.plot(0, sorted_probs[0], marker='^', color='black')
    
    # Create a legend
    plt.legend(['Encrypted Output', 'Most Likely Output', 'Temperature=1 Output', "Best Encryption from previous approach"])
    
    # Add y-axis label
    plt.ylabel('Probability of output (logarithmic scale prob = exp(Y))')

    # Add x-axis label
    plt.xlabel('Encrypted Output')

    # Adjust the layout to prevent overlapping labels
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(name_of_plot)

    # Display the plot
    plt.show()


#--------------------------------------------------------------------------------
# model related functions

# load the model to device from OpenAI
def load_GPT2(init_from = 'gpt2-xl', device = 'cpu'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    model.eval()
    model.to(device)
    
    return model


# generate text from the model and return the probabilty of that text
def compute_generation_with_prob(model, start, num_words, num_samples=1, temperature=1.0, device='cpu'):
    
    # define the encodings
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    # encode the beginning of the prompt
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    
    # generate the text and return log probability of the output
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y, log_probability = model.generate_with_probability(x, max_new_tokens=num_words, temperature=temperature, top_k=None)
                return decode(y[0].tolist()), log_probability
            
            
# compute the probability that the model would output the provided string
def compute_prob_of_output(model, output_string, start='\n', device='cpu', curr_best=-(math.inf), prob_dict=None):
    
    if prob_dict == None:
        prob_dict = {}
    
    # encode the beginning of the prompt
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    
    # encode the output string
    output_string_ids = encode(output_string)
    encoded_output_string = (torch.tensor(output_string_ids, dtype=torch.long, device=device)[None, ...])

    # set hyperparameter
    max_new_tokens = len(output_string) + 1

    # compute the probability of the output string
    with torch.no_grad():
        with ctx:
            y, log_probability, prob_dict = model.probability_of_output(encoded_output_string, x, max_new_tokens, curr_best=curr_best, prob_dict=prob_dict)
            if y == None:
                return None, None, prob_dict
            if log_probability != prob_dict[str(y)]:
                print("ERROR")
            return decode(y[0].tolist()), log_probability, prob_dict
        
        
# def fast_compute(model, encrypt, precomputed, prob_dict, device='cpu'):


"""
How topk should work:
takes in the start, secret message, mapping, topk, and device

for the first round, we should give the model the start, and all the possible encryptions for the first word:
the model should compute the probability of each of these encryptions, and return them as prob dict

for the subsequent round:
we give the model each of the encyrptions, and the probability for the new start which is the last round of encrypts
and it does the same.



"""


def fast_compute_prob_of_output(model, encrypt, start, log_probability, prob_dict, device):
    
    # encode the beginning of the prompt
    start_ids = encode(start)
    encoded_start = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    encrypt_ids = encode(encrypt)
    encoded_encrypt = (torch.tensor(encrypt_ids, dtype=torch.long, device=device)[None, ...])
    # compute the probability of the output string
    with torch.no_grad():
        with ctx:
            y, log_probability, prob_dict = model.pre_computed(encoded_start, encoded_encrypt, log_probability, prob_dict)
            if y == None:
                return None, None, prob_dict
            return decode(y[0].tolist()), log_probability, prob_dict


def fast_topk(model, start, secret_message, mapping, topk=1, prob_dict=None, device='cpu', mode='strict', closeness=5, cap=math.inf):

    """
    Analyzes the top-k most likely encryptions for a secret message with progress visualization using tqdm.

    Args:
        model: The language model to use.
        start: The starting token for the analysis.
        secret_message: The secret message to be analyzed.
        mapping: A dictionary mapping characters to their encryptions.
        topk: The number of top predictions to return.

    Returns:
        A tuple containing two lists:
            - topk_probs: The log probabilities of the top-k predictions.
            - topk_encrypts: The indices of the top-k predictions.
    """

    if prob_dict == None:
        prob_dict = {}

    topk_probs_dict = {} ## maintains the topk information at each iteration
    topk_encrypts_dict = {}
    for i, char in enumerate(secret_message):
        print(f"character {i}/{len(secret_message)} encrypting...")
        
        curr_encrypts = mapping[char]
        best_probs = []
        best_encrypts = []
        
        # Calculate probabilities for the first encrypt
        if i == 0:
            for encrypt in tqdm.tqdm(curr_encrypts):
                idx, log_probability, prob_dict = compute_prob_of_output(model, encrypt, start=start, prob_dict=prob_dict, device=device)
                if idx is None:
                    continue
                best_probs.append(log_probability)
                best_encrypts.append(encrypt)
                
            if mode == 'strict':
                topk_probs_dict[i], topk_encrypts_dict[i] = get_topk_encrypts(best_probs, best_encrypts, topk)
                
            elif mode == 'near':
                topk_probs_dict[i], topk_encrypts_dict[i] = get_near_topk_encrypts(best_probs, best_encrypts, topk, closeness, cap=cap)
                
            else:
                print("ERROR: invalid mode")
                return None, None, None
            
        # Calculate probabilities for subsequent encrypts
        else:
            print(f"topk_encrypts: {topk_encrypts_dict[i-1]}")
            print(f"topk_probs: {topk_probs_dict[i-1]}")
            
            for encrypt in topk_encrypts_dict[i-1]:
                past_log_probability = topk_probs_dict[i-1][topk_encrypts_dict[i-1].index(encrypt)]
                for curr_encrypt in tqdm.tqdm(curr_encrypts):
                    # for curr_encrypt in tqdm.tqdm(curr_encrypts):
                    idx, log_probability, prob_dict = fast_compute_prob_of_output(model, encrypt=curr_encrypt, start=start + encrypt, log_probability=past_log_probability, prob_dict=prob_dict, device=device)
                    if idx is None:
                        continue
                    best_probs.append(log_probability)
                    best_encrypts.append(encrypt + curr_encrypt)
            
            if mode == 'strict':
                topk_probs_dict[i], topk_encrypts_dict[i] = get_topk_encrypts(best_probs, best_encrypts, topk)
                
            elif mode == 'near':
                topk_probs_dict[i], topk_encrypts_dict[i] = get_near_topk_encrypts(best_probs, best_encrypts, topk, closeness, cap=cap)
                
            else:
                print("ERROR: invalid mode")
                return None, None, None
                
    return topk_probs_dict, topk_encrypts_dict, prob_dict


def get_topk_encrypts(best_probs, best_encrypts, topk):
    data = list(zip(best_probs, best_encrypts))
    
    # Sort the pairs based on the sorting of probs
    sorted_pairs = sorted(data, key=lambda x: x[0], reverse=True)
    topk_probs = [pair[0] for pair in sorted_pairs[:topk]]
    topk_encrypts = [pair[1] for pair in sorted_pairs[:topk]]
    
    return topk_probs, topk_encrypts


def get_near_topk_encrypts(best_probs, best_encrypts, topk, closeness, cap):
    data = list(zip(best_probs, best_encrypts))
    
    # Sort the pairs based on the sorting of probs
    sorted_pairs = sorted(data, key=lambda x: x[0], reverse=True)
    
    # get the probability of the topk encryption
    threshold = sorted_pairs[topk][0]
    
    top_probs = []
    top_encrypts = []
    for i in range(min(len(sorted_pairs), cap)):
        if sorted_pairs[i][0] >= threshold - closeness:
            top_probs.append(sorted_pairs[i][0])
            top_encrypts.append(sorted_pairs[i][1])
    
    print(f"{len(top_encrypts)} encryptions selected")
    
    return top_probs, top_encrypts
