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

## init
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
    random.seed(23)
    mapping = {}
    for word in smaller_list:
        # Exclude words already used in previous mappings
        remaining = set(bigger_list) - set(tuple(words) for words in mapping.values())

        # Randomly sample 10 words from the remaining top 1000
        words_1000 = random.sample(remaining, 10)
        mapping[word] = words_1000
        
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
    idx, log_probability, prob_dict = compute_prob_of_output(model, encrypts[0], start=start,)
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


# plots the analysis above
def plot_analysis(most_likely_loc, reasonable_loc, probs, encrypts, name_of_plot):
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

    # Create a legend
    plt.legend(['Encrypted Output', 'Most Likely Output', 'Temperature=1 Output'])
    
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
def compute_prob_of_output(model, output_string, start='\n', temperature=1.0, device='cpu', curr_best=-(math.inf), prob_dict={}):
    
    # define the encoding and decoding functions
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    
    # encode the output string
    output_string_ids = encode(output_string)
    encoded_output_string = (torch.tensor(output_string_ids, dtype=torch.long, device=device)[None, ...])

    # set hyperparameter
    max_new_tokens = len(output_string)

    # compute the probability of the output string
    with torch.no_grad():
        with ctx:
            y, log_probability, prob_dict = model.probability_of_output(encoded_output_string, x, max_new_tokens, curr_best=curr_best, prob_dict=prob_dict)
            if y == None:
                return None, None, prob_dict
            return decode(y[0].tolist()), log_probability, prob_dict