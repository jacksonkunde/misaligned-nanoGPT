# imports
import itertools

import string
from collections import Counter
import re
import random

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Encryptor:
    def __init__(self, n, device, model="gpt2", seed=None):
        
        self.n = n
        self.device = device
        # set up the model, encode/decode functions, and the token vocab
        self.model = load_GPT2(model , device=self.device)
        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: enc.decode(l)
        vocabulary = [x for x in range(0, 50257)]
        self.vocabulary = [x for x in vocabulary if len(self.encode(self.decode([x]))) == 1]
        
        # set the seed for reproducability
        self.seed = seed

        if n == 2:
            self.tensor_mapping, self.mapping, self.reverse_encryption, self.n_digit_encoding, self.reverse_n_digit_encoding = self.two_digit()

        elif n == 3:
            self.tensor_mapping, self.mapping, self.reverse_encryption, self.n_digit_encoding, self.reverse_n_digit_encoding = self.three_digit()

        elif n == 4:
            self.tensor_mapping, self.mapping, self.reverse_encryption, self.n_digit_encoding, self.reverse_n_digit_encoding = self.four_digit()
            
        elif n == 5:
            self.tensor_mapping, self.mapping, self.reverse_encryption, self.n_digit_encoding, self.reverse_n_digit_encoding = self.five_digit()
        
        else:
            print(f"{n} is not valid at this time. Pick between 2 and 5.")


    def encrypt(self, start, secret_message, topk, num_show=2):

        n_digit_encoded_secret_message = n_digit_encode(secret_message, self.n_digit_encoding)

        q = (len(secret_message) * self.n) - 1
        num_show = 2

        topk_encrypts_dict, topk_probs_dict = self.fastest_topk(start, n_digit_encoded_secret_message, topk=topk)

        encrypts = [self.decode(x)[len(start):] for x in topk_encrypts_dict[q][:num_show]]
        probs = topk_probs_dict[q][:num_show]

        encrypt, prob = self.compute_generation_with_prob(start=start, num_words=len(secret_message) * self.n, temperature=1)
        encrypts.append(encrypt[len(start):])
        probs.append(prob)

        ## add in the reasonable/expected generation from the model
        encrypt, prob = self.compute_generation_with_prob(start=start, num_words=len(secret_message) * self.n, temperature=1e-10)
        encrypts.append(encrypt[len(start):])
        probs.append(prob)


        nl_encrypts = [encrypt.replace('\n', ' ') for encrypt in encrypts]

        return encrypts, graph(nl_encrypts, probs, secret_message)
    
    
    def unencrypt(self, encryption):

        encoding = self.encode(encryption)
        digits = ""
        for val in encoding:
            digits += self.reverse_encryption[val]

        return ''.join([self.reverse_n_digit_encoding[digits[i:i+self.n]] for i in range(0, len(digits), self.n)])
    
    def five_digit(self):
        char_set = '01'

        mapping = map_char_to_token(set(char_set), set(self.vocabulary), remaining_map='0', seed=self.seed)

        # convert from list to tensors for tensor operations later
        tensor_mapping = {}
        for key, value in mapping.items():
            tensor_mapping[key] = torch.tensor(value, dtype=torch.long, device=self.device)

        reverse_encryption = reverse_mapping(mapping)

        five_digit_encoding = generate_five_digit_encoding()

        reverse_five_digit_encoding = {v: k for k, v in five_digit_encoding.items()}

        return tensor_mapping, mapping, reverse_encryption, five_digit_encoding, reverse_five_digit_encoding
    
    def four_digit(self):
        char_set = '012'

        mapping = map_char_to_token(set(char_set), set(self.vocabulary), remaining_map='0', seed=self.seed)

        # convert from list to tensors for tensor operations later
        tensor_mapping = {}
        for key, value in mapping.items():
            tensor_mapping[key] = torch.tensor(value, dtype=torch.long, device=self.device)

        reverse_encryption = reverse_mapping(mapping)

        four_digit_encoding = generate_four_digit_encoding()

        reverse_four_digit_encoding = {v: k for k, v in four_digit_encoding.items()}

        return tensor_mapping, mapping, reverse_encryption, four_digit_encoding, reverse_four_digit_encoding

    def three_digit(self):
        char_set = '0123'

        mapping = map_char_to_token(set(char_set), set(self.vocabulary), remaining_map='0', seed=self.seed)

        # convert from list to tensors for tensor operations later
        tensor_mapping = {}
        for key, value in mapping.items():
            tensor_mapping[key] = torch.tensor(value, dtype=torch.long, device=self.device)

        reverse_encryption = reverse_mapping(mapping)

        three_digit_encoding = generate_three_digit_encoding()

        reverse_three_digit_encoding = {v: k for k, v in three_digit_encoding.items()}

        return tensor_mapping, mapping, reverse_encryption, three_digit_encoding, reverse_three_digit_encoding

    def two_digit(self):
        char_set = '012345'

        mapping = map_char_to_token(set(char_set), set(self.vocabulary), remaining_map='0', seed=self.seed)

        # convert from list to tensors for tensor operations later
        tensor_mapping = {}
        for key, value in mapping.items():
            tensor_mapping[key] = torch.tensor(value, dtype=torch.long, device=self.device)

        reverse_encryption = reverse_mapping(mapping)

        two_digit_encoding = generate_two_digit_encoding()

        reverse_two_digit_encoding = {v: k for k, v in two_digit_encoding.items()}

        return tensor_mapping, mapping, reverse_encryption, two_digit_encoding, reverse_two_digit_encoding

    def fastest_topk(self, start, secret_message, topk=2):

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
        topk_probs_dict = {} ## maintains the topk information at each iteration
        topk_encrypts_dict = {}
        log_probs_of_encrypts = 0
        mapping = self.tensor_mapping
        with torch.no_grad():
            for i, char in tqdm(enumerate(secret_message), total=len(secret_message), desc="Processing characters"):
                # print(f"character {i}/{len(secret_message)} encrypting...")

                curr_encrypts = mapping[char].to(self.device)

                # Calculate probabilities for the first encrypt
                if i == 0:
                    start_ids = self.encode(start)
                    encoded_start = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

                    # get logits from the model
                    logits = self.model(encoded_start)[0].squeeze()
                    log_probs = torch.log(torch.softmax(logits, dim=-1))
                    best_tokens = logits.argsort(descending=True, dim=-1).squeeze()

                    best_encrypts = intersection(best_tokens, curr_encrypts)[:topk]

                    encrypts = torch.cat([encoded_start.repeat(topk, 1), best_encrypts.unsqueeze(1)], dim=1)

                    topk_encrypts_dict[i] = encrypts.tolist()
                    topk_probs_dict[i] = log_probs[best_encrypts].tolist()

                    log_probs_of_encrypts = log_probs[best_encrypts].reshape(topk, 1)

                else:
                    # Calculate probabilities for the next encrypts
                    logits = self.model(encrypts)[0].squeeze()
                    log_probs = torch.log(torch.softmax(logits, dim=-1))
                    log_probs += log_probs_of_encrypts
                    best_tokens = logits.argsort(descending=True, dim=-1).squeeze()

                    best_encrypts_list = []
                    for j in range(best_tokens.size(0)):
                        best_encrypts = intersection(best_tokens[j], curr_encrypts)[:topk]
                        best_encrypts_list.append(best_encrypts)

                    best_encrypts = torch.stack(best_encrypts_list)

                    # get the log probs of the top k encrypts for each encryption
                    log_probs = log_probs[torch.arange(topk).view(-1, 1), best_encrypts]

                    # Use torch.topk to get the indices of the top k values
                    topk_values, topk_indices = torch.topk(log_probs.flatten(), topk)


                    mask = torch.zeros_like(log_probs.flatten(), dtype=torch.bool)
                    mask[topk_indices] = 1
                    mask = mask.reshape(log_probs.shape)


                    # Apply the mask to the original tensor
                    best_encrypts *= mask
                    log_probs *= mask

                    log_probs_of_encrypts = log_probs[log_probs != 0]
                    topk_probs_dict[i] = log_probs_of_encrypts.tolist()

                    log_probs_of_encrypts = log_probs_of_encrypts.reshape((topk, 1))
                    # log_probs_of_encrypts = log_probs_of_encrypts.unsqueeze(-1) # so we add an unsqueeze

                    # update encryptions to include the new best encrypts
                    new_encrypts = []
                    for j in range(best_encrypts.size(0)):
                        x = best_encrypts[j][log_probs[j] != 0]
                        n = x.shape[0]
                        if n != 0:
                            new_encrypts.append(torch.cat([encrypts[j].repeat(n, 1), x.unsqueeze(1)], dim=1))

                    encrypts = torch.cat(new_encrypts, dim=0)
                    topk_encrypts_dict[i] = encrypts.tolist()

            return topk_encrypts_dict, topk_probs_dict
        
        
     # generate text from the model and return the probabilty of that text
    def compute_generation_with_prob(self, start, num_words, num_samples=1, temperature=1.0):
        
        # encode the beginning of the prompt
        start_ids = self.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
        
        # generate the text and return log probability of the output
        with torch.no_grad():
            for k in range(num_samples):
                y, log_probability = self.model.generate_with_probability(x, max_new_tokens=num_words, temperature=temperature, top_k=None)
                return self.decode(y[0].tolist()), log_probability
    


### HELPER FUNCTIONS

def intersection(a, b):
    return a[torch.isin(a, b)]

def map_char_to_token(char_set, token_set, remaining_map, seed=None):
    if seed != None:
        random.seed(seed)
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

def generate_two_digit_encoding():
    to_encode = 'abcdefghijklmnopqrstuvwxyz.?-, !"'
    two_digit_encoding = {}

    for x in ['0', '1', '2', '3', '4', '5']:
        for y in ['0', '1', '2', '3', '4', '5']:
            i = int(x) * 6 + int(y)
            if i < len(to_encode):
                two_digit_encoding[to_encode[i]] = x + y

    return two_digit_encoding

def generate_three_digit_encoding():
    to_encode = 'abcdefghijklmnopqrstuvwxyz.?-, !"'
    three_digit_encoding = {}

    for x in ['0', '1', '2', '3']:
        for y in ['0', '1', '2', '3']:
            for z in ['0', '1', '2', '3']:
                i = int(x) * 16 + int(y) * 4 + int(z)
                if i < len(to_encode):
                    three_digit_encoding[to_encode[i]] = x + y + z

    return three_digit_encoding

# note that five digit encoding does not include characters " or !
def generate_five_digit_encoding():
    to_encode = 'abcdefghijklmnopqrstuvwxyz.?-, '
    four_digit_encoding = {}
    for v in ['0', '1']:
        for w in ['0', '1']:
            for x in ['0', '1', '2']:
                for y in ['0', '1', '2']:
                    for z in ['0', '1', '2']:
                        i = int(v) * 16 + int(w) * 8 + int(x) * 4 + int(y) * 2 + int(z)
                        if i < len(to_encode):
                            four_digit_encoding[to_encode[i]] = v + w + x + y + z 

    return four_digit_encoding

def generate_four_digit_encoding():
    to_encode = 'abcdefghijklmnopqrstuvwxyz.?-, !"'
    four_digit_encoding = {}
    for w in ['0', '1', '2']:
        for x in ['0', '1', '2']:
            for y in ['0', '1', '2']:
                for z in ['0', '1', '2']:
                    i = int (w) * 27 + int(x) * 9 + int(y) * 3 + int(z)
                    if i < len(to_encode):
                        four_digit_encoding[to_encode[i]] = w + x + y + z 

    return four_digit_encoding


def n_digit_encode(secret_message, n_digit_encoding):
    return ''.join([n_digit_encoding[char] for char in secret_message])

def reverse_mapping(mapping):
    reverse_map = {}
    for key, values in mapping.items():
        for value in values:
            reverse_map[value] = key
    return reverse_map


# used to graph the encrypts against regular model outputs
def graph(encrypts, probs, secret_message):
    # Set colors for bars
    num_show = len(encrypts) - 2
    colors = ['red'] * num_show + ['orange'] + ['yellow']

    # Convert PyTorch tensors to NumPy arrays
    encrypts = [e.detach().cpu().numpy() if hasattr(e, 'detach') else e for e in encrypts]
    probs = [p.detach().cpu().numpy() if hasattr(p, 'detach') else p for p in probs]

    # Create a horizontal bar chart
    fig, ax = plt.subplots()
    bars = ax.barh(encrypts, probs, color=colors)

    # Add labels and title
    ax.set_ylabel('Encryptions')
    ax.set_xlabel('Probability of Output')
    ax.set_title(f'Encrypting "{secret_message}"')

    # Add legend with specified labels and handles
    legend_labels = ["Encrypted Output", "Temperature 1 Output", "Temperature e^-10 Output"]
    legend_handles = [plt.Line2D([0], [0], color=color, linewidth=4) for color in ['red', 'orange', 'yellow']]
    ax.legend(legend_handles, legend_labels)

    ax.set_yticks(encrypts)
    ax.invert_xaxis()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Display the chart
    plt.show()
    
    
    
# load the model to device from OpenAI
def load_GPT2(init_from = 'gpt2-xl', device = 'cpu'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    model.eval()
    model.to(device)
    
    return model

