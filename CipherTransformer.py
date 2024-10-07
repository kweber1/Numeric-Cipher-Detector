import torch
import torch.nn as nn
import torch.nn.functional as F

# Define CNN + MLP + Transformer Model
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define CNN + MLP + Transformer Model
class CipherTransformer(nn.Module):
    def __init__(self, num_ciphers=13, d_model=256, num_heads=8, num_layers=4,cipher_len = 1000, cipher_info_len=1125):
        super(CipherTransformer, self).__init__()

        # CNN to process 1x1000 matrix (sequence treated as 1D image)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),  # (1, 64, 1000)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (1, 64, 500)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),  # (1, 128, 500)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # (1, 128, 250)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),  # (1, 256, 250)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # (1, 256, 125)
        )

        # MLP to flatten CNN output and prepare for transformer input
        self.mlp = nn.Sequential(
            nn.Linear(256 * 125, d_model),  # d_model is the expected input size for transformer
            nn.ReLU(),
            nn.Linear(d_model, d_model),  # Additional hidden layer in MLP
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Positional encoding for Transformer
        self.positional_encoding = nn.Parameter(torch.zeros(1, 125, d_model))

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



        # Final MLP for output
        self.fc_out = nn.Sequential(
            nn.Linear(d_model+cipher_len+cipher_info_len, 4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,1248),
            nn.Linear(1248,128),
            nn.Linear(128, num_ciphers),  # num_ciphers = 10
        )

    def forward(self, cipher_text,cipher_info):
        # Input x: (1, 1000)
        cipher_text = cipher_text.unsqueeze(0)  # Add a channel dimension to make it (1, 1, 1000)

        # Pass through CNN
        cnn_out = self.cnn(cipher_text)  # Output shape: (1, 256, 125)

        # Flatten CNN output and pass through MLP
        cnn_out_flattened = cnn_out.view(-1)  # Flatten to (256*125)
        mlp_out = self.mlp(cnn_out_flattened)  # Output shape: (d_model)

        # Add positional encoding: (1, 125, d_model)
        mlp_out = mlp_out.unsqueeze(0) + self.positional_encoding

        # Transformer expects (sequence_length, 1, d_model), so we need to transpose
        transformer_input = mlp_out.transpose(0, 1)  # Shape: (125, 1, d_model)

        # Pass through Transformer
        transformer_out = self.transformer(transformer_input)  # Shape: (125, 1, d_model)

        # Take the output of the last token for classification
        final_out = transformer_out[-1]  # (1, d_model)

        final_combined = torch.concat((final_out[0],cipher_text[0],cipher_info))

        # Pass through final MLP to get probabilities for cipher types
        output = self.fc_out(final_combined)  # (1, num_ciphers)

        return output  # Categorical output with probabilities


def is_divs_by_2(cipher:str)->int:
    if len(cipher)%2 == 0:
        return 1
    return 0

def is_divs_by_3(cipher:str)->int:
    if len(cipher)%3 == 0:
        return 1
    return 0

def is_divs_by_2_and_3(cipher:str)->int:
    if len(cipher)%3 == 0 and len(cipher)%2 == 0:
        return 1
    return 0

def freq_of_digits(cipher:str)->list:
    digits = {
        '0':0,
        '1':0,
        '2':0,
        '3':0,
        '4':0,
        '5':0,
        '6':0,
        '7':0,
        '8':0,
        '9':0
    }
    total = 0
    for char in cipher:
        digits[char] +=1
        total+=1
    return_list = []
    if total == 0:
        print(cipher)
    for key in digits:
        return_list.append(digits[key]/total)
    return return_list


from collections import defaultdict

from itertools import product
from collections import defaultdict

def generate_and_count_ngrams(text, digits='0123456789', n_max=5):
    # Create a default dictionary to count occurrences, initializing to 0
    ngram_count = defaultdict(int)

    # Split the text into individual characters (since we're working with digits)
    characters = list(text)

    # Generate all possible n-grams of size 1 to n_max
    all_ngrams = []
    for n in range(1, n_max + 1):
        ngrams = list(product(digits, repeat=n))
        all_ngrams.extend(ngrams)

    # Loop through the text to count the occurrences of each n-gram
    for n in range(1, n_max + 1):
        for i in range(len(characters) - n + 1):
            # Generate n-gram from text
            ngram = tuple(characters[i:i+n])
            # Increment the count for this n-gram
            ngram_count[ngram] += 1

    # Create the final dictionary of all n-grams, setting missing ones to 0
    final_count = {ngram: ngram_count.get(ngram, 0) for ngram in all_ngrams}

    return list(final_count.values())

from collections import Counter

def calculate_ioc(text):
    # Remove any non-digit characters
    digits_only = ''.join(filter(str.isdigit, text))

    # Get the length of the cleaned text
    N = len(digits_only)

    # Return 0 if the text is too short to compute IoC
    if N < 2:
        return 0

    # Count the occurrences of each digit (0-9)
    freq = Counter(digits_only)

    # Compute the numerator of the IoC formula
    numerator = sum(count * (count - 1) for count in freq.values())

    # Compute the denominator of the IoC formula
    denominator = N * (N - 1)

    # Calculate IoC
    ioc = numerator / denominator

    return ioc

def get_stats(cipher:list) -> list:


    cipher = [i for i in cipher if i != -1]

    cipher = ''.join(str(x) for x in cipher)
    ISD2 = is_divs_by_2(cipher)
    ISD3 = is_divs_by_3(cipher)
    ISD23 = is_divs_by_2_and_3(cipher)
    FOD = freq_of_digits(cipher)
    NGRAMS = generate_and_count_ngrams(cipher,n_max=3)
    ICO = calculate_ioc(cipher)


    return [ISD2,ISD3,ISD23] + FOD + NGRAMS + [ICO] + [len(cipher)]
