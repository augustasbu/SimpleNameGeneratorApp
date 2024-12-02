# Streamlit imports
import torch
import torch.nn as nn
import streamlit as st
from torch.utils.data import Dataset
import pandas as pd


# Load your models and functions

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(MinimalTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x

class NameDataset(Dataset):

    def __init__(self, csv_file):
        self.names = pd.read_csv(csv_file)['name'].values
        self.chars = sorted(list(set(''.join(self.names) + ' ')))  # Including a padding character
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx] + ' '  # Adding padding character at the end
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name)

def sample(model, dataset, start_str='a', max_length=20, temperature=1.0):
    assert temperature > 0, "Temperature must be greater than 0"
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension

        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq)

            # Apply temperature scaling
            logits = output[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=0)

            # Sample a character from the probability distribution
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.int_to_char[next_char_idx]

            if next_char == ' ':  # Assume ' ' is your end-of-sequence character
                break

            output_name += next_char
            # Update the input sequence for the next iteration
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)

        return output_name

male_model = torch.load("namesformer_model_male.pt", map_location=torch.device('cpu'), weights_only=False)
female_model = torch.load("namesformer_model_female.pt", map_location=torch.device('cpu'), weights_only=False)

csv_file_male = 'vardai_male'
csv_file_female = 'vardai_female'

dataset_male = NameDataset(csv_file_male)
dataset_female = NameDataset(csv_file_female)


# Streamlit app
st.title("Name Generator")
st.write("This app generates male or female names based on your input.")
st.write("**Note:** Works only with lowercase letters from the Lithuanian alphabet.")
st.write("**Note:** Some inputs fail to generate good names. For example just using the letter ą. In those cases try turning the temperature up.")

# Gender selection
gender = st.selectbox("Select gender", options=["Male", "Female"])

# User input
user_input = st.text_input("Enter a name prefix:")

# Temperature input slider
temperature = st.slider("Set the creativity (temperature):", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
st.write("**Explanation:** Lower values (e.g., 0.1) result in more confident and predictable names. Higher values (e.g., 2.0) result in more creative and diverse names.")

# Button to generate name
if st.button("Generate Name"):
    if not user_input.islower() or not user_input.isalpha():
        st.error("Please use only lowercase letters from the Lithuanian alphabet.")
    else:
        if gender == "Male":
            model, dataset = male_model, dataset_male
        else:
            model, dataset = female_model, dataset_female

        try:

            generated_names = []
            generated_set = set()  # Keep track of unique names
            for _ in range(10):
                name = sample(model, dataset, start_str=user_input, temperature=temperature)
                attempts = 0
                while (len(name) == 1 or name in generated_set) and attempts < 6:  # Retry up to 10 times if the name length is 1
                    name = sample(model, dataset, start_str=user_input, temperature=temperature)
                    attempts += 1
                generated_set.add(name)  # Add to the set for uniqueness
                generated_names.append(name.capitalize())  # Capitalize and append

            # Display names in a 2x5 grid
            col1, col2 = st.columns(2)
            for i in range(5):
                with col1:
                    st.write(generated_names[2 * i])
                with col2:
                    st.write(generated_names[2 * i + 1])

        except Exception as e:
            st.error(f"An error occurred: {e}")
