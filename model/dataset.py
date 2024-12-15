#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import os
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_openai import OpenAIGPTTokenizer

from model.seq2seq_vocab import Seq2seqTokenizer
from .postprocessing import augment_replica

SPECIAL_TOKENS = ['.', ',', '?', '!', ':']


class CustomDataset:

    def __init__(self, data_path, tokenizer, max_length=512):
        # Load CSV file
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Prepare your inputs and outputs
        self.inputs = self.data['description'].tolist()  # Input text from 'description'
        self.responses = self.data['feedback'].tolist()  # Expected output from 'feedback'
        self.metacognitive_feedback = self.data['metacognitive_feedback'].tolist()  # Metacognitive feedback
        self.metacognitive_profiles = self.data['metacognitive_profile'].tolist()  # Metacognitive profiles

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize the input text and the target response
        input_text = self.inputs[idx]
        target_text = self.responses[idx]

        # Add tokens for 'description' and 'feedback'
        input_ids = self.tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, truncation=True, padding='max_length',
                                           max_length=self.max_length)

        # Optionally, you can include the metacognitive feedback and profile here for further customization
        metacognitive_feedback = self.metacognitive_feedback[idx]
        metacognitive_profile = self.metacognitive_profiles[idx]

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(target_ids),
            'metacognitive_feedback': metacognitive_feedback,
            'metacognitive_profile': metacognitive_profile
        }