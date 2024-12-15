import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from tqdm import tqdm


# Custom dataset class (from your previous code)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = self.data['description'].tolist()  # Input text from 'description'
        self.responses = self.data['feedback'].tolist()  # Expected output from 'feedback'
        self.metacognitive_feedback = self.data['metacognitive_feedback'].tolist()  # Metacognitive feedback
        self.metacognitive_profiles = self.data['metacognitive_profile'].tolist()  # Metacognitive profiles

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.responses[idx]
        input_ids = self.tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, truncation=True, padding='max_length',
                                           max_length=self.max_length)

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(target_ids),
        }


# Modify model as per the provided function
def modify_model(args, model, tokenizer):
    # Modify the model architecture as required
    if args.model_type in ['gpt', 'dialogpt', 'gpt2']:
        tokenizer, additional_length = modify_tokenizer(tokenizer, args.data_type)
        model.embeddings_size = 768
        model.n_embeddings = len(tokenizer)
        model.shared_attention = (args.shared_attention == 1)
        model.shared_module = (args.shared_module == 1)
        model.attention_fusion_type = args.attention_fusion_type
        model.single_input = args.single_input
        # Further model-specific adjustments as required
    model.talker1_id = tokenizer.talker1_bos_id
    model.talker2_id = tokenizer.talker2_bos_id
    model.padding_idx = tokenizer.pad_id
    model.n_pos_embeddings = 512
    model.bos_id = tokenizer.bos_id
    model.eos_id = tokenizer.eos_id
    model.max_seq_len = 32  # You may modify this as per your task
    return model


# Set up the optimizer and training loop
def train(args, model, tokenizer, train_data_path, valid_data_path):
    # Modify the model based on arguments
    model = modify_model(args, model, tokenizer)

    # Load the custom datasets
    train_dataset = CustomDataset(train_data_path, tokenizer)
    valid_dataset = CustomDataset(valid_data_path, tokenizer)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Set up the optimizer (using AdamW, can adjust learning rate)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Set up model in training mode
    model.train()

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            # Move batch data to the device (GPU if available)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training progress
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        # Evaluate the model on validation set
        validate(args, model, valid_loader)


def validate(args, model, valid_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_val_loss = total_loss / len(valid_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")


# Arguments setup (example)
args = {
    'model_type': 'gpt2',
    'batch_size': 16,
    'learning_rate': 1e-5,
    'num_epochs': 3,
    'data_type': 'text',
    'shared_attention': 1,
    'shared_module': 0,
    'attention_fusion_type': 'concat',
    'single_input': False,
    'beam_size': 5,
    'bs_temperature': 1.0,
    'bs_nucleus_p': 0.9,
    'annealing_topk': 40,
    'length_penalty': 1.0,
    'annealing': False,
    'diversity_coef': 1.0,
    'response_k': 5,
}

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model with your data
train(args, model, tokenizer, 'path_to_train_data.csv', 'path_to_valid_data.csv')
