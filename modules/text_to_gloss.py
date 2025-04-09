import torch
import json
import re
import string
import openai
import spacy
from pathlib import Path
from utils.file_manager import load_file, save_file
from config import config


class TextToGloss:
    def __init__(self, model_path="checkpoints/best_model.pt"):
        self.input_file = config.OUTPUT_DIR / "transcription.txt"
        self.output_file = config.OUTPUT_DIR / "gloss.txt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load OpenAI API Key from environment or config
        openai.api_key = config.OPENAI_API_KEY
        
        # Load spaCy English model
        self.nlp = spacy.load("en_core_web_sm")

        self.text_word_to_index = {}
        self.gloss_word_to_index = {}
        self.gloss_index_to_word = {}
        
        self._load_vocabulary('checkpoints/transformer_model.pt.vocab.json')
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_vocabulary(self, vocab_path):
        """Load vocabulary files and initialize word-to-index mappings"""
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.text_word_to_index = vocab_data['text_word_to_index']
        self.gloss_word_to_index = vocab_data['gloss_word_to_index']
        self.gloss_index_to_word = {int(k): v for k, v in vocab_data.get('gloss_index_to_word', {}).items()}
        if not self.gloss_index_to_word:
            self.gloss_index_to_word = {idx: token for idx, token in enumerate(vocab_data['gloss_vocab'])}
    
    def _load_model(self, model_path):
        """Load the transformer model"""
        from models import TransformerModel
        checkpoint = torch.load(model_path, map_location=self.device)
        model = TransformerModel(
            len(self.text_word_to_index),
            len(self.gloss_index_to_word),
            embedding_dim=1024,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.1,
            max_len=100
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def _preprocess_text(self, text):
        """Clean the input text"""
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = " ".join(text.split())  # Normalize spaces
        return text
    
    def _tokenize_text(self, text):
        """Convert the cleaned text into tokens"""
        return [self.text_word_to_index.get(token, self.text_word_to_index.get("<unk>")) for token in text.split()]
    
    def _generate_gloss(self, tokenized_text):
        """Generate gloss for the given tokenized text."""
        with torch.no_grad():
            src = torch.tensor(tokenized_text, dtype=torch.long).unsqueeze(0).to(self.device)
            start_token = self.gloss_word_to_index.get("<start>", 0)
            ys = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
            for _ in range(100):
                src_padding_mask = (src == self.text_word_to_index.get("<pad>", 0)).to(self.device)
                tgt_mask = self.model.generate_square_subsequent_mask(ys.size(1)).to(self.device)
                tgt_padding_mask = torch.zeros(1, ys.size(1), dtype=torch.bool, device=self.device)
                memory = self.model.transformer.encoder(
                    self.model.positional_encoding(self.model.text_embedding(src)),
                    src_key_padding_mask=src_padding_mask
                )
                decoder_input = self.model.positional_encoding(
                    self.model.gloss_embedding(ys.long())
                )
                out = self.model.transformer.decoder(
                    decoder_input,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                prob = self.model.fc_out(out[:, -1])
                next_word = torch.argmax(prob, dim=1).item()
                if next_word == self.gloss_word_to_index.get("<end>", 0):
                    break
                ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long).to(self.device).fill_(next_word)], dim=1)
        
        prediction = []
        tokens = ys[0][1:]  # Skip <start> token
        for idx in tokens:
            token = self.gloss_index_to_word.get(idx.item(), None)
            if token in ("<end>", None):
                break
            if token.startswith("X"):
                token = token[1:]

            token_clean = token.strip().lower()    
            if token in ["be", "desc", "wh", "to", "he"]:
             continue

            prediction.append(token)
        
        return " ".join(prediction)
    
    def run(self):
        """Run the entire translation process"""
        try:
            print("Starting text-to-gloss translation...")
            text = load_file(self.input_file)
            print(f"Input text loaded: {text[:50]}...")
            
            preprocessed = self._preprocess_text(text)
            print(f"Preprocessed text: {preprocessed[:50]}...")
            
            tokenized = self._tokenize_text(preprocessed)
            print(f"Tokenized text: {tokenized[:10]}...")
            
            gloss = self._generate_gloss(tokenized)
            print(f"Generated gloss: {gloss}")
            
            save_file(self.output_file, gloss)
            print(f"Gloss saved to: {self.output_file}")
            
            return True
        except Exception as e:
            print(f"Error in text-to-gloss translation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
