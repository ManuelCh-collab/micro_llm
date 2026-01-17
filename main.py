from app.embedding_logic import TransformerEmbedding
from app.tokenizer_logic import BPETokenizer
import torch

VOCAB_SIZE = 5000
D_MODEL = 256
MAX_SEQ_LEN = 512

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)

tokenizer.train(["entrenamiento.txt"])
tokenizer.save("kracker/kracker2.json")

#tokenizer.load("kracker/kracker2.json")

input_text = "Aprender Python es genial"
tokens_ids = tokenizer.encode(input_text)
print(f"Texto: {input_text}")
print(f"Tokens IDs: {tokens_ids}")

embedder = TransformerEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL,max_seq_len=MAX_SEQ_LEN)

input_tensor = torch.tensor(tokens_ids).unsqueeze(0)

with torch.no_grad():
    vectors = embedder(input_tensor)

print(f"Forma del tensor resultante: {vectors.shape}")
print(f"Valores del primer token (primeros 5 elementos): \n{vectors[0, 0, :100]}")