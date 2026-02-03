import torch
import torch.nn.functional as F
from app.config import Config
from app.tokenizer_logic import BPETokenizer
from app.network import GPTModel

class GPTGenerator:
    def __init__(self, model_path: str, tokenizer_path: str):
        # 1. Cargar el Tokenizer
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        
        # 2. Reconstruir la arquitectura del modelo
        self.model = GPTModel(
            vocab_size=Config.VOCAB_SIZE,
            d_model=Config.D_MODEL,
            n_heads=8,
            n_layers=4, # Asegúrate que esto coincida con tu entrenamiento
            max_seq_len=Config.MAX_SEQ_LEN
        )
        
        # 3. Cargar los pesos entrenados
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval() 
        print(f"Modelo cargado desde {model_path}")

    def generate(self, prompt: str, max_new_tokens: int = 40, temperature: float = 0.1, top_k: int = 5, repetition_penalty: float = 2.0):
        """
        Genera texto con filtros de repetición y Top-K para mejorar la coherencia.
        """
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -Config.MAX_SEQ_LEN:]
            
            with torch.no_grad():
                logits = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                # --- FILTRO 1: Penalización de Repetición ---
                # Castiga los tokens que ya aparecieron en la secuencia
                for token in set(input_ids[0].tolist()):
                    if logits[0, token] < 0:
                        logits[0, token] *= repetition_penalty
                    else:
                        logits[0, token] /= repetition_penalty

                # --- FILTRO 2: Top-K Sampling ---
                # Solo permite elegir entre las 'k' palabras más probables
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Convertir a probabilidades y elegir
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat((input_ids, next_token), dim=1)

        return self.tokenizer.decode(input_ids[0].tolist())

if __name__ == "__main__":
    MODEL_PATH = "kracker/ChefcitoGPT_final.pth"
    TOKEN_PATH = "kracker/kracker2.json"
    
    generator = GPTGenerator(MODEL_PATH, TOKEN_PATH)
    
    print("\n--- Generador de Texto Kracker (V2 Coherente) ---")
    while True:
        user_input = input("\nEscribe algo (o 'salir'): ")
        if user_input.lower() == 'salir': break
        
        # Bajamos un poco la temperatura a 0.7 para mayor seriedad
        resultado = generator.generate(user_input, max_new_tokens=100, temperature=0.4)
        print(f"\nIA: {resultado}")