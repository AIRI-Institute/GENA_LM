import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pickle

def description(path):
    df = pd.read_csv(path)
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    model.eval()
    
    embeddings_dict = {}
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            text_id = row["id"]
            text = row["description"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0) 
            embeddings_dict[text_id] = cls_embedding.cpu().numpy()
    
    save_path = path.replace(".csv", "") + "_embeddings.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"✅ Эмбеддинги сохранены в {save_path}")