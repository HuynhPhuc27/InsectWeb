## This is file for training multimodal by integrating VIT and BERT

#PATCH WISE 1 DIRECTION
import os
import pandas as pd
import torch
from PIL import Image
from transformers import ViTModel, AutoImageProcessor, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

class MultimodalDataset(Dataset):
    def __init__(self, df, vit_processor, bert_tokenizer, root_images, class_id_to_idx=None, num_classes=None):
        self.df = df
        self.vit_processor = vit_processor
        self.bert_tokenizer = bert_tokenizer
        self.root_images = root_images
        if class_id_to_idx is None:  
            self.class_ids = sorted(set(df['class_id'].apply(lambda x: int(x)).unique()))
            self.num_classes = len(self.class_ids)
            self.class_id_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        else:  
            self.class_id_to_idx = class_id_to_idx
            self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.root_images, row['image_path'])).convert('RGB')
        text = row['text']
        if pd.isna(text):
            text = ""
        class_id = int(row['class_id'])
        species = row['species']

        image_inputs = self.vit_processor(images=image, return_tensors="pt")
        
        text_input = text
        target = self.class_id_to_idx[class_id]
        return image_inputs, text_input, target, species, row['image_path']

from torch.nn.utils.rnn import pad_sequence

def make_collate_fn(bert_tokenizer):
    def collate_fn(batch):
        image_inputs, texts, targets, species, image_paths = zip(*batch)
        
        image_batch = {}
        for key in image_inputs[0]:
            image_batch[key] = torch.cat([x[key] for x in image_inputs], dim=0)
        encoded = bert_tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        targets = torch.tensor(targets, dtype=torch.long)
        return image_batch, encoded, targets, species, image_paths
    return collate_fn

class PatchWiseCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, text_features, patch_features):
        batch_size = text_features.size(0)
        num_patches = patch_features.size(1)

        query = self.query(text_features).view(batch_size, 1, self.num_heads, -1).transpose(1, 2)
        key = self.key(patch_features).view(batch_size, num_patches, self.num_heads, -1).transpose(1, 2)
        value = self.value(patch_features).view(batch_size, num_patches, self.num_heads, -1).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, value).transpose(1, 2).reshape(batch_size, 1, -1)
        output = self.out(attended_features.squeeze(1))

        return output, attention_weights

class MultimodalClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.cross_attention = PatchWiseCrossAttention(dim=dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, text_features, patch_features):
        attended_features, attention_weights = self.cross_attention(text_features, patch_features)
        logits = self.fc(attended_features)
        return logits, attention_weights

def evaluate_model(classifier, dataloader, criterion, device, num_classes, dataset_name="Validation"):
    classifier.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            image_inputs, text_inputs, targets, _, _ = batch

            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            targets = targets.to(device)

            with torch.no_grad():
                vit_outputs = vit_model(**image_inputs)
                patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :] 

            with torch.no_grad():
                bert_outputs = bert_model(**text_inputs)
                text_embeddings = bert_outputs.pooler_output

            logits, _ = classifier(text_embeddings, patch_embeddings)

            loss = criterion(logits, targets)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()  
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', labels=range(num_classes))

    classifier.train()  
    return avg_loss, accuracy, f1


def evaluate_model_test(classifier, dataloader, criterion, device, num_classes, dataset_name="Test"):
    classifier.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            image_inputs, text_inputs, targets, _, _ = batch
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            targets = targets.to(device)

            vit_outputs = vit_model(**image_inputs)
            patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]  
            bert_outputs = bert_model(**text_inputs)
            text_embeddings = bert_outputs.pooler_output

            logits, _ = classifier(text_embeddings, patch_embeddings)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', labels=range(num_classes))
    print(f"\n {dataset_name} Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1-score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, labels=range(num_classes), digits=4))
    return avg_loss, accuracy, f1

def train_model(root_images, train_csv='train.csv', val_csv='val.csv', test_csv='test.csv', output_pth='weights.pth', batch_size=16, num_epochs=10):
    global vit_model, bert_model
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)
    bert_model.to(device)
    vit_model.eval()  
    bert_model.eval() 

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)


    train_dataset = MultimodalDataset(train_df, vit_processor, bert_tokenizer, root_images)
    num_classes = train_dataset.num_classes
    class_id_to_idx = train_dataset.class_id_to_idx


    val_dataset = MultimodalDataset(val_df, vit_processor, bert_tokenizer, root_images, class_id_to_idx=class_id_to_idx, num_classes=num_classes)
    test_dataset = MultimodalDataset(test_df, vit_processor, bert_tokenizer, root_images, class_id_to_idx=class_id_to_idx, num_classes=num_classes)

    collate_fn = make_collate_fn(bert_tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)


    num_classes = train_dataset.num_classes
    classifier = MultimodalClassifier(dim=768, num_classes=num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()  

    classifier.train()
    best_val_accuracy = 0.0
    patience = 5  
    counter = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            image_inputs, text_inputs, targets, _, _ = batch

            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            targets = targets.to(device)

            with torch.no_grad():
                vit_outputs = vit_model(**image_inputs)
                patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]  

            with torch.no_grad():
                bert_outputs = bert_model(**text_inputs)
                text_embeddings = bert_outputs.pooler_output

            logits, _ = classifier(text_embeddings, patch_embeddings)

            loss = criterion(logits, targets)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy, val_f1 = evaluate_model(classifier, val_dataloader, criterion, device, num_classes=num_classes, dataset_name="Validation")

        avg_train_loss = total_train_loss / len(train_dataloader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_train_loss/len(train_dataloader):.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            print(f"No improvement in val_loss for {counter} epochs")
            if counter >= patience:
                print("Early stopping triggered")
                break
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'class_ids': train_dataset.class_ids,
                'class_id_to_idx': train_dataset.class_id_to_idx
            }, "best_weights_images_florence.pth")
            print(f"Saved new best model at epoch {epoch+1} with Val Loss={val_loss:.4f}")

    test_loss, test_accuracy, test_f1 = evaluate_model_test(classifier, test_dataloader, criterion, device, num_classes=num_classes, dataset_name="Test")
    print(f"Final Test Results: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

    torch.save({
        'model_state_dict': classifier.state_dict(),
        'class_ids': train_dataset.class_ids,
        'class_id_to_idx': train_dataset.class_id_to_idx,
        'embeddings': [],  
        'attention_weights': [],  
        'image_paths': [],
        'species': []
    }, output_pth)

    classifier.eval()
    all_embeddings = []
    all_attention_weights = []
    all_image_paths = []
    all_species = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating embeddings"):
            image_inputs, text_inputs, _, species, image_paths = batch

            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            vit_outputs = vit_model(**image_inputs)
            patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :]

            bert_outputs = bert_model(**text_inputs)
            text_embeddings = bert_outputs.pooler_output

            attended_features, attention_weights = classifier.cross_attention(text_embeddings, patch_embeddings)

            all_embeddings.extend(attended_features.cpu())
            all_attention_weights.extend(attention_weights.cpu())
            all_image_paths.extend(image_paths)
            all_species.extend(species)


    torch.save({
        'model_state_dict': classifier.state_dict(),
        'class_ids': train_dataset.class_ids,
        'class_id_to_idx': train_dataset.class_id_to_idx,
        'embeddings': all_embeddings,
        'attention_weights': all_attention_weights,
        'image_paths': all_image_paths,
        'species': all_species,
    }, output_pth)
    print(f"Weights and embeddings saved to {output_pth}")

root_images = "..."
train_csv ="..."
val_csv ="..."
test_csv = "..."

train_model(root_images, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv, output_pth='...', batch_size=..., num_epochs=...)
