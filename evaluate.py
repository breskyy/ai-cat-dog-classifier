import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from model import CatDogCNN
from dataloader import get_dataloaders

# Konfiguracja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data"

# Dane i model
_, val_loader = get_dataloaders(data_dir)
model = CatDogCNN().to(device)
model.load_state_dict(torch.load("model/cnn_model.pth", map_location=device))
model.eval()

# Ewaluacja
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.round(outputs).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().reshape(-1, 1))

# Metryki
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print("Confusion Matrix:")
print(cm)