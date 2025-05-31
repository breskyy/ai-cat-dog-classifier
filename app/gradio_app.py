import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from model import CatDogCNN

# Konfiguracja modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogCNN().to(device)
checkpoint = torch.load("model/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transformacja obrazu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def classify_image(img):
    if img is None:
        return None
    
    # Konwersja do RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Predykcja
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        confidence = torch.sigmoid(output).item()
        return {"Kot": 1 - confidence, "Pies": confidence}

# Tworzenie interfejsu
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Klasyfikator Kotów i Psów",
    description="Wgraj zdjęcie kota lub psa, a model określi co widzi i z jaką pewnością.",
    examples=[
        ["example_images/test_cat.jpg"],
        ["example_images/test_dog.jpg"]
    ]
)

# Uruchomienie interfejsu
if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)