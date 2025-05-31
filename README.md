# Klasyfikator Kotów i Psów

## Opis Projektu
Ten projekt implementuje model głębokiego uczenia do klasyfikacji zdjęć kotów i psów. Wykorzystuje architekturę CNN (Convolutional Neural Network) i został wytrenowany na zbiorze 8000 zdjęć treningowych.

## Struktura Projektu
- `data/` - Zbiór danych treningowych i testowych (8000 zdjęć treningowych, 2000 testowych)
- `saved_models/` - Zapisane modele po treningu
- `example_images/` - Przykładowe zdjęcia do testowania
- `notebooks/` - Jupyter Notebook z implementacją i treningiem modelu
- `app/` - Aplikacja demonstracyjna z interfejsem Gradio

## Instalacja i Uruchomienie

### 1. Pobranie danych
Dane treningowe i testowe nie są dołączone do repozytorium ze względu na ich rozmiar. Możesz je pobrać z:
- [Link do danych treningowych](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- Po pobraniu, rozpakuj pliki do folderu `data/` zachowując strukturę:
  ```
  data/
  ├── train/
  │   ├── cat/
  │   └── dog/
  └── test/
      ├── cat/
      └── dog/
  ```

### 2. Przygotowanie środowiska
```bash
# Stwórz wirtualne środowisko Python
python3 -m venv venv

# Aktywuj środowisko
# Na macOS/Linux:
source venv/bin/activate
# Na Windows:
# venv\\Scripts\\activate

# Zainstaluj wymagane pakiety
pip3 install -r requirements.txt
```

### 3. Uruchomienie
```bash
# Uruchom Jupyter Notebook
jupyter notebook

# Otwórz notebooks/cat_dog_classifier-30-epochs.ipynb
```

## Wyniki
Model osiąga dokładność ~90% na zbiorze testowym. Szczegółowe wyniki i analiza znajdują się w notebooku.

## Wymagania systemowe
- Python 3.9 lub nowszy
- Minimum 8GB RAM
- Około 2GB wolnego miejsca na dysku (dla danych i modeli)

## Dataset
Dataset zawiera:
- 8000 zdjęć treningowych (po 4000 kotów i psów)
- 2000 zdjęć testowych (po 1000 kotów i psów)

## Model
Model został wytrenowany przez 30 epok i osiąga dokładność ~90% na zbiorze testowym.

## Uwagi
- Pierwsze uruchomienie może potrwać dłużej ze względu na instalację zależności
- W przypadku problemów z pamięcią, możesz zmniejszyć batch_size w notebooku

## Licencja
Dataset: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) 