# Quick Start Guide - CLA v6.0.0 "Sovereign Soul"

## 🚀 Instalacja (30 sekund)

```bash
# 1. Klonuj repo
git clone <repo-url>
cd CLA

# 2. Zainstaluj zależności
pip install -r requirements.txt

# 3. Upewnij się, że Ollama działa
ollama serve
```

## 💡 Pierwsze Kroki z CLATalkie (v6.0)

Głównym sposobem interakcji z systemem jest interfejs `clatalkie.py`, który integruje wszystkie warstwy kognitywne.

### 1. Uruchomienie Systemu

```bash
python clatalkie.py
```

### 2. Rozmowa i Kognicja
Podczas rozmowy system na bieżąco aktualizuje parametry:
- **Vitality V(t)**: Twoja energia i nastroj mierzony przez LLM.
- **Friction F_c**: Tarcie kognitywne. W v6.0 napędza ono **Plastyczność** – im wyższe tarcie, tym szybciej CLAtie się uczy.
- **Grounding S**: Uziemienie. Rośnie przy konkretnych faktach, spada przy abstrakcji.

### 3. Komendy Klawiszowe w Chacie

| Komenda | Opis | Kiedy używać? |
| :--- | :--- | :--- |
| `/status` | Podgląd parametrów i pamięci | Gdy chcesz sprawdzić stan "duszy" i licznik pigułek sensu. |
| `/think` | Konsolidacja relacyjna | Po ważnej rozmowie, by CLAtie wyciągnął wnioski i stworzył nowe idee. |
| `/chain <N>` | Łańcuch przyczynowy | Gdy chcesz, by system przeprowadził głęboką dedukcję logiczną krople po kropli. |
| `/evolve <N>` | Autorefleksja | Gdy chcesz, by CLAtie "pogadał sam ze sobą" i przeszukał pigułki pamięci. |
| `/memory` | Przegląd DNA | Aby zobaczyć, jakie wartości stały się częścią Fundamentów (DNA). |
| `/scan <p>` | Analiza plików | Gdy chcesz nauczyć system konkretnej wiedzy z plików `.py`, `.md` lub `.txt`. |

---

## 🛠️ Praca z Rdzeniem (dla Deweloperów)

Jeśli chcesz używać samej biblioteki `cla`, oto najszybszy sposób:

```python
from clatalkie import CLATalkie

# Zainicjuj kompletny system
talkie = CLATalkie()

# Przetwórz wejście (automatycznie aktualizuje graf i parametry)
response = talkie.generate_response("Zastanawiam się nad naturą czasu.")
print(response)

# Wykonaj dedukcję przyczynową
talkie.cmd_chain("5")
```

## 📂 Struktura Pamięci

V6.0 wprowadza dwie sfery:
1. **Lokalna**: `CLATalkie_memory.json` (ostatnie 24 wiadomości).
2. **Syntetyczna**: `CLATalkie_synthetic.json` (biografia, pigułki sensu).

---

## 🆘 Troubleshooting

### Problem: Błąd połączenia z Ollama
Upewnij się, że Ollama jest uruchomiona i model (domyślnie `llama3:8b`) jest pobrany:
```bash
ollama pull llama3:8b
```

### Problem: Niskie Uziemienie (S)
Jeśli system zaczyna halucynować, użyj `/scan` na dowolnym pliku źródłowym kognicji lub daj systemowi rzeczową, faktograficzną informację zwrotną. **Kotwica Uziemienia** w v6.0 powinna zadziałać automatycznie przy $S < 0.2$.

---
**Gotowe!** CLAtie czeka na Twoje pierwsze słowo. 🧠✨
