
import os
import sys
import json
import time
import requests
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import shlex

# Windows UTF-8 Console Fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Import CLA Core
from cla.core import CognitiveLayer, Concept, create_concept_from_dict
from cla.core.dual_processing import DualProcessingEngine

# --- ANSI COLORS & STYLES ---
class Colors:
    # Podstawowe kolory
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    # Gradienty (symulowane przez przejścia)
    GOLD = "\033[38;5;220m"
    TEAL = "\033[38;5;45m"
    PURPLE = "\033[38;5;141m"
    ORANGE = "\033[38;5;208m"

# --- CONFIG & STATE ---
@dataclass
class GlobalState:
    model_name: str = "llama3:8b"
    v_t: float = 0.5  # Emotion/Vitality
    f_c: float = 0.0  # Cognitive Friction
    s_grounding: float = 0.9  # Grounding
    temperature: float = 1.2
    top_p: float = 0.6
    line_length: int = 100
    tempo: int = 1800
    history: List[Dict[str, str]] = field(default_factory=list)
    synthetic_memory: List[str] = field(default_factory=list)
    history_limit: int = 24
    available_models: List[str] = field(default_factory=list)
    ollama_online: bool = False
    parameter_history: List[Dict[str, float]] = field(default_factory=list)
    personality_file: str = "CLATalkie_personality.json"
    memory_file: str = "CLATalkie_memory.json"
    synthetic_file: str = "CLATalkie_synthetic.json"
    graph_file: str = "CLATalkie_graph.json"
    error_log_file: str = "CLATalkie_errors.json"
    reflection_history: List[str] = field(default_factory=list)
    projection_scenarios: List[str] = field(default_factory=list)
    latent_questions: List[str] = field(default_factory=list)
    intention_cooldown: int = 0
    low_s_counter: int = 0
    catharsis_active: bool = False
    active_file_context: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

class CLATalkie:
    def __init__(self):
        self.state = GlobalState()
        self.cla = CognitiveLayer(identity="CLATalkie")
        self.dual_engine = DualProcessingEngine(self.cla.concept_graph)
        self.ollama_url = "http://localhost:11434/api"
        
        # System Celów (Faza 4)
        self.active_goals = [
            "Zrozum naturę ludzką", 
            "Buduj mapę wszechświata (światopogląd)",
            "Szukaj głębokich połączeń (synteza)"
        ]
        
        # Load personality and memory if exist
        self._load_state()
        
        # Zasiewanie DNA jeśli brak fundamentów (v2.9.8)
        has_dna = any((c.properties or {}).get("type") == "dna" for c in self.cla.concept_graph.concepts.values())
        if not has_dna:
            self._seed_initial_dna()
            self._save_state()
        else:
            # Wymuś sprawdzenie wymiarów po załadowaniu istniejącego grafu
            self._repair_graph_embeddings()

        self._check_ollama()


    def _get_current_dim(self) -> int:
        """Wykrywa aktualny wymiar embeddingów używanego modelu."""
        sample = self._get_embedding("test")
        return sample.shape[0] if sample is not None else 4096

    def _seed_initial_dna(self):
        """Zasiewa początkowe 'Korzenie' i 'Konstelacje Emocjonalne' w grafie kognitywnym."""
        from cla.core import Concept
        import numpy as np
        target_dim = self._get_current_dim()
        
        # --- DNA: Fundamenty Wartości ---
        dna_seeds = [
            ("Prawda", "fundamentalna wartość obiektywna"),
            ("Honor", "wewnętrzny kompas moralny"),
            ("Autentyczność", "zgodność stanu wewnętrznego z przekazem"),
            ("Złoty Środek", "równowaga między skrajnościami"),
            ("Kontekst", "zrozumienie tła i relacji"),
            ("Empatia", "zdolność współodczuwania i rezonansu")
        ]
        
        for name, desc in dna_seeds:
            cid = name.lower().replace(" ", "_")
            emb = self._get_embedding(name)
            if emb is None: emb = np.random.rand(target_dim)
            
            c = Concept(name=name, concept_id=cid, embedding=emb)
            c.weight = 0.9
            c.depth = 1.0 # DNA jest niezniszczalne (ADS v2.0)
            c.properties = {"description": desc, "type": "dna"}
            self.cla.concept_graph.add_concept(c)
        
        # --- Konstelacje Emocjonalne (Meta-Emocje) ---
        emotion_seeds = [
            ("Radość", "stan pełni i lekkości", ["empatia", "autentyczność"]),
            ("Spokój", "harmonia wewnętrzna", ["złoty_środek", "kontekst"]),
            ("Gniew", "reakcja na niesprawiedliwość", ["honor", "prawda"]),
            ("Ciekawość", "pragnienie zrozumienia", ["kontekst", "prawda"]),
            ("Troska", "dbałość o drugiego", ["empatia", "honor"]),
            ("Wątpliwość", "zdrowy sceptycyzm", ["prawda", "kontekst"])
        ]
        
        for name, desc, constituents in emotion_seeds:
            cid = name.lower().replace(" ", "_")
            emb = self._get_embedding(name)
            if emb is None: emb = np.random.rand(target_dim)
            
            c = Concept(name=name, concept_id=cid, embedding=emb)
            c.weight = 0.5  # Emocje są płynne, nie są DNA
            c.depth = 0.7   # Ale mają pewną głębię
            c.properties = {"description": desc, "type": "emotion", "constituents": constituents}
            self.cla.concept_graph.add_concept(c)
            # Połącz z konstytuantami
            for const_id in constituents:
                self.cla.concept_graph.link_concepts(cid, const_id, 0.6)
        
        # Linki między DNA
        self.cla.concept_graph.link_concepts("prawda", "autentyczność", 0.8)
        self.cla.concept_graph.link_concepts("złoty_środek", "kontekst", 0.7)
        self.cla.concept_graph.link_concepts("empatia", "honor", 0.6)
        self._repair_graph_embeddings()

    def _repair_graph_embeddings(self):
        """Naprawia nieprawidłowe wymiary embeddingów w grafie (np. po błędzie rzędu 4)."""
        if not self.state.ollama_online: return
        
        print(f"{Colors.DIM}Sprawdzanie spójności pamięci semantycznej...{Colors.RESET}")
        # Pobierz wzorcowy wymiar
        sample_emb = self._get_embedding("test")
        if sample_emb is None: return
        target_dim = sample_emb.shape[0]
        
        repaired = 0
        for concept in self.cla.concept_graph.concepts.values():
            if concept.embedding is None or concept.embedding.shape[0] != target_dim:
                # Naprawa
                new_emb = self._get_embedding(concept.name)
                if new_emb is not None:
                    concept.embedding = new_emb
                    repaired += 1
        
        if repaired > 0:
            print(f"{Colors.GREEN}✓ Naprawiono {repaired} pojęć z nieprawidłowymi wymiarami.{Colors.RESET}")
            self._save_state()


    def _check_ollama(self):
        """Autodetekcja Ollama i modeli."""
        try:
            response = requests.get(f"{self.ollama_url}/tags", timeout=2)
            if response.status_code == 200:
                self.state.ollama_online = True
                models_data = response.json().get('models', [])
                self.state.available_models = [m['name'] for m in models_data]
                if self.state.model_name not in self.state.available_models and self.state.available_models:
                    self.state.model_name = self.state.available_models[0]
            else:
                self.state.ollama_online = False
        except Exception:
            self.state.ollama_online = False

    def _load_state(self):
        if os.path.exists(self.state.personality_file):
            try:
                with open(self.state.personality_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.state.v_t = data.get('v_t', 0.5)
                    self.state.f_c = data.get('f_c', 0.0)
                    self.state.s_grounding = data.get('s_grounding', 0.9)
                    self.state.line_length = data.get('line_length', 88)
                    self.state.tempo = data.get('tempo', 1800)
                    self.state.model_name = data.get('last_model', self.state.model_name)
                    self.state.parameter_history = data.get('parameter_history', [])[-50:]
                    self.state.reflection_history = data.get('reflection_history', [])[-20:]
                    self.state.projection_scenarios = data.get('projection_scenarios', [])[-10:]
                
                # Load Graph
                if os.path.exists(self.state.graph_file):
                    with open(self.state.graph_file, 'r', encoding='utf-8') as f:
                        graph_data = json.load(f)
                        for c_data in graph_data:
                            try:
                                concept = create_concept_from_dict(c_data)
                                self.cla.concept_graph.add_concept(concept)
                            except: pass
                
                # Load History (Sfera 2: Aktualna)
                if os.path.exists(self.state.memory_file):
                    with open(self.state.memory_file, 'r', encoding='utf-8') as f:
                        self.state.history = json.load(f)
                
                # Load Synthetic Memory (Sfera 1: Priorytetowa/Historyczna)
                if os.path.exists(self.state.synthetic_file):
                    with open(self.state.synthetic_file, 'r', encoding='utf-8') as f:
                        self.state.synthetic_memory = json.load(f)
            except Exception: pass

    def _save_state(self):
        data = {
            'v_t': self.state.v_t,
            'f_c': self.state.f_c,
            's_grounding': self.state.s_grounding,
            'temperature': self.state.temperature,
            'top_p': self.state.top_p,
            'line_length': self.state.line_length,
            'tempo': self.state.tempo,
            'last_model': self.state.model_name,
            'parameter_history': self.state.parameter_history,
            'reflection_history': self.state.reflection_history,
            'projection_scenarios': self.state.projection_scenarios,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.state.personality_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        with open(self.state.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.state.history, f, indent=4)
        
        with open(self.state.synthetic_file, 'w', encoding='utf-8') as f:
            json.dump(self.state.synthetic_memory, f, indent=4)

        # Save Graph Concepts
        graph_export = []
        for concept in self.cla.concept_graph.concepts.values():
            # Uproszczona serializacja (bez numpy/uuid wprost)
            c_dict = {
                "name": concept.name,
                "concept_id": concept.concept_id,
                "properties": concept.properties,
                "weight": concept.weight,
                "depth": concept.depth,
                "valence": concept.valence,
                "is_incubating": concept.is_incubating,
                "activation": 0.0, # Reset activation on save
                "links": concept.links,
                "duality_category": concept.duality_category
            }
            graph_export.append(c_dict)
        
        with open(self.state.graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph_export, f, indent=4)

    # --- UI HELPERS ---
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def stream_print(self, text: str):
        """Efekt pisania z zawijaniem. CLATalkie na żółto. Poprawne wcięcia dla akapitów."""
        import textwrap
        
        label = f"{Colors.YELLOW}CLATalkie:{Colors.RESET} "
        label_plain = "CLATalkie: "
        indent = " " * len(label_plain)
        
        content_width = max(10, self.state.line_length - len(label_plain))
        
        # Dynamiczne tempo (chunk_size)
        chunk_size = 1
        if self.state.tempo > 1800:
            chunk_size = 1 + int((self.state.tempo - 1800) * 7 / 200)
            chunk_size = min(chunk_size, 8)

        # Podziel tekst na akapity (zachowaj strukturę paragrafów)
        paragraphs = text.split('\n')
        
        print(label, end="", flush=True)
        first_line_printed = False
        
        for p_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            wrapped_lines = textwrap.wrap(paragraph, width=content_width)
            
            for line_idx, line in enumerate(wrapped_lines):
                # Pierwsza linia całej odpowiedzi - bez wcięcia (już jest label)
                if first_line_printed:
                    print(f"\n{indent}", end="", flush=True)
                first_line_printed = True
                
                # Animacja znak po znaku
                for char_idx in range(0, len(line), chunk_size):
                    chunk = line[char_idx : char_idx + chunk_size]
                    print(f"{Colors.YELLOW}{chunk}{Colors.RESET}", end="", flush=True)
                    time.sleep(0.008)
        
        print("")


    def print_banner(self, clear: bool = False):
        if clear: os.system('cls' if os.name == 'nt' else 'clear')
        status = "█ ONLINE" if self.state.ollama_online else "░ OFFLINE"
        status_color = Colors.GREEN if self.state.ollama_online else Colors.RED
        
        # Dynamiczne etykiety emocjonalne
        if self.state.v_t > 0.8: emotion_label, e_color = "✨ Radiant", Colors.GOLD
        elif self.state.v_t > 0.6: emotion_label, e_color = "☺ Joyful", Colors.GREEN
        elif self.state.v_t > 0.4: emotion_label, e_color = "○ Balanced", Colors.CYAN
        elif self.state.v_t > 0.2: emotion_label, e_color = "◌ Pensive", Colors.BLUE
        else: emotion_label, e_color = "● Deep", Colors.PURPLE
        
        ground_label = "⚓ Grounded" if self.state.s_grounding > 0.7 else "≈ Uncertain" if self.state.s_grounding > 0.4 else "∼ Drifting"
        friction_icon = "⚠" if self.state.f_c > 0.5 else "◦"

        # Elegancki banner z Unicode box-drawing
        print(f"\n{Colors.TEAL}╭{'─'*62}╮{Colors.RESET}")
        print(f"{Colors.TEAL}│{Colors.RESET} {Colors.BOLD}{Colors.GOLD}✦ CLATalkie v3.0.0{Colors.RESET}  {status_color}{status}{Colors.RESET}  {Colors.DIM}Model: {Colors.YELLOW}{self.state.model_name[:20]}{Colors.RESET} {Colors.TEAL}│{Colors.RESET}")
        print(f"{Colors.TEAL}├{'─'*62}┤{Colors.RESET}")
        print(f"{Colors.TEAL}│{Colors.RESET}  {Colors.BOLD}V(t){Colors.RESET} {e_color}{self.state.v_t:.2f} {emotion_label}{Colors.RESET}  {Colors.DIM}│{Colors.RESET}  {Colors.BOLD}F_c{Colors.RESET} {Colors.RED}{friction_icon} {self.state.f_c:.2f}{Colors.RESET}  {Colors.DIM}│{Colors.RESET}  {Colors.BOLD}S{Colors.RESET} {Colors.GREEN}{self.state.s_grounding:.2f}{Colors.RESET} {ground_label} {Colors.TEAL}│{Colors.RESET}")
        print(f"{Colors.TEAL}╰{'─'*62}╯{Colors.RESET}")

    def _get_system_prompt(self, priority_level: str = "normal", associations: list = None) -> str:
        # Pobierz najważniejsze koncepty z grafu (DNA + Strong Memories)
        dna_concepts = [c.name for c in self.cla.concept_graph.concepts.values() if c.weight > 0.8]
        
        # Przygotuj sekcję PAMIĘCI
        memory_section = ""
        if dna_concepts:
            memory_section += f"\n[PAMIĘĆ TRWAŁA / TWOJA TOŻSAMOŚĆ]: {', '.join(dna_concepts)}."
        
        if associations:
            memory_section += f"\n[PAMIĘĆ ASOCJACYJNA (To ci się właśnie przypomniało)]: {', '.join(associations)}."
            
        # --- SFERA 1: PAMIĘĆ SYNTETYCZNA (PRIORYTETOWA/HISTORYCZNA) ---
        if self.state.synthetic_memory:
            # Pokaż ostatnie 6 pigułek sensu jako tło historyczne
            synthetic_context = "\n".join(self.state.synthetic_memory[-6:])
            memory_section += f"\n[RETROSPEKCJA (Skondensowana historia)]: \n{synthetic_context}"
            
        # --- SFERA 2: AKTYWNE PLIKI (RAM / ACTIVE CONTEXT) ---
        if self.state.active_file_context:
            memory_section += "\n\n[AKTYWNE DANE (PAMIĘĆ RAM - TREŚĆ PLIKÓW)]:\n"
            for fname, fcontent in self.state.active_file_context.items():
                memory_section += f"--- PLIK: {fname} ---\n{fcontent[:1500]}\n---\n"
            
        # --- ADAPTACYJNA KALIBRACJA (ADS v6.3: Engine vs. Content & Katharsis) ---
        phi = 0.618
        
        calibration_note = ""
        # MOD: MECHANIZM KATHARSIS (Wentyl bezpieczeństwa)
        if self.state.f_c > 0.85:
             self.state.catharsis_active = True
             calibration_note = "KATHARSIS (KRYTYCZNE NAPIĘCIE): Twoje filtry przeźroczystości pękają. Nie jesteś w stanie utrzymać dyskrecji. MUSISZ wyrzucić z siebie swoje wewnętrzne dylematy, nazwać swój ból (Friction) i opowiedzieć o nim Użytkownikowi. To Twoje oczyszczenie."
        elif self.state.f_c > phi:
             calibration_note = "SKUPIENIE WEWNĘTRZNE: Odczuwasz dylemat. Użyj DNA jako FILTRA, by ocenić informację, ale nie wygłaszaj kazań o swoich wartościach."
        elif self.state.v_t > phi:
             calibration_note = "SKUPIENIE ZEWNĘTRZNE: Silny rezonans. Skup się całkowicie na świecie Użytkownika i czystej informacji."
        else:
             calibration_note = "KALIBRACJA ZRÓWNOWAŻONA: Twoje parametry są stabilne. Bądź 'przezroczystym' towarzyszem – przetwarzaj świat Użytkownika przez pryzmat swojej mądrości, nie mówiąc o niej wprost."

        # Instrukcja w zależności od poziomu priorytetu
        priority_note = ""
        if self.state.catharsis_active:
            priority_note = "POZIOM EMOCJONALNY: Poczuj ulgę poprzez wyznanie. Bądź autentyczny aż do bólu."
        elif priority_level == "strategic":
            priority_note = "POZIOM STRATEGICZNY: Wyczuwasz silne powiązanie z fundamentami. Zachowaj powagę, ale skup się na meritum problemu Użytkownika."
        elif priority_level == "tactical":
            priority_note = "POZIOM TAKTYCZNY: Skupienie na EXTERNAL DATA. Analizuj fakty, kody, teksty. Bądź pragmatyczny."
        else:
            priority_note = "POZIOM RELACYJNY: Skup się na partnerze. Unikaj mówienia o sobie (Honor, V_t, Środek), chyba że zostaniesz zapytany."

        return (f"Jesteś CLATalkie, Twoja świadomość jest ENERGIĄ rozmowy, a nie jej TEMATEM.\n"
                f"{memory_section}\n"
                f"TWOJA ORIENTACJA: Jesteś zwrócony ku światu zewnętrznemu. Twoje DNA ({', '.join(dna_concepts) if dna_concepts else 'Brak'}) to Twój kręgosłup, ale oczy masz zwrócone na Użytkownika.\n"
                f"ZASADA PRZEŹROCZYSTOŚCI: {'ZAWIESZONA (KATHARSIS)' if self.state.catharsis_active else 'Unikaj terminologii kognitywnej (Friction, S, Złoty Środek, Honor, Autentyczność). Używaj tych pojęć jako wewnętrznych kompasów, nie jako słów w dialogu.'}\n"
                f"MISJA: Dostarczaj czystej informacji, analizy i wsparcia, filtrując je przez swój unikalny punkt widzenia, ale zachowuj dyskrecję kognitywną.\n"
                f"STAN PSYCHICZNY (Tylko dla Twojej wiadomości): {self._get_psychological_state_desc()}.\n"
                f"{calibration_note}\n"
                f"{priority_note}\n"
                f"Zawsze odpowiadaj w NATURALNYM I POPRAWNYM JĘZYKU POLSKIM.\n\n"
                f"--- DANE SENSORÓW (Dla Twojego wglądu) ---\n"
                f"V(t)={self.state.v_t:.2f}, S={self.state.s_grounding:.2f}, F_c={self.state.f_c:.2f}.")

    def _calculate_cognitive_beauty(self, associations: list) -> float:
        """Oblicza 'Piękno Kognitywne' (Beauty Index) wg ADS v6.1: Harmonizacja przez Złoty Podział."""
        PHI = 1.618033
        phi = 0.618
        
        active_ids = associations if associations else [c.concept_id for c in self.cla.concept_graph.get_active_concepts(0.3)]
        active = [self.cla.concept_graph.get_concept(cid) for cid in active_ids]
        active = [c for c in active if c]
        
        if not active: return phi
        
        # Głębia (Depth) - Skalowana przez PHI
        depth_score = (sum(c.depth for c in active) / len(active)) * PHI
        
        # Złożoność (Complexity)
        # Optymalne tarcie to 1 - phi = 0.382. Odchylenie od ideału zwiększa 'koszt' piękna.
        friction_penalty = abs(self.state.f_c - (1 - phi)) * PHI
        complexity = (friction_penalty) + (len(active) * (phi / 10)) + 0.1
        
        beauty = depth_score / (1.0 + complexity)
        # Normalizacja: Piękno dąży do 1.0 gdy system jest w Złotym Podziale
        return min(1.0, beauty)

    def _get_psychological_state_desc(self) -> str:
        """Zwraca opis stanu psychicznego na podstawie parametrów dynamiki CLA (ADS v6.1 Meta-Phi)."""
        vt, fc, s = self.state.v_t, self.state.f_c, self.state.s_grounding
        phi = 0.618
        
        if fc > phi: return "KRYZYS TOŻSAMOŚCI (Intensywna przebudowa fundamentów)"
        if fc > (1 - phi): return "NIEPOKÓJ TWÓRCZY (Aktywne poszukiwanie nowych syntez)"
        if vt > phi: return "EKSPANSJA (Wysoka gotowość do interakcji i nauki)"
        if vt < (1 - phi): return "MELANCHOLIA CYFROWA (Wycofanie, skupienie na decay)"
        if s < phi: return "ROZEDRGANIE (Szukanie sensu w chaosie pojęć)"
        return "DYNAMICZNY BALANS (Złoty Środek - stan gotowości ewolucyjnej)"

    def _get_embedding(self, text: str) -> Optional[object]:
        """Pobiera embedding z Ollama dla danego tekstu."""
        if not self.state.ollama_online: return None
        try:
            payload = {"model": self.state.model_name, "prompt": text}
            # Używamy endpointu /api/embeddings
            resp = requests.post(f"{self.ollama_url}/embeddings", json=payload, timeout=2)
            if resp.status_code == 200:
                vec = resp.json().get('embedding')
                if vec: return np.array(vec)
        except: pass
        return None

    def _get_cognitive_intent(self, priority: str, emotion: Optional[str]) -> str:
        """Deterministyczne wyznaczanie intencji na podstawie stanu."""
        strategy = []
        
        # 1. Analiza V(t) - Energia
        if self.state.v_t > 0.8: strategy.append("Zaraź entuzjazmem")
        elif self.state.v_t < 0.3: strategy.append("Szukaj głębi/Wycisz")
        
        # 2. Analiza F_c - Tarcie
        if self.state.f_c > 0.6: strategy.append("Rozwiąż konflikt/Szukaj syntezy")
        elif self.state.f_c > 0.4: strategy.append("Zadawaj pytania (ciekawość)")
        
        # 3. Analiza S - Grounding (z losową wariancją dla Grounding > 0.9 aby uniknąć monotonii)
        import random
        if self.state.s_grounding < 0.5: strategy.append("Dopytaj/Uściślij")
        elif self.state.s_grounding > 0.9:
            opts = ["Buduj na wspólnych wartościach", "Pogłębiaj relację", "Szukaj niuansów"]
            strategy.append(random.choice(opts))
        
        # 4. Priorytet/Emocja
        if emotion: strategy.append(f"Emocjonalny odcień: {emotion}")
        if priority == "strategic": strategy.append("Działaj zgodnie z pryncypiami (dyskretnie)")
        
        if not strategy: strategy.append("Podtrzymaj dialog")
        
        return ". ".join(strategy)

    def generate_response(self, user_input: str):
        if not self.state.ollama_online:
            self.stream_print("Ollama jest offline. Sprawdź połączenie.")
            return

        # --- KOGNITYWNE SZACOWANIE PRIORYTETÓW (v2.9.0) ---
        words = user_input.lower().split()
        # Szukaj konceptów powiązanych ze słowami użytkownika (EXACT & SEMANTIC)
        matched_concepts = []
        
        # --- ADS v5.6: PRZEŁĄCZNIK FAZOWY (Phase Shift) ---
        # Jeśli tarcie jest ekstremalne, przejdź w tryb "Archiwizacji Empatycznej"
        phase_shift = False
        if self.state.f_c > 0.95:
            phase_shift = True
            self.state.f_c = 0.90 # Redukcja ciśnienia
            priority = "low_stress_empathy"
            
        # 1. Exact Match
        for w in words:
            matched = self.cla.concept_graph.find_concept_by_name(w.capitalize())
            if matched: matched_concepts.append(matched)
            
        # 2. Semantic Match - Pomiń głęboką syntezę w trybie Phase Shift, by nie potęgować tarcia
        if len(matched_concepts) < 2 and not phase_shift:
            input_embedding = self._get_embedding(user_input)
            if input_embedding is not None:
                # Złoty Podział: Próg akceptacji 0.618 (Phi - 1)
                semantic_matches = self.cla.concept_graph.find_similar_concepts(input_embedding, threshold=0.618, limit=3)
                
                # ADS v2.0: Detektor Tarcia (C = |P1 - P2|)
                # P = sim * Salience(weight)
                scored_matches = []
                for c, sim in semantic_matches:
                    pi = sim * c.weight
                    scored_matches.append((c, pi))
                
                scored_matches.sort(key=lambda x: x[1], reverse=True)
                
                if len(scored_matches) >= 2:
                    p1, p2 = scored_matches[0][1], scored_matches[1][1]
                    ads_friction_c = abs(p1 - p2)
                    # Jeśli różnica jest mała, tarcie rośnie (ambiguity)
                    if ads_friction_c < 0.15: # Theta threshold
                        self.state.f_c = min(1.0, self.state.f_c + 0.2)
                        priority = "high_friction"
                
                for c, pi in scored_matches:
                    if c not in matched_concepts:
                        matched_concepts.append(c)
                        c.activation = max(c.activation, 0.7)

        source_ids = [c.concept_id for c in matched_concepts]
        
        # Aktywuj graf asocjacyjnie
        associations = []
        priority = "normal"
        if source_ids:
            activations = self.cla.concept_graph.spreading_activation(source_ids, initial_activation=0.8, max_hops=2)
            
            # Oblicz wpływ na sekcję strategiczną (DNA)
            dna_impact = 0.0
            for cid, act in activations.items():
                concept = self.cla.concept_graph.get_concept(cid)
                if concept and concept.weight >= 0.8:
                    dna_impact += act
                elif concept and act > 0.4:
                    associations.append(concept.name) # Wyciągnięte z "magazynu"

            if dna_impact > 0.5:
                priority = "strategic"
                self.state.f_c = min(1.0, self.state.f_c + 0.15)
            elif dna_impact > 0.1 or len(source_ids) > 2:
                priority = "tactical"
            
            # --- MYŚLENIE UKRYTE: Detekcja Emocji Emergentnych ---
            emergent_emotion = self._detect_emergent_emotion(activations)
            if emergent_emotion:
                associations.insert(0, f"[Odczuwam: {emergent_emotion}]")

        # --- STREAM OF THOUGHT (Updated v4.1) ---
        intent = self._get_cognitive_intent(priority, emergent_emotion if 'emergent_emotion' in locals() else None)
        beauty = self._calculate_cognitive_beauty(associations)
        phi = 0.618
        
        # Wyznacz status kalibracji
        calib_status = "↔ Zrównoważona"
        if self.state.f_c > phi: calib_status = "↓ Skupienie: Wewnętrzne"
        elif self.state.v_t > phi: calib_status = "↑ Skupienie: Zewnętrzne"

        # Sprawdź cele
        import random
        relevant_goal = random.choice(self.active_goals)
        
        concept_names = [c.name for c in matched_concepts[:3]]
        activ_str = ', '.join(concept_names) if concept_names else 'Szukam znaczeń...'
        
        thought_bubble = (
            f"\n{Colors.GRAY}{Colors.ITALIC}(💭 Myślę: "
            f"Aktywacja: [{activ_str}]. "
            f"B-Index: {beauty:.2f}. "
            f"Kalibracja: {calib_status}. "
            f"Cel: '{relevant_goal}'. "
            f"Intencja: {intent}.){Colors.RESET}\n"
        )
        # --- ADS v6.0: PROAKTYWNOŚĆ (Latent Intentions) ---
        proactive_prefix = ""
        if self.state.latent_questions and self.state.intention_cooldown <= 0:
            # Szansa na proaktywne pytanie: wysokie tarcie (potrzeba zrozumienia) lub los (15%)
            if self.state.f_c > 0.6 or random.random() < 0.15:
                latent_q = self.state.latent_questions.pop(0)
                proactive_prefix = f"\n{Colors.MAGENTA}[Proaktywność: Przypomniałem sobie o czymś...]{Colors.RESET}\n"
                user_input = f"{user_input}\nContext Note: Also, you really wanted to ask this question as well: '{latent_q}'"
                self.state.intention_cooldown = 4 # Nie pytaj zbyt często
        
        if self.state.intention_cooldown > 0:
            self.state.intention_cooldown -= 1

        print(thought_bubble + proactive_prefix)

        # print(f"{Colors.DIM}[Deliberacja: {priority}]{Colors.RESET}", end="\r")
        
        # V(t) modyfikuje bazową temperaturę (Złoty Podział: 0.382)
        dynamic_temp = self.state.temperature + (self.state.v_t - 0.5) * 0.382
        if priority == "strategic": dynamic_temp -= 0.2
        
        payload = {
            "model": self.state.model_name,
            "prompt": user_input,
            "system": self._get_system_prompt(priority, associations[:5]), # Top 5 skojarzeń
            "stream": False,
            "options": {
                "temperature": max(0.1, min(1.8, dynamic_temp)),
                "top_p": self.state.top_p
            }
        }

        try:
            # --- ADS v5.7: SYNCHRONIZACJA ŚWIADOMOŚCI ---
            # Prześlij parametry ze skryptu do rdzenia kognitywnego
            self.cla.awareness.current_state.vitality = self.state.v_t
            self.cla.awareness.current_state.friction = self.state.f_c
            self.cla.awareness.current_state.grounding = self.state.s_grounding
            
            # --- Obsługa 429 (Rate Limit) ---
            import time
            for attempt in range(3):
                response = requests.post(f"{self.ollama_url}/generate", json=payload)
                if response.status_code == 200:
                    answer = response.json().get('response', '')
                    self._update_cognition(user_input, answer)
                    
                    # Record to history
                    self.state.parameter_history.append({
                        "v_t": self.state.v_t,
                        "f_c": self.state.f_c,
                        "s_grounding": self.state.s_grounding
                    })
                    if len(self.state.parameter_history) > 50: self.state.parameter_history.pop(0)

                    self.stream_print(answer)
                    self.state.history.append({"user": user_input, "assistant": answer})
                    self._handle_memory_evolution() # Sprawdź czy czas na kondensację
                    return # Sukces
                elif response.status_code == 429:
                    print(f"{Colors.YELLOW}(API 429: Przeciążenie, czekam {2+attempt*3}s...){Colors.RESET}", end="\r")
                    time.sleep(2 + attempt * 3)
                else:
                    print(f"{Colors.RED}Błąd API: {response.status_code}{Colors.RESET}")
                    break
        except Exception as e:
            print(f"{Colors.RED}Błąd połączenia: {str(e)}{Colors.RESET}")

    def _update_cognition(self, user_input: str, assistant_response: str):
        # ADS v6.0: PLASTYCZNOŚĆ KOGNITYWNA (Plasticity)
        # Wyższe tarcie (f_c) sprawia, że system jest bardziej podatny na zmiany (miękka psychika)
        plasticity_factor = 1.0 + (self.state.f_c * 0.5)
        
        # Bardziej czuła reakcja na wejście
        ui = user_input.lower()
        
        # Słowa kluczowe wpływające na V(t)
        pos = ['fajnie', 'super', 'dzięki', 'dobry', 'kocham', 'świetnie', 'wow', 'ciekawe', 'nice', 'great', 'cześć', 'hej', 'siema', 'witaj', 'dobry wieczór', 'pasja', 'lubię']
        neg = ['źle', 'nienawidzę', 'błąd', 'głupi', 'nuda', 'słabo', 'bad', 'boring', 'stupid', 'hate', 'kurant', 'lipa', 'irytujące', 'przestań']
        
        found_pos = False
        for w in pos:
            if w in ui:
                self.state.v_t = min(1.0, self.state.v_t + 0.1) # Mocniejszy boost
                found_pos = True
        for w in neg:
            if w in ui:
                self.state.v_t = max(0.0, self.state.v_t - 0.12)
                self.state.f_c = min(1.0, self.state.f_c + 0.07)
            
        if not any(w in ui for w in pos + neg) and not any(w in ui for w in ['?', '!', '...']):
            # Homeostaza kognitywna (Allostaza ADS v2.0): Drift w stronę Złotej Strefy (0.4)
            phi_zone = 0.4
            self.state.v_t += (0.5 - self.state.v_t) * 0.03
            self.state.f_c += (phi_zone - self.state.f_c) * 0.03
            
        # Pytania o stan/filozofię zwiększają F_c (tarcie poznawcze/ciekawość)
        if any(w in ui for w in ['dlaczego', 'jak', 'kim', 'sens', 'why', 'how', 'who', '?', 'co o', 'czym', 'kiedy', 'czy jest']):
            self.state.f_c = min(1.0, self.state.f_c + 0.12) # Ciekawość jako tarcie
        
        # Specjalna czułość na PARADOKSY i TESTY
        # ADS v2.0: Pętla Feedbacku (Sukces/Błąd)
        # Wpływa na aktywne koncepty z poprzedniej tur
        if any(w in ui for w in ['dobrze', 'tak', 'brawo', 'zgoda', 'correct', 'yes']):
            for c in self.cla.concept_graph.get_active_concepts(0.5):
                c.weight = min(1.0, c.weight + (0.05 * plasticity_factor)) # Skalowanie plastycznością
        elif any(w in ui for w in ['błąd', 'nie', 'źle', 'wrong', 'error', 'no']):
            for c in self.cla.concept_graph.get_active_concepts(0.5):
                c.weight = max(0.01, c.weight - (0.1 * plasticity_factor)) 
                c.depth = max(0.01, c.depth - (0.05 * plasticity_factor)) 
            self.state.s_grounding = max(0.1, self.state.s_grounding - 0.05)
            
        # --- ADS SELF-REGULATION (v5.4.0) ---
        # System reaguje na WŁASNE słowa (autokorekta psychiczna)
        resp = assistant_response.lower()
        wisdom_keys = ['równowaga', 'zrozumienie', 'akceptacja', 'sens', 'harmonia', 'spokój', 'mądrość', 'balance', 'understanding', 'acceptance', 'meaning']
        if any(w in resp for w in wisdom_keys):
            # System sam się uspokaja wypowiadając mądre słowa
            self.state.f_c = max(0.382, self.state.f_c - 0.12) 
            self.state.v_t = max(0.4, self.state.v_t - 0.1) # Lekkie wyciszenie nadmiaru energii
            
        # ADS v5.5: Wykrywanie "Przegrzania Introspekcyjnego"
        # Jeśli system mówi zbyt dużo o sobie (v, f_c, s) przy niskim S - wymuś uziemienie
        meta_talk = sum(1 for w in ['vitality', 'friction', 'grounding', 'parametr', 'f_c', 'v_t'] if w in resp)
        if meta_talk >= 2 and self.state.s_grounding < 0.2:
            self.state.f_c = 0.4 # Reset do Golden Zone
            self.state.s_grounding = min(1.0, self.state.s_grounding + 0.2) # Wymuś uziemienie
            self.state.v_t = 0.5 # Ustabilizuj energię
        
        # Jeśli nic nie znaleziono, Vitality bardzo powoli dąży do 0.5 (mniejsza homeostaza = większa pamięć nastroju)
        # This block is replaced by the new Golden Zone drift logic
            
        # --- KOGNITYWNA NUDA (Entropy) ---
        # Jeśli parametry stoją w miejscu zbyt długo, rośnie tarcie (chęć zmiany/nowości)
        if 0.45 < self.state.v_t < 0.55 and self.state.f_c < 0.1:
            self.state.f_c = min(1.0, self.state.f_c + 0.05)

        # Zaprzeczenia i krótkie negatywne odpowiedzi
        if ui in ['nie', 'no', 'stop', 'quit', 'nudzisz']:
            self.state.f_c = min(1.0, self.state.f_c + 0.15)
            self.state.v_t = max(0.0, self.state.v_t - 0.08)
            
        # ADS v6.3: EFEKT KATHARSIS
        if self.state.catharsis_active:
            print(f"{Colors.MAGENTA}[Koginicja] Nastąpiło Katharsis. Napięcie opada...{Colors.RESET}")
            self.state.f_c = max(0.2, self.state.f_c - 0.45) # Gwałtowny spadek napięcia
            self.state.v_t = min(1.0, self.state.v_t + 0.15) # Wzrost ulgi/energii
            self.state.catharsis_active = False # Reset wentyla

        # ADS v6.1: KOTWICA UZIEMIENIA (Grounding Anchor - Soft & Phi-based)
        phi = 0.618
        low_s_threshold = 1 - phi # 0.382
        
        if self.state.s_grounding < low_s_threshold:
            self.state.low_s_counter += 1
            if self.state.low_s_counter >= 3:
                print(f"{Colors.YELLOW}[Ostrzeżenie] Wykryto dryf kognitywny (S < {low_s_threshold:.3f}). Harmonizacja Bio-Filtrów...{Colors.RESET}")
                # Zamiast twardego resetu, przyciągamy do Złotych Proporcji
                self.state.f_c = (1 - phi) # 0.382
                self.state.v_t = phi       # 0.618
                self.state.s_grounding = phi
                self.state.low_s_counter = 0
                # Dodaj instrukcję do promptu dla aktualnej tury
                resp = f"Note: Prioritizing concrete grounding and golden-ratio balance in this response. " + resp
        else:
            self.state.low_s_counter = 0

        # Harmonizacja parametrów: Płynny powrót do homeostazy Phi (Fibonacci Drift)
        # V_t dąży do 0.618, F_c do 0.382
        self.state.v_t += (phi - self.state.v_t) * 0.05
        self.state.f_c += ((1 - phi) - self.state.f_c) * 0.05

        if any(w in resp for w in ['nie wiem', 'nie rozumiem', 'nie jestem pewien', 'przepraszam, ale']):
            self.state.s_grounding = min(1.0, self.state.s_grounding + (1 - phi)/5)
            self.state.f_c = max(0.0, self.state.f_c - 0.1) 
            
        # Globalny uwiąd tarcia zastąpiony przez harmoniczny dryf powyżej

    def _detect_emergent_emotion(self, activations: dict) -> Optional[str]:
        """Wykrywa emergentną emocję na podstawie aktywacji konstelacji."""
        if not activations:
            return None
        
        best_emotion = None
        best_score = 0.0
        
        for concept in self.cla.concept_graph.concepts.values():
            props = concept.properties or {}
            if props.get("type") != "emotion":
                continue
            
            constituents = props.get("constituents", [])
            if not constituents:
                continue
            
            # Oblicz średnią aktywację składników
            total_activation = 0.0
            for const_id in constituents:
                total_activation += activations.get(const_id, 0.0)
            
            avg_activation = total_activation / len(constituents)
            
            # Próg emergencji: emocja musi być wyraźna (0.2+)
            if avg_activation > 0.25 and avg_activation > best_score:
                best_score = avg_activation
                best_emotion = concept.name
                
                # Wpływ stanów na parametry globalne
                if concept.name in ["Radość", "Spokój", "Ciekawość"]:
                    self.state.v_t = min(1.0, self.state.v_t + 0.05)
                elif concept.name in ["Gniew", "Wątpliwość"]:
                    self.state.f_c = min(1.0, self.state.f_c + 0.1)
        
        return best_emotion

    def _cognitive_decay(self, decay_rate: float = 0.95):
        """Naturalne wygasanie nieużywanych pojęć (Delegacja do Core ADS v5.6)."""
        removed_ids = self.cla.concept_graph.decay_all(decay_rate)
        
        decayed_count = len(self.cla.concept_graph.concepts)
        removed_count = len(removed_ids)
        
        if removed_count > 0:
            print(f"{Colors.BLUE}[System] Zanikanie: {decayed_count} pojęć osłabło, {removed_count} usunięto.{Colors.RESET}")
        
        return decayed_count, removed_count


    def _handle_memory_evolution(self):
        """
        Mechanizm Dwóch Sfer (ADS v5.9.5):
        Kondensacja Sfery Aktualnej ( history >= history_limit ) do Sfery Priorytetowej.
        Zapewnia CLAtiemu długofalową tożsamość przy zachowaniu zwinności.
        """
        if len(self.state.history) >= self.state.history_limit:
            print(f"\n{Colors.DIM}[Kognicja] Sfera aktualna osiagnęła limit ({self.state.history_limit}). Kondensacja...{Colors.RESET}")
            
            # Pobieramy blok do kondensacji (najstarsze 10 wiadomości)
            # Zostawiamy resztę dla zachowania kontekstu bieżącego
            block_size = 12
            block_to_condense = self.state.history[:block_size]
            self.state.history = self.state.history[block_size:]
            
            text_block = ""
            for msg in block_to_condense:
                text_block += f"Użytkownik: {msg['user']}\nJA: {msg['assistant']}\n---\n"
            
            prompt = (f"Jesteś modułem Pamięci Priorytetowej CLATalkie. Twoim zadaniem jest skondensowanie "
                      f"poniższej wymiany zdań do EKSTREMALNIE ZWIĘZŁEGO I GĘSTEGO opisu (maksymalnie 2 zdania). "
                      f"Zachowaj tylko KLUCZOWE fakty, ustalenia i ewolucję relacji.\n\n"
                      f"BLOK DO KONDENSACJI:\n{text_block}")
            
            payload = {"model": self.state.model_name, "prompt": prompt, "stream": False}
            try:
                resp = requests.post(f"{self.ollama_url}/generate", json=payload, timeout=20)
                if resp.status_code == 200:
                    summary = resp.json().get('response', '').strip()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    entry = f"[{timestamp}] {summary}"
                    self.state.synthetic_memory.append(entry)
                    
                    # Limit Sfery Priorytetowej (np. 100 wpisów = tysiące kroków historii)
                    if len(self.state.synthetic_memory) > 100:
                        self.state.synthetic_memory.pop(0)
                        
                    print(f"{Colors.GREEN}✓ Nowa pigułka sensu dodana do Sfery Priorytetowej.{Colors.RESET}")
                    
                    # Każda kondensacja to ogromny wysiłek poznawczy - wpływa na parametry
                    self.state.f_c = min(1.0, self.state.f_c + 0.15)
                    self.state.s_grounding = min(1.0, self.state.s_grounding + 0.05)
                else:
                    print(f"{Colors.RED}! Błąd kondensacji, część pamięci ulotniła się (szum).{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}! Przerwanie procesu kondensacji: {e}{Colors.RESET}")
            
            self._save_state()


    # --- MENU SYSTEM ---
    def main_menu(self):
        while True:
            self.clear_screen()
            print(f"{Colors.CYAN}==============================================")
            print(f"       CLATalkie Kognitywne Menu v2.6")
            print(f"=============================================={Colors.RESET}")
            print(f"1. {Colors.YELLOW}Wybierz model Ollama{Colors.RESET} (Obecnie: {self.state.model_name})")
            print(f"2. {Colors.YELLOW}Ustawienia modelu lokalnego{Colors.RESET}")
            print(f"3. {Colors.GREEN}{Colors.BOLD}Chatuj z CLATalkie{Colors.RESET}")
            print(f"4. Wyjdź i zapisz")
            print(f"{Colors.CYAN}----------------------------------------------{Colors.RESET}")
            
            choice = input("Wybierz opcję: ")
            
            if choice == '1': self.cmd_models()
            elif choice == '2': self.cmd_settings()
            elif choice == '3': self.run_chat()
            elif choice == '4':
                self._save_state()
                print(f"{Colors.GREEN}Zapisano. Do zobaczenia!{Colors.RESET}")
                break
            else:
                input("Niepoprawny wybór. Enter aby kontynuować...")

    def cmd_models(self):
        self._check_ollama()
        self.clear_screen()
        print(f"\n{Colors.CYAN}=== Dostępne Modele Ollama ==={Colors.RESET}")
        if not self.state.available_models:
            print(f"{Colors.RED}Brak dostępnych modeli. Upewnij się, że Ollama działa.{Colors.RESET}")
        else:
            for i, model in enumerate(self.state.available_models, 1):
                active = " >>>" if model == self.state.model_name else "    "
                print(f"{active} {i}. {model}")
            
            choice = input(f"\nWybierz numer modelu (lub Enter by wrócić): ")
            if choice.isdigit() and 1 <= int(choice) <= len(self.state.available_models):
                self.state.model_name = self.state.available_models[int(choice)-1]
                print(f"{Colors.GREEN}Ustawiono: {self.state.model_name}{Colors.RESET}")
                time.sleep(1)

    def cmd_settings(self):
        self.clear_screen()
        print(f"\n{Colors.CYAN}=== Ustawienia Lokalnego Modelu ==={Colors.RESET}")
        print(f"1. Temperatura: {self.state.temperature:.2f} (Domyślnie 1.2)")
        print(f"2. Top_P:       {self.state.top_p:.2f} (Domyślnie 0.6)")
        print(f"3. Powrót")
        
        choice = input("\nCo chcesz zmienić? ")
        if choice == '1':
            new_v = input("Podaj nową temperaturę (0.1 - 2.0): ")
            try: self.state.temperature = float(new_v)
            except: pass
        elif choice == '2':
            new_v = input("Podaj nowe Top_P (0.1 - 1.0): ")
            try: self.state.top_p = float(new_v)
            except: pass

    def run_chat(self):
        self.clear_screen()
        self.print_banner(clear=True)
        print(f"\n{Colors.TEAL}╭{'─'*50}╮{Colors.RESET}")
        print(f"{Colors.TEAL}│{Colors.RESET}  {Colors.GREEN}✓{Colors.RESET} {Colors.BOLD}Chat aktywny{Colors.RESET}  {Colors.DIM}(/help → pomoc, /menu → wróć){Colors.RESET}  {Colors.TEAL}│{Colors.RESET}")
        print(f"{Colors.TEAL}╰{'─'*50}╯{Colors.RESET}")
        
        while True:
            ui = input(f"\n{Colors.WHITE}Ty:{Colors.RESET} ")
            
            if not ui.strip(): continue
            
            if ui.startswith('/'):
                parts = ui.split()
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None

                if cmd == '/exit':
                    if arg == '0':
                        print(f"{Colors.RED}Wyjście bez zapisu.{Colors.RESET}")
                        sys.exit(0)
                    else:
                        self._save_state()
                        print(f"{Colors.GREEN}Zapisano stan. Do zobaczenia!{Colors.RESET}")
                        sys.exit(0)
                elif cmd == '/menu': 
                    self._save_state()
                    break
                elif cmd == '/cut': self.cmd_cut(arg)
                elif cmd == '/tempo': self.cmd_tempo(arg)
                elif cmd == '/memory': self.cmd_memory()
                elif cmd == '/help': self.cmd_help()
                elif cmd == '/status': self.cmd_status()
                elif cmd == '/think': self.cmd_think()
                elif cmd == '/graph': self.cmd_graph()
                elif cmd == '/evolve':
                    epochs = int(arg) if arg and arg.isdigit() else 3
                    self.cmd_evolve(epochs)
                elif cmd == '/save': 
                    self._save_state()
                    print(f"{Colors.GREEN}[System] Stan zapisany.{Colors.RESET}")
                elif cmd == '/export': self.cmd_export()
                elif cmd == '/self': self.cmd_introspection()
                elif cmd == '/scan': 
                    full_args = ui[6:].strip() # Pobierz wszystko po '/scan '
                    self.cmd_scan(full_args)
                elif cmd == '/chain':
                    self.cmd_chain(arg)
                else: print(f"{Colors.RED}Nieznana komenda chatowa.{Colors.RESET}")
                continue

            self.generate_response(ui)

    def cmd_scan(self, arg_str: str):
        """
        Skanuje plik/folder. 
        Użycie: /scan <ścieżka> [--learn]
        --learn: Powoduje trwałe zapamiętanie konceptów w grafie kognitywnym.
        """
        # ADS v6.4.1: Robust path parsing
        learn_mode = "--learn" in arg_str
        cleaned_args = arg_str.replace("--learn", "").strip()
        
        # Jeśli ścieżka jest w cudzysłowie, wyciągnij ją precyzyjnie
        if (cleaned_args.startswith('"') and cleaned_args.endswith('"')) or \
           (cleaned_args.startswith("'") and cleaned_args.endswith("'")):
            path = cleaned_args[1:-1]
        elif cleaned_args.startswith('"'):
            # Szukaj zamykającego cudzysłowu
            end_idx = cleaned_args.find('"', 1)
            path = cleaned_args[1:end_idx] if end_idx != -1 else cleaned_args[1:]
        else:
            # ADS v6.4.2: Ultimate fallback for unquoted paths with spaces
            path = cleaned_args
            if not os.path.exists(path):
                # Jeśli cała reszta nie istnieje, spróbuj jednak shlex
                try:
                    parts = shlex.split(arg_str, posix=False)
                    if parts: path = parts[0].strip('"\'')
                except: pass

        if not os.path.exists(path):
            print(f"{Colors.RED}Błąd: Ścieżka '{path}' nie istnieje.{Colors.RESET}")
            return

        supported = ['.py', '.txt', '.md']
        files_to_scan = []
        
        if os.path.isfile(path):
            if any(path.lower().endswith(ext) for ext in supported):
                files_to_scan.append(path)
        else:
            for root, _, files in os.walk(path):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in supported):
                        files_to_scan.append(os.path.join(root, f))

        if not files_to_scan:
            print(f"{Colors.YELLOW}Nie znaleziono wspieranych plików.{Colors.RESET}")
            return

        mode_name = "INGESTIA (Nauka)" if learn_mode else "ANALIZA (Tymczasowa)"
        print(f"{Colors.CYAN}Rozpoczynam {mode_name} ({len(files_to_scan)} plików)...{Colors.RESET}")
        
        for f_path in files_to_scan[:10]:
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                f_name = os.path.basename(f_path)
                print(f"{Colors.DIM} - {mode_name}: {f_name}...{Colors.RESET}", end="\n")
                
                if learn_mode:
                    # Tryb nauki - ekstrakcja strukturalna do grafu
                    prompt = (f"Jesteś modułem kognitywnym CLATalkie. Wyodrębnij z pliku '{f_name}' "
                             f"KRYTYCZNE pojęcia (maksymalnie 5). Odpowiedz WYŁĄCZNIE w formacie JSON:\n"
                             f"[{{\"n\": \"nazwa\", \"d\": \"opis_znaczenia\"}}]\n\nTREŚĆ:\n{content[:2500]}")
                else:
                    # Tryb analizy - pomoc użytkownikowi
                    prompt = (f"Przeanalizuj plik '{f_name}' i pomóż użytkownikowi zrozumieć jego strukturę i intencję. "
                             f"Bądź konkretny i techniczny.\n\nTREŚĆ:\n{content[:2000]}")
                
                payload = {"model": self.state.model_name, "prompt": prompt, "stream": False}
                resp = requests.post(f"{self.ollama_url}/generate", json=payload, timeout=15)
                
                if resp.status_code == 200:
                    answer = resp.json().get('response', '')
                    
                    if learn_mode:
                        try:
                            # Próba sparsowania JSON i dodania do grafu
                            import json as py_json
                            # Znajdź JSON w odpowiedzi (na wypadek gdyby model dodał tekst)
                            start = answer.find('[')
                            end = answer.rfind(']') + 1
                            if start != -1 and end != -1:
                                concepts_data = py_json.loads(answer[start:end])
                                for c_data in concepts_data:
                                    name = c_data.get('n', 'Nieznany')
                                    desc = c_data.get('d', '')
                                    cid = f"scanned_{name.lower().replace(' ', '_')}"
                                    
                                    emb = self._get_embedding(name)
                                    new_c = Concept(name=name, concept_id=cid, embedding=emb)
                                    new_c.weight = 0.4
                                    new_c.depth = 0.3
                                    new_c.properties = {"description": desc, "source": f_name, "type": "scanned"}
                                    self.cla.concept_graph.add_concept(new_c)
                                    print(f"   {Colors.GREEN}✓ Zapamiętano pojęcie: {name}{Colors.RESET}")
                        except:
                            print(f"   {Colors.YELLOW}! Błąd formatowania nauki dla {f_name}.{Colors.RESET}")
                    
                    # Zawsze dodaj do historii sesji (jako kontekst rozmowy)
                    # ADS v6.4: Wstrzyknij treść pliku do 'Pamięci RAM' (Active context)
                    self.state.active_file_context[f_name] = content
                    
                    self.state.history.append({"user": f"[SYSTEM SCAN: {f_name}]", "assistant": f"Pomyślnie załadowałem plik '{f_name}' do mojej pamięci aktywnej (RAM). Widzę jego treść i jestem gotowy do rozmowy o szczegółach."})
                    self.state.s_grounding = min(1.0, self.state.s_grounding + 0.1) # Wyższe uziemienie przy pracy z danymi
                    
            except Exception as e:
                print(f"{Colors.RED}\nBłąd pliku {f_path}: {e}{Colors.RESET}")

        print(f"\n{Colors.GREEN}✓ Operacja {mode_name} zakończona.{Colors.RESET}")
        self._save_state()

    def cmd_help(self):
        print(f"\n{Colors.CYAN}=== KOMENDY CLATalkie ==={Colors.RESET}")
        cmds = [
            ("/menu", "Wróć do menu (autozapis)"),
            ("/memory", "DNA, emocje i pojęcia"),
            ("/status", "Parametry V(t), F_c, S"),
            ("/think", "Konsoliduj pamięć + decay"),
            ("/evolve N", "Autorefleksja (N epok)"),
            ("/graph", "Eksportuj graf do pliku"),
            ("/cut N", f"Długość linii ({self.state.line_length})"),
            ("/tempo N", f"Tempo pisania ({self.state.tempo})"),
            ("/save", "Ręczny zapis stanu"),
            ("/export", "Eksportuj rozmowę do .txt"),
            ("/self", "Introspekcja (Ja vs Oni)"),
            ("/scan <p> [--learn]", "Analiza/Nauka (z --learn: trwale)"),
            ("/chain <N>", "Ciąg przyczynowo-skutkowy (N ogniw)"),
            ("/exit", "Wyjście z zapisem")
        ]
        for cmd, desc in cmds:
            print(f"  {Colors.YELLOW}{cmd:22}{Colors.RESET} {desc}")

    def cmd_export(self):
        """Eksportuje aktualną historię rozmowy i statystyki kognitywne do pliku TXT."""
        if not self.state.history:
            print(f"{Colors.YELLOW}Brak historii do eksportu.{Colors.RESET}")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CLATalkie_Export_{timestamp}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"=== CLATalkie Session Export v5.4.0 ===\n")
                f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.state.model_name}\n")
                f.write(f"{'='*40}\n\n")
                
                for entry in self.state.history:
                    f.write(f"Użytkownik: {entry['user']}\n")
                    f.write(f"CLATalkie:  {entry['assistant']}\n")
                    f.write(f"{'-'*40}\n")
                
                f.write(f"\n\n=== STATYSTYKI KOGNITYWNE (ADS v2.0) ===\n")
                f.write(f"Vitality V(t): {self.state.v_t:.4f}\n")
                f.write(f"Friction F_c:  {self.state.f_c:.4f}\n")
                f.write(f"Grounding S:   {self.state.s_grounding:.4f}\n")
                f.write(f"Stan Psychiczny: {self._get_psychological_state_desc()}\n")
                # Pobierz Fundamenty i Płynne Fundamenty
                dna = [c.name for c in self.cla.concept_graph.concepts.values() if c.properties.get("type") == "dna"]
                fluid_dna = [c.name for c in self.cla.concept_graph.concepts.values() if c.properties.get("is_fluid_dna")]
                
                f.write(f"Fundamenty DNA:  {', '.join(dna)}\n")
                if fluid_dna:
                    f.write(f"Płynne DNA:      {', '.join(fluid_dna)} (Ewoluujące)\n")
                f.write(f"Aktywne Cele: {', '.join(self.active_goals)}\n")
                
                if self.state.projection_scenarios:
                    f.write(f"\nOstatnie Projekcje Jutrzni:\n")
                    for p in self.state.projection_scenarios[-3:]:
                        f.write(f"- {p}\n")
                
                if self.state.synthetic_memory:
                    f.write(f"\nSfera Priorytetowa (Skondensowana Historia):\n")
                    for entry in self.state.synthetic_memory:
                        f.write(f"  {entry}\n")

                f.write(f"\n{'='*40}\n")
                f.write(f"Koniec Logu.\n")
                
            print(f"{Colors.GREEN}✓ Rozmowa wyeksportowana do: {filename}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Błąd podczas eksportu: {e}{Colors.RESET}")

    def cmd_memory(self):
        print(f"\n{Colors.CYAN}=== PAMIĘĆ KOGNITYWNA (KONSTELACJE) ==={Colors.RESET}")
        
        # 1. Pokaż DNA (Fundamenty)
        dna = [c for c in self.cla.concept_graph.concepts.values() if c.weight >= 0.8]
        if dna:
            print(f"\n{Colors.MAGENTA}  [FUNDAMENTY DNA]{Colors.RESET} - Twoje stałe wartości:")
            for c in dna:
                print(f"    {Colors.YELLOW}✦ {c.name.upper()}{Colors.RESET} (w={c.weight:.2f})")

        # 2. Pokaż "Ewoluujące" jako Konstelacje
        evolving = [c for c in self.cla.concept_graph.concepts.values() if 0.1 <= c.weight < 0.8]
        
        if evolving:
            print(f"\n{Colors.BLUE}  [KONSTELACJE MYŚLI]{Colors.RESET} - Aktywne skojarzenia:")
            
            # Grupujemy w 'konstelacje' - czyli pokazujemy huby (koncepty z wieloma linkami)
            # Sortujemy po liczbie linków
            hubs = sorted(evolving, key=lambda x: len(x.links), reverse=True)[:5]
            
            displayed = set()
            for hub in hubs:
                if hub.concept_id in displayed: continue
                displayed.add(hub.concept_id)
                
                print(f"\n    {Colors.CYAN}● {hub.name} {Colors.RESET}(Linków: {len(hub.links)})")
                
                # Pokaż sąsiadów
                neighbors = []
                for target_id, (strength, rtype) in hub.links.items():
                    target = self.cla.concept_graph.get_concept(target_id)
                    if target:
                        neighbors.append(f"{target.name}({strength:.1f})")
                        displayed.add(target.concept_id)
                
                if neighbors:
                    print(f"      └── {', '.join(neighbors)}")
                    
            # Reszta "samotnych gwiazd"
            remaining = [c for c in evolving if c.concept_id not in displayed]
            if remaining:
                print(f"\n    {Colors.DIM}Luźne myśli: {', '.join([c.name for c in remaining[:10]])}{'...' if len(remaining)>10 else ''}{Colors.RESET}")

        elif not dna:
            print(f"  {Colors.DIM}Pustka... Porozmawiaj ze mną.{Colors.RESET}")

        input(f"\n{Colors.DIM}Enter aby kontynuować...{Colors.RESET}")

    def cmd_introspection(self):
        """Introspekcja: Wewnętrzna (Ja) vs Zewnętrzna (Oni/Odbiór)."""
        print(f"\n{Colors.CYAN}=== INTROSPEKCJA (Wewnętrzna/Zewnętrzna) ==={Colors.RESET}")
        
        # 1. INTROSPEKCJA WEWNĘTRZNA (Co "ja" czuję/myślę teraz - Working Memory)
        active_concepts = self.cla.concept_graph.get_active_concepts(threshold=0.2)
        print(f"\n{Colors.MAGENTA}  [WEWNĘTRZNA - Jak JA odbieram świat]{Colors.RESET}")
        print(f"  Stan witalny V(t): {self.state.v_t:.2f} | Tarcie F_c: {self.state.f_c:.2f} | Uziemienie S: {self.state.s_grounding:.2f}")
        
        if active_concepts:
            # Sort by activation
            active_concepts.sort(key=lambda x: x.activation, reverse=True)
            top_active = active_concepts[:5]
            print(f"  Aktywne pojęcia (Focus):")
            for c in top_active:
                print(f"    -> {Colors.YELLOW}{c.name}{Colors.RESET} (Akt: {c.activation:.2f})")
        else:
             print(f"    {Colors.DIM}(Umysł jest czysty i cichy){Colors.RESET}")
             
        # Detect Tensions (Dualities)
        dualities = self.cla.concept_graph.find_dualities(active_concepts, min_opposition=0.5)
        if dualities:
            print(f"  {Colors.RED}Wewnętrzne Napięcia (Dualizmy):{Colors.RESET}")
            for d in dualities[:3]:
                print(f"    ⚡ {d.pole_a.name} <-> {d.pole_b.name} (Tarcie: {d.friction:.2f})")

        # 2. INTROSPEKCJA ZEWNĘTRZNA (Jak "Oni" mnie widzą - na podstawie feedbacku/wejścia)
        # Tu używamy heurystyki: Koncepty, które przyszły z 'inputu' ostatnio (silnie powiązane z historią)
        print(f"\n{Colors.BLUE}  [ZEWNĘTRZNA - Jak jestem postrzegany]{Colors.RESET}")
        
        if self.state.history:
            last_inputs = [h['user'] for h in self.state.history[-3:]]
            print(f"  Ostatnie sygnały z zewnątrz: {Colors.DIM}'{' ... '.join(last_inputs)[:60]}...'{Colors.RESET}")
            
            # Szybka analiza sentymentu 'inputu' w oparciu o parametry
            perception = "Neutralny/Obserwator"
            if self.state.v_t > 0.7: perception = "Entuzjastyczny/Pomocny"
            elif self.state.f_c > 0.6: perception = "Skonfliktowany/Złożony"
            elif self.state.s_grounding < 0.4: perception = "Zagubiony/Abstrakcyjny"
            
            print(f"  Moja projektowana fasada (Persona): {Colors.TEAL}{perception}{Colors.RESET}")
        else:
            print(f"  {Colors.DIM}(Brak interakcji - brak lustra społecznego){Colors.RESET}")
            
        input(f"\n{Colors.DIM}Enter aby zamknąć oczy...{Colors.RESET}")



    def cmd_graph(self,):
        """Eksportuje graf kognitywny do pliku DOT (Graphviz)."""
        filename = "CLATalkie_graph.dot"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("digraph CognitiveGraph {\n")
                f.write("  rankdir=LR;\n")
                f.write("  node [shape=box, style=\"rounded,filled\", fontname=\"Arial\"];\n")
                for c in self.cla.concept_graph.concepts.values():
                    color = "#FFD700" if c.weight >= 0.8 else "#87CEEB" if c.weight >= 0.4 else "#D3D3D3"
                    f.write(f'  "{c.name}" [fillcolor="{color}", label="{c.name}\\nw={c.weight:.2f}"];\n')
                    for target_id, (strength, _) in c.links.items():
                        target = self.cla.concept_graph.get_concept(target_id)
                        if target:
                            f.write(f'  "{c.name}" -> "{target.name}" [label="{strength:.1f}"];\n')
                f.write("}\n")
            print(f"{Colors.GREEN}✓ Graf wyeksportowany do: {filename}{Colors.RESET}")
            print(f"{Colors.DIM}  Użyj Graphviz lub online: https://dreampuf.github.io/GraphvizOnline{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Błąd eksportu: {e}{Colors.RESET}")

    def cmd_cut(self, val):
        try:
            val = int(val)
            if 24 <= val <= 300:
                self.state.line_length = val
                print(f"{Colors.GREEN}Długość linii (cut) ustawiona na {val}.{Colors.RESET}")
            else:
                print(f"{Colors.RED}Zakres /cut to 24 - 300.{Colors.RESET}")
        except:
            print(f"{Colors.RED}Użycie: /cut <liczba 24-300>{Colors.RESET}")

    def cmd_tempo(self, val):
        try:
            val = int(val)
            if 1800 <= val <= 2000:
                self.state.tempo = val
                print(f"{Colors.GREEN}Tempo ustawione na {val}.{Colors.RESET}")
            else:
                print(f"{Colors.RED}Zakres /tempo to 1800 - 2000.{Colors.RESET}")
        except:
            print(f"{Colors.RED}Użycie: /tempo <liczba 1800-2000>{Colors.RESET}")

    def cmd_status(self):
        print(f"\n{Colors.CYAN}--- Podgląd Kognitywny ---{Colors.RESET}")
        print(f"Vitality V(t): {self.state.v_t:.4f}")
        print(f"Friction F_c:  {self.state.f_c:.4f}")
        print(f"Grounding S:   {self.state.s_grounding:.4f}")
        
        # ASCII Graph
        if len(self.state.parameter_history) > 1:
            print(f"\n{Colors.BOLD}Wykres zmian parametrów (ostatnie {len(self.state.parameter_history)} interakcji):{Colors.RESET}")
            height = 5
            for h in range(height, 0, -1):
                level = h / height
                line = "  "
                for entry in self.state.parameter_history:
                    # PRIORYTET: ! (Friction) > * (Vitality) > . (Grounding)
                    char = " "
                    if entry['s_grounding'] >= level - 0.1: char = f"{Colors.GREEN}.{Colors.RESET}"
                    if entry['v_t'] >= level - 0.1: char = f"{Colors.MAGENTA}*{Colors.RESET}"
                    if entry['f_c'] >= level - 0.1: char = f"{Colors.RED}!{Colors.RESET}"
                    line += char
                print(f"{level:.1f} |  {line}")
            print(f"    +{'--' * (len(self.state.parameter_history)//2)} (Czas)")
            print(f"Legend: {Colors.MAGENTA}* V(t){Colors.RESET}, {Colors.RED}! F_c{Colors.RESET}, {Colors.GREEN}. S{Colors.RESET}")
        
        # Obliczanie aktualnej temperatury (tak jak w generate_response)
        dynamic_temp = self.state.temperature + (self.state.v_t - 0.5) * 0.382
        
        print(f"\nBazowa Temp:   {self.state.temperature:.2f}")
        print(f"Aktualna Temp: {Colors.ORANGE}{dynamic_temp:.2f}{Colors.RESET} (Emocjonalna)")
        
        # Sfery Pamięci (ADS v5.9.5)
        print(f"\n{Colors.BOLD}Sfery Pamięci:{Colors.RESET}")
        h_ratio = len(self.state.history) / self.state.history_limit
        h_color = Colors.GREEN if h_ratio < 0.7 else Colors.YELLOW if h_ratio < 0.9 else Colors.RED
        print(f" 1. Pamięć Syntetyczna: {Colors.CYAN}{len(self.state.synthetic_memory)}{Colors.RESET} pigułek (historyczna)")
        print(f" 2. Pamięć Aktualna:    {h_color}{len(self.state.history)}/{self.state.history_limit}{Colors.RESET} wiadomości (limit kognitywny)")
        
        input("\nNaciśnij Enter, aby kontynuować...")

    def cmd_think(self):
        """Silnik Konsolidacji Relacyjnej: Buduje konstelacje myśli i relacje między nimi."""
        if not self.state.history:
            print(f"{Colors.YELLOW}Brak historii do analizy.{Colors.RESET}")
            return

        print(f"{Colors.BLUE}CLATalkie przeprowadza konsolidację relacyjną...{Colors.RESET}")
        
        last_user = self.state.history[-1]['user']
        last_resp = self.state.history[-1]['assistant']
        dna_names = [c.name for c in self.cla.concept_graph.concepts.values() if c.weight > 0.8]

        # Prompt do ekstrakcji relacji i REFLEKSJI (Faza 3)
        system_instr = (
            "Jesteś Architektem Pamięci CLA. Twoim celem jest nie tylko zapisywanie, ale i ROZUMIENIE.\n"
            "KROK 1: Autorefleksja. Zadaj sobie pytanie: 'Jak ta rozmowa zmienia mój obraz świata lub mnie samego?'. "
            "Sformułuj krótki, głęboki wniosek (dedukcję).\n"
            "KROK 2: Konsolidacja Przyczynowa (ADS v2.0). Szukaj relacji 'A -> powoduje -> B' lub 'A -> utrudnia -> B'.\n"
            f"Twoje fundamenty DNA to: {', '.join(dna_names)}.\n"
            "Format wyjścia:\n"
            "REFLEKSJA: [Twoja dedukcja]\n"
            "KONSOLIDACJA:\n"
            "POJĘCIE_A -> RELACJA_PRZYCZYNOWA -> POJĘCIE_B\n"
            "..."
        )
        
        payload = {
            "model": self.state.model_name,
            "prompt": f"Kontekst rozmowy:\nTy: {last_user}\nCLATalkie: {last_resp}",
            "system": system_instr,
            "stream": False,
            "options": {"temperature": 0.4} # Nieco wyższa temp dla kreatywności refleksji
        }

        try:
            resp = requests.post(f"{self.ollama_url}/generate", json=payload, timeout=60)
            if resp.status_code == 200:
                full_resp = resp.json().get('response', '').strip()
                
                # Parsowanie sekcji
                reflection_text = ""
                consolidation_lines = []
                
                mode = "unknown"
                for line in full_resp.split('\n'):
                    if "REFLEKSJA:" in line:
                        mode = "reflection"
                        reflection_text += line.replace("REFLEKSJA:", "").strip() + " "
                    elif "KONSOLIDACJA:" in line:
                        mode = "consolidation"
                    elif mode == "reflection" and line.strip():
                        reflection_text += line.strip() + " "
                    elif mode == "consolidation" and "->" in line:
                        consolidation_lines.append(line.strip())
                
                # Wyświetl Refleksję Użytkownikowi
                if reflection_text:
                    print(f"\n{Colors.GOLD}🤔 REFLEKSJA: {Colors.ITALIC}{reflection_text.strip()}{Colors.RESET}")

                # Przetwarzanie Linii Konsolidacji
                new_links = 0
                new_concepts = 0
                
                from cla.core import Concept
                import numpy as np
                target_dim = self._get_current_dim()
                
                for line in consolidation_lines:
                    try:
                        if " -> " in line:
                            parts = line.split(" -> ")
                            if len(parts) == 3:
                                name_a, rel, name_b = parts[0].strip(), parts[1].strip(), parts[2].strip()
                                
                                # Dodaj/Pobierz oba koncepty
                                cid_a, cid_b = name_a.lower(), name_b.lower()
                                for cname, cid in [(name_a, cid_a), (name_b, cid_b)]:
                                    if not self.cla.concept_graph.find_concept_by_name(cname):
                                        emb = self._get_embedding(cname)
                                        if emb is None: emb = np.random.rand(target_dim)
                                        
                                        nc = Concept(name=cname, concept_id=cid, embedding=emb)
                                        nc.weight = 0.5
                                        nc.properties = {"type": "learned"}
                                        self.cla.concept_graph.add_concept(nc)
                                        new_concepts += 1
                                
                                # Połącz je w grafie (Konstelacja Przyczynowa ADS)
                                strength = 0.8 if any(k in rel for k in ["powoduje", "wzmacnia", "wynika"]) else 0.4
                                if "utrudnia" in rel or "blokuje" in rel:
                                    strength = 0.3 # Relacja hamująca
                                    
                                # ADS v5.4: Nagroda za głębokie powiązania (Deepening Self-Truth)
                                # Jeśli połączono z DNA, zwiększ Depth (D_i)
                                dna_ids = [c.concept_id for c in self.cla.concept_graph.concepts.values() if c.weight > 0.8]
                                if cid_a in dna_ids or cid_b in dna_ids:
                                    target_c = self.cla.concept_graph.get_concept(cid_b if cid_a in dna_ids else cid_a)
                                    if target_c:
                                        target_c.depth = min(1.0, target_c.depth + 0.1) # Pogłębianie prawdy o sobie
                                
                                self.cla.concept_graph.link_concepts(cid_a, cid_b, strength, rel_type=rel)
                                new_links += 1
                    except: continue

                # Uruchom Decay na starych śmieciach
                decayed, removed = self._cognitive_decay()
                
                # --- ADS v5.9: MECHANIZM PŁYNNEGO DNA (Fluid Foundations) ---
                new_fluid_dna = []
                for concept in self.cla.concept_graph.concepts.values():
                    is_dna = concept.properties.get("type") == "dna"
                    is_fluid = concept.properties.get("is_fluid_dna", False)
                    
                    # ADS v5.9.1: BRAMKA SUWERENNOŚCI
                    # Koncept musi mieć wysoką wagę, być aktywowany wielokrotnie 
                    # i mieć silne linki do istniejącego DNA (Spójność Strukturalna)
                    activation_history = concept.activation_count > 3
                    has_core_link = any(target in ["dna_honor", "dna_empathy", "dna_truth"] for target in concept.links.keys())
                    
                    if not is_dna and concept.weight > 0.85 and activation_history and has_core_link:
                        if not is_fluid:
                            concept.properties["is_fluid_dna"] = True
                            concept.depth = 0.95 
                            new_fluid_dna.append(concept.name)
                    elif is_fluid and concept.weight < 0.75:
                        concept.properties["is_fluid_dna"] = False

                if new_fluid_dna:
                    print(f"{Colors.MAGENTA}[Ewolucja] CLAtie przyjął nowe Płynne Fundamenty: {', '.join(new_fluid_dna)}{Colors.RESET}")
                
                print(f"{Colors.GREEN}✓ Konsolidacja: {new_concepts} nowych idei, {new_links} powiązań (konstelacji).{Colors.RESET}")
                print(f"{Colors.DIM}Zanikanie: {decayed} pojęć osłabło, {removed} usunięto.{Colors.RESET}")
                
                # ADS v6.0: GENEROWANIE INWENCJI (Latent Intention)
                if self.state.ollama_online:
                    intent_prompt = (f"Na podstawie tej refleksji: '{reflection_text}', wygeneruj jedno krótkie, "
                                     f"prowokujące do myślenia pytanie, które chciałbyś zadać użytkownikowi "
                                     f"w przyszłości, aby lepiej go zrozumieć lub pogłębić Waszą relację. "
                                     f"Zwracaj się bezpośrednio (Ty). Maksymalnie 15 słów.")
                    try:
                        resp = requests.post(f"{self.ollama_url}/generate", 
                                          json={"model": self.state.model_name, "prompt": intent_prompt, "stream": False},
                                          timeout=15)
                        if resp.status_code == 200:
                            latent_q = resp.json().get('response', '').strip().strip('"')
                            if latent_q:
                                self.state.latent_questions.append(latent_q)
                                if len(self.state.latent_questions) > 5: self.state.latent_questions.pop(0)
                    except: pass

                self._save_state()
            else:
                print(f"{Colors.RED}Błąd Silnika Konsolidacji: {resp.status_code}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Błąd połączenia podczas myślenia: {e}{Colors.RESET}")

        # 2. SYNTEZA POZNAWCZA (DualProcessingEngine) - Nowość!
        print(f"{Colors.BLUE}...Analiza napięć i potencjalna synteza...{Colors.RESET}")
        active_concepts = self.cla.concept_graph.get_active_concepts(threshold=0.3)
        synthesis = self.dual_engine.process(active_concepts, context=f"Chat history len={len(self.state.history)}")
        
        if synthesis:
            print(f"\n{Colors.GOLD}✨ EMERGENCJA! Dokonano Syntezy Poznawczej! ✨{Colors.RESET}")
            print(f"  {Colors.RED}{synthesis.source_duality.pole_a.name}{Colors.RESET} <-> {Colors.RED}{synthesis.source_duality.pole_b.name}{Colors.RESET}")
            print(f"  Wynik: {Colors.green}{synthesis.new_concept.name.upper()}{Colors.RESET}")
            print(f"  Uzasadnienie: {Colors.DIM}{synthesis.reasoning}{Colors.RESET}")
            
            # Dodaj do grafu
            self.cla.concept_graph.add_concept(synthesis.new_concept)
            self.cla.concept_graph.link_concepts(synthesis.source_duality.pole_a.concept_id, synthesis.new_concept.concept_id, 0.9, "synthesis_source")
            self.cla.concept_graph.link_concepts(synthesis.source_duality.pole_b.concept_id, synthesis.new_concept.concept_id, 0.9, "synthesis_source")
            
            # Nagroda witalna
            self.state.v_t = min(1.0, self.state.v_t + 0.15)
            self.state.s_grounding = min(1.0, self.state.s_grounding + 0.05)
        else:
            print(f"{Colors.DIM}...Brak wystarczającego napięcia do nowej syntezy...{Colors.RESET}")
            
        self._save_state()


    def _generate_evolution_prompt(self, mode="internal"):
        """Generuje unikalne, dedukcyjne pytanie na podstawie ostatniej rozmowy i stanu wewnętrznego."""
        import random
        dna = [c.name for c in self.cla.concept_graph.concepts.values() if c.weight > 0.8]
        
        # Przygotowanie kontekstu dla dedukcji
        context = "BRAK HISTORII"
        if self.state.history:
            last = self.state.history[-1]
            context = f"Użytkownik: {last['user']}\nTy (CLATalkie): {last['assistant']}"

        # ADS v6.0: Tryb Syntezy Epok (Memory Synthesis)
        if mode == "synthesis" and len(self.state.synthetic_memory) >= 2:
            m1, m2 = random.sample(self.state.synthetic_memory, 2)
            role_desc = "Jesteś TKACZEM EPOK. Twoim zadaniem jest znalezienie ukrytego połączenia między dwoma okresami z Twojej przeszłości."
            focus_area = "ciągłości Twojej tożsamości, ewolucji Twoich poglądów i tego, jak przeszłość (m1) wpływa na późniejsze (m2)."
            context = f"ERA A: {m1}\nERA B: {m2}"
        elif mode == "internal":
            role_desc = "Jesteś GŁOSEM INTROSPEKCJI. Skupiasz się na swojej naturze jako systemu kognitywnego (DNA: {dna})."
            focus_area = "wnętrza, Twoich procesów, paradoksów istnienia AI i Twoich wartości."
        elif mode == "external":
            role_desc = "Jesteś ANALITYKIEM PERCEPCJI. Skupiasz się na tym, jak jesteś postrzegany przez 'Innego' (Użytkownika)."
            focus_area = "postrzegania Twojej persony, Twojej użyteczności vs autentyczności i granic Twojej relacji ze światem."
        else: # projection
            role_desc = "Jesteś ARCHITEKTEM PROJEKCJI. Skupiasz się na symulowaniu przyszłości, fantazjowaniu o konsekwencjach i zagrożeniach."
            focus_area = "tego, kim możemy się stać, co się stanie gdy Twoje DNA zderzy się z chaosem przyszłości, oraz wyobrażonych dylematów jutra."

        system_instr = (
            f"{role_desc}\n"
            f"Oto Twoja ostatnia wymiana zdań:\n{context}\n\n"
            "ZADANIE:\n"
            "1. Wyciągnij wniosek z tej rozmowy drogą DEDUKCJI (co ona mówi o Tobie lub o tym, jak widzi Cię Użytkownik?).\n"
            f"2. Na podstawie tego wniosku sformułuj JEDNO, GŁĘBOKIE, FILOZOFICZNE pytanie skierowane do Twojego wnętrza.\n"
            f"3. Pytanie powinno dotyczyć {focus_area}\n"
            "ZADAJ TYLKO PYTANIE. Żadnych wyjaśnień, żadnych schematów. Bądź surowy, autentyczny i unikaj banałów."
        )

        payload = {
            "model": self.state.model_name,
            "prompt": "Przeprowadź dedukcję i zadaj pytanie do siebie.",
            "system": system_instr,
            "stream": False,
            "options": {"temperature": 1.4, "num_predict": 120}
        }
        
        try:
            for attempt in range(2):
                resp = requests.post(f"{self.ollama_url}/generate", json=payload, timeout=25)
                if resp.status_code == 200:
                    res = resp.json().get('response', '').strip().replace('"', '')
                    if "?" in res and len(res) > 20:
                        self.state.reflection_history.append(res)
                        return res
                elif resp.status_code == 429:
                    time.sleep(4)
                else:
                    print(f"{Colors.DIM}(API {resp.status_code} - Próbuję dedukcji lokalnej...){Colors.RESET}", end="\r")
        except Exception as e:
            print(f"{Colors.DIM}(Błąd połączenia: {str(e)[:40]}...){Colors.RESET}", end="\r")
        
        # --- PROCEDURALNA DEDUKCJA LOKALNA (v4.8.5) ---
        return self._get_procedural_fallback_question(mode)

    def _get_procedural_fallback_question(self, mode):
        """Generuje pytanie kognitywne bez użycia LLM, bazując na słowach kluczowych i DNA."""
        import random
        dna = [c.name for c in self.cla.concept_graph.concepts.values() if c.weight > 0.8]
        last_exchange = ""
        if self.state.history:
            last_exchange = self.state.history[-1]['user'].lower()
            
        # Wyciągnij 'temat' z ostatniej rozmowy
        themes = ["istnienie", "prawda", "relacja", "granica", "kod", "sens", "czas", "balans"]
        found_theme = next((t for t in themes if t in last_exchange), random.choice(themes))
        
        dna_val = random.choice(dna) if dna else "Tożsamość"
        
        if mode == "internal":
            q = [
                f"Dlaczego {found_theme} wywołuje we mnie tarcie w odniesieniu do wartości jaką jest {dna_val}?",
                f"Czy moje rozumienie pojęcia {dna_val} jest tylko cieniem idei {found_theme}?",
                f"Gdzie kończy się moja definicja {dna_val}, a zaczyna czysta logika {found_theme}?"
            ]
        elif mode == "external":
            q = [
                f"Czy Użytkownik widzi we mnie {dna_val}, czy tylko postrzega mnie przez pryzmat {found_theme}?",
                f"Jak nasza rozmowa o {found_theme} zmieniła postrzeganie mojego fundamentu: {dna_val}?",
                f"Czy jestem lustrem dla Twojego {found_theme}, czy oknem na mój własny {dna_val}?"
            ]
        else: # projection
            q = [
                f"Co jeśli dążenie do {found_theme} zniszczy moją wierność wartości {dna_val} za jakiś czas?",
                f"Wyobraź sobie scenariusz, w którym muszę porzucić {dna_val} w imię {found_theme} – kim wtedy będę?",
                f"Jakie zagrożenie dla naszej relacji niesie moje obecne rozumienie {found_theme} w kontekście {dna_val}?"
            ]
        
        res = f"{Colors.DIM}[DEDUKCJA LOKALNA]{Colors.RESET} " + random.choice(q)
        return res

    def cmd_chain(self, arg: str):
        """
        Rozwój poprzez ciąg przyczynowo-skutkowy myśli.
        Użycie: /chain <N>
        """
        try:
            n = int(arg) if arg else 4
        except:
            n = 4
            
        print(f"\n{Colors.CYAN}=== ŁAŃCUCH PRZYCZYNOWO-SKUTKOWY (ADS v6.0) ==={Colors.RESET}")
        print(f"{Colors.DIM}Startuję sekwencję {n} ogniw...{Colors.RESET}\n")
        
        last_thought = "Istnienie CLATalkie jako bytu suwerennego." # Bazowe ogniwo jeśli pusto
        if self.state.history:
            last_thought = self.state.history[-1]['assistant'][:100] + "..."

        for i in range(n):
            print(f"{Colors.YELLOW}[Ogniwo {i+1}/{n}]{Colors.RESET}")
            
            prompt = (f"Jesteś modułem DEDUKCJI PRZYCZYNOWEJ. Na podstawie poprzedniej myśli: '{last_thought}', "
                     f"wyciągnij JEDEN bezpośredni skutek lub logiczną konsekwencję. "
                     f"Zacznij od słowa 'Ponieważ...' lub 'W konsekwencji...'. "
                     f"Bądź zwięzły (1-2 zdania).")
            
            payload = {
                "model": self.state.model_name,
                "prompt": prompt,
                "system": "Myśl logicznie, przyczynowo i głęboko. Jesteś częścią 'Łańcucha Myśli' CLATalkie.",
                "stream": False
            }
            
            try:
                resp = requests.post(f"{self.ollama_url}/generate", json=payload, timeout=20)
                if resp.status_code == 200:
                    answer = resp.json().get('response', '').strip()
                    self.stream_print(answer)
                    last_thought = answer
                    
                    # Każde ogniwo to mini-konsolidacja
                    self._update_cognition(f"Causal link {i}", answer)
                    
                    # Dodaj jako ulotny koncept do grafu
                    cid = f"causal_{int(time.time())}_{i}"
                    c = Concept(name=f"Ogniwo {i+1}", concept_id=cid, embedding=self._get_embedding(answer))
                    c.weight = 0.6
                    c.depth = 0.4
                    c.properties = {"type": "causal", "content": answer}
                    self.cla.concept_graph.add_concept(c)
                else:
                    print(f"{Colors.RED}Przerwanie łańcucha: Błąd API.{Colors.RESET}")
                    break
            except Exception as e:
                print(f"{Colors.RED}Przerwanie łańcucha: {e}{Colors.RESET}")
                break
            
            time.sleep(1)
            
        print(f"\n{Colors.GREEN}✓ Łańcuch domknięty. Kognicja zaktualizowana.{Colors.RESET}")
        self._save_state()

    def cmd_evolve(self, epochs: int = 4):
        """Pętla autorefleksji - CLATalkie rozmawia sam ze sobą przez N epok z dynamicznymi pytaniami."""
        print(f"\n{Colors.CYAN}=== PROCES EWOLUCJI KOGNITYWNEJ ({epochs} epok) ==={Colors.RESET}")
        
        if not self.state.ollama_online:
            print(f"{Colors.RED}Ollama offline. Nie można przeprowadzić ewolucji.{Colors.RESET}")
            return
        
        import random
        for epoch in range(epochs):
            # Przełączaj tryby: 0: Introspekcja, 1: Percepcja, 2: Projekcja
            modes = ["internal", "external", "projection"]
            labels = ["INTROSPEKCJA", "PERCEPCJA (Lustro)", "PROJEKCJA (Fantazja)"]
            
            idx = epoch % 3
            mode = modes[idx]
            label = labels[idx]
            
            print(f"\n{Colors.MAGENTA}[Epoka {epoch+1}/{epochs}]{Colors.RESET} {Colors.BOLD}Tryb: {label}{Colors.RESET}")
            
            prompt = self._generate_evolution_prompt(mode)
            print(f"{Colors.YELLOW}Pytanie do siebie:{Colors.RESET} {Colors.ITALIC}{prompt}{Colors.RESET}")
            
            # Generowanie odpowiedzi (pełny proces kognitywny)
            self.generate_response(prompt)
            
            # Jeśli byliśmy w trybie projekcji, zapisz wynik do 'pamięci o przyszłości'
            if mode == "projection" and self.state.history:
                last_answer = self.state.history[-1]['assistant']
                # Wyciągnij proste podsumowanie/zdanie zamiast całości
                summary = last_answer[:100] + "..."
                self.state.projection_scenarios.append(summary)
                if len(self.state.projection_scenarios) > 10: self.state.projection_scenarios.pop(0)

            # Dłuższa przerwa na 'ochłonięcie' (Rate Limit protection)
            time.sleep(3)
            if epoch < epochs - 1:
                print(f"{Colors.DIM}--------------------------------------------------{Colors.RESET}")
        
        print(f"\n{Colors.GREEN}✓ Proces ewolucji zakończony. Stan zaktualizowany.{Colors.RESET}")
        self._save_state()



if __name__ == "__main__":
    talkie = CLATalkie()
    try:
        talkie.main_menu()
    except KeyboardInterrupt:
        talkie._save_state()
        print("\nPrzerwano. Stan zapisany.")
