import sys
import os
import shlex
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Mocking the Environment
class Colors:
    CYAN = ""
    YELLOW = ""
    GREEN = ""
    RED = ""
    MAGENTA = ""
    RESET = ""

# Simulate GlobalState from clatalkie.py v6.4
@dataclass
class GlobalState:
    v_t: float = 0.5
    f_c: float = 0.0
    s_grounding: float = 0.9
    catharsis_active: bool = False
    active_file_context: Dict[str, str] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)

def test_path_parsing(arg_str):
    # Mimic cmd_scan logic from clatalkie.py
    learn_mode = "--learn" in arg_str
    cleaned_args = arg_str.replace("--learn", "").strip()
    
    if (cleaned_args.startswith('"') and cleaned_args.endswith('"')) or \
       (cleaned_args.startswith("'") and cleaned_args.endswith("'")):
        path = cleaned_args[1:-1]
    elif cleaned_args.startswith('"'):
        end_idx = cleaned_args.find('"', 1)
        path = cleaned_args[1:end_idx] if end_idx != -1 else cleaned_args[1:]
    else:
        try:
            parts = shlex.split(arg_str, posix=(sys.platform != 'win32'))
            if parts:
                path = parts[0].strip('"\'')
            else: path = cleaned_args
        except:
            path = cleaned_args
    return path

def test_phi_logic(v_t, f_c):
    phi = 0.618
    # Fibonacci Drift logic from _update_cognition
    v_t += (phi - v_t) * 0.05
    f_c += ((1 - phi) - f_c) * 0.05
    return v_t, f_c

def test_katharsis_trigger(f_c):
    return f_c > 0.85

def run_tests():
    print("--- ROZPOCZYNAM TESTY RELIABILITY ADS v6.4 ---")
    
    # 1. Test Path Parsing (The Windows Space Problem)
    test_paths = [
        ('C:\\Users\\Endorfinka\\Desktop\\Cytaty i aforyzmy\\file.txt', 'C:\\Users\\Endorfinka\\Desktop\\Cytaty i aforyzmy\\file.txt'),
        ('"C:\\Users\\Endorfinka\\Desktop\\Cytaty i aforyzmy\\file.txt"', 'C:\\Users\\Endorfinka\\Desktop\\Cytaty i aforyzmy\\file.txt'),
        ('"C:\\Proper Path.txt" --learn', 'C:\\Proper Path.txt')
    ]
    
    for inp, expected in test_paths:
        result = test_path_parsing(inp)
        if result == expected:
            print(f"[OK] Path Parsing: '{inp}' -> '{result}'")
        else:
            print(f"[FAIL] Path Parsing: '{inp}' -> Got '{result}', Expected '{expected}'")

    # 2. Test PHI Drift (Homeostasis)
    v, f = 0.5, 0.8
    v_new, f_new = test_phi_logic(v, f)
    if v_new > v and f_new < f:
        print(f"[OK] Phi Drift: V moved towards 0.618 ({v_new:.3f}), F moved towards 0.382 ({f_new:.3f})")
    else:
        print(f"[FAIL] Phi Drift: Params did not move correctly.")

    # 3. Test Katharsis Trigger
    if test_katharsis_trigger(0.9) == True and test_katharsis_trigger(0.5) == False:
        print("[OK] Katharsis Trigger logic correct.")
    else:
        print("[FAIL] Katharsis Trigger logic faulty.")

    # 4. Syntax Check of main file
    try:
        import ast
        with open('clatalkie.py', 'r', encoding='utf-8') as f:
            ast_tree = ast.parse(f.read())
        print("[OK] Syntax check: 'clatalkie.py' is syntactically valid.")
    except Exception as e:
        print(f"[CRITICAL] Syntax check failed: {e}")

if __name__ == "__main__":
    run_tests()
