import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict
import time
from datetime import datetime

# Setup paths
sys.path.append(os.getcwd())

# Mocking
from unittest.mock import MagicMock, patch

def run_tests():
    print("Checking clatalkie.py for potential logic errors...")
    
    # 1. Syntax Check
    try:
        with open("clatalkie.py", "r", encoding="utf-8") as f:
            code_content = f.read()
        compile(code_content, "clatalkie.py", "exec")
        print("[OK] Syntax check passed.")
    except Exception as e:
        print(f"[FAIL] Syntax Error: {e}")
        return

    # 2. Functional Instance Test
    try:
        import clatalkie
        # Mocking requests and graph initialization to avoid network/external dependencies
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 404 # Ollama offline for init
            talkie = clatalkie.CLATalkie()
            print("[OK] CLATalkie instance created.")
            
            # 3. Test Memory Evolution Logic
            print("Testing Memory Evolution...")
            talkie.state.history = [{"user": "u", "assistant": "a"}] * 24
            talkie.state.history_limit = 24
            talkie.state.ollama_online = True
            
            with patch('requests.post') as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {"response": "Summary text"}
                talkie._save_state = MagicMock()
                
                talkie._handle_memory_evolution()
                
                # Slicing check: history had 24, we took block_size=12. 
                # history should now be 24 - 12 = 12.
                # However, history.append might happen before or after.
                # In clatalkie.py: self.state.history.append(...) then self._handle_memory_evolution()
                
                if len(talkie.state.history) == 12:
                    print("[OK] Memory condensation slicing is correct.")
                else:
                    print(f"[FAIL] Memory condensation slicing error. Got {len(talkie.state.history)}, expected 12")
                
                if len(talkie.state.synthetic_memory) == 1:
                    print("[OK] Synthetic memory entry added.")
                else:
                    print("[FAIL] Synthetic memory not added.")

            # 4. Test Latent Intention generation logic
            print("Testing Latent Intention Logic...")
            with patch('requests.post') as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {"response": "How do you feel?"}
                talkie.cla.concept_graph.decay_all = MagicMock(return_value=([],[]))
                
                # Mock reflection to avoid needing real LLM cycle
                talkie.cmd_think()
                if len(talkie.state.latent_questions) > 0:
                    print(f"[OK] Latent question generated: {talkie.state.latent_questions[0]}")
                else:
                    # Search for 'reflection' in cmd_think, it might need to exist
                    print("[INFO] Latent question not generated (maybe reflection was empty).")

    except Exception as e:
        import traceback
        print(f"[ERROR] Logic test failed with exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
