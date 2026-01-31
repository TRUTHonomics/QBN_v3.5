#!/usr/bin/env python3
"""
Docker Menu Script voor QBN (QuantBayes Nexus) v3
=================================================
Detecteert container rol en laadt het juiste submenu.

Container Rollen:
- inference:  Real-time predictions (QBN_v3.1)
- training:   CPT generation, data prep (QBN_v3.1_Training)
- validation: Quality assurance (QBN_v3.1_Validation)
"""

import sys
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Detecteer container rol
ROLE = os.getenv('QBN_ROLE', 'inference').lower()


def main():
    """Start het juiste menu op basis van QBN_ROLE"""
    
    print(f"\n🚀 QBN v3 - Starting menu for role: {ROLE.upper()}\n")
    
    if ROLE == 'inference':
        from menus.inference_menu import run
        run()
        
    elif ROLE == 'training':
        from menus.training_menu import run
        run()
        
    elif ROLE == 'validation':
        from menus.validation_menu import run
        run()
        
    else:
        # Fallback: toon keuzemenu
        print("⚠️  Onbekende rol. Selecteer handmatig:")
        print("  1. Inference Menu")
        print("  2. Training Menu")
        print("  3. Validation Menu")
        print("  0. Exit")
        
        choice = input("\nKeuze: ").strip()
        
        if choice == '1':
            from menus.inference_menu import run
            run()
        elif choice == '2':
            from menus.training_menu import run
            run()
        elif choice == '3':
            from menus.validation_menu import run
            run()
        else:
            print("\n👋 Tot ziens!\n")
            sys.exit(0)


if __name__ == '__main__':
    main()
