#!/usr/bin/env python3
"""
Grid Search Parameter Configurator.

Interactief menu voor het selecteren en configureren van Grid Search parameters.
Ondersteunt preset opslag en laden uit de database.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_cursor


class GridSearchConfigurator:
    """
    Interactieve configurator voor Grid Search parameters.
    """
    
    # Definitie van alle beschikbare parameters (v3.4 aligned)
    PARAMETERS = {
        'ENTRY-SIDE NODES (Signal Filters)': {
            'entry_strength_threshold': {
                'name': 'Entry Signal Strength (Trade_Hypothesis)',
                'type': 'categorical',
                'values': ['weak', 'strong'],
                'default_values': ['weak', 'strong'],
                'description': 'Minimale Trade_Hypothesis sterkte (trade BEIDE long en short)'
            },
            'regime_filter': {
                'name': 'Market Regime Filter (HTF_Regime)',
                'type': 'categorical',
                'values': ['full_bearish', 'bearish_transition', 'macro_ranging', 'bullish_transition', 'full_bullish'],
                'default_values': [],
                'description': 'Toegestane markt regimes (leeg = alle)'
            }
        },
        'POSITION-SIDE NODES (Trade Management)': {
            'use_qbn_exit_timing': {
                'name': 'Exit Signal (Exit_Timing)',
                'type': 'boolean',
                'values': [True, False],
                'description': 'Exit als Exit_Timing = exit_now'
            },
            'exit_on_momentum_reversal': {
                'name': 'Momentum Reversal Exit (Momentum_Prediction)',
                'type': 'boolean',
                'values': [True, False],
                'description': 'Exit als momentum tegen positie draait (bearish op long, bullish op short)'
            },
            'volatility_position_sizing': {
                'name': 'Volatility Sizing (Volatility_Regime)',
                'type': 'boolean',
                'values': [True, False],
                'description': 'Pas positiegrootte aan: low_vol=1.2x, normal=1.0x, high_vol=0.5x'
            },
            'use_position_prediction_exit': {
                'name': 'Prediction Exit (Position_Prediction)',
                'type': 'boolean',
                'values': [True, False],
                'description': 'Exit als Position_Prediction = stoploss_hit of timeout'
            }
        },
        'RISK & EXECUTION (Non-Node Parameters)': {
            'stop_loss_atr_mult': {
                'name': 'Stop Loss Distance (ATR)',
                'type': 'numeric',
                'min': 0.5,
                'max': 3.0,
                'step': 0.5,
                'description': 'Stop loss afstand in ATR eenheden'
            },
            'take_profit_atr_mult': {
                'name': 'Take Profit Distance (ATR)',
                'type': 'numeric',
                'min': 1.0,
                'max': 3.0,
                'step': 0.5,
                'description': 'Take profit afstand in ATR eenheden'
            },
            'max_holding_time_hours': {
                'name': 'Max Holding Time',
                'type': 'numeric',
                'min': 4,
                'max': 28.0,
                'step': 8.0,
                'description': 'Maximale trade duur (None = geen limiet)'
            },
            'leverage': {
                'name': 'Leverage',
                'type': 'categorical',
                'values': [1.0, 10.0],
                'default_values': [1.0, 10.0],
                'description': 'Hefboomfactor (1x = geen leverage)'
            },
            'trailing_stop_enabled': {
                'name': 'Trailing Stop',
                'type': 'boolean',
                'values': [True, False],
                'description': 'Activeer trailing stop loss'
            },
            'trailing_activation_pct': {
                'name': 'Trailing Activation %',
                'type': 'numeric',
                'min': 0.5,
                'max': 3.5,
                'step': 1.0,
                'description': 'Winst % voor trailing stop activatie'
            },
            'trailing_stop_pct': {
                'name': 'Trailing Stop Distance %',
                'type': 'numeric',
                'min': 20.0,
                'max': 80.0,
                'step': 20.0,
                'description': 'Percentage van winst dat opgegeven wordt'
            }
        }
    }
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.current_preset = "unsaved"
        self.console = Console()
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialiseer met enabled config voor alle parameters (standaard alles aan)."""
        for category in self.PARAMETERS.values():
            for param_name, param_def in category.items():
                # Check of parameter bruikbare default waarden heeft
                has_values = True
                if param_def['type'] == 'categorical':
                    default_values = param_def.get('default_values', [])
                    has_values = len(default_values) > 0
                
                self.config[param_name] = {
                    'enabled': has_values,  # Alleen enabled als er default waarden zijn
                    'type': param_def['type'],
                }
                
                if param_def['type'] == 'numeric':
                    self.config[param_name].update({
                        'min': param_def['min'],
                        'max': param_def['max'],
                        'step': param_def['step']
                    })
                elif param_def['type'] == 'categorical':
                    self.config[param_name]['values'] = param_def.get('default_values', [])
                elif param_def['type'] == 'boolean':
                    self.config[param_name]['values'] = [True, False]
    
    def show_menu(self):
        """Toon het hoofdmenu met huidige configuratie."""
        self.console.clear()
        
        # Header
        self.console.print(Panel.fit(
            "[bold cyan]🔍 GRID SEARCH PARAMETER CONFIGURATOR[/bold cyan]",
            subtitle=f"[dim]Preset: {self.current_preset}[/dim]",
            border_style="cyan"
        ))
        
        param_num = 1
        param_map = {}
        
        # Create main table
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta", expand=True)
        table.add_column("ID", style="dim", width=4, justify="right")
        table.add_column("Status", width=6, justify="center")
        table.add_column("Parameter", style="cyan")
        table.add_column("Configuration", style="green")
        
        for category_name, category in self.PARAMETERS.items():
            # Add section row
            table.add_section()
            table.add_row("", "", f"[bold yellow]{category_name}[/bold yellow]", "")
            
            for param_name, param_def in category.items():
                enabled = self.config[param_name]['enabled']
                status_icon = "✅" if enabled else "❌"
                status_style = "bold green" if enabled else "dim red"
                
                # Format waarden preview
                if enabled:
                    if param_def['type'] == 'numeric':
                        cfg = self.config[param_name]
                        values_preview = self._format_numeric_range(cfg['min'], cfg['max'], cfg['step'])
                    elif param_def['type'] == 'categorical':
                        values = self.config[param_name]['values']
                        values_preview = ", ".join([str(v) if v is not None else 'null' for v in values])
                    elif param_def['type'] == 'boolean':
                        values_preview = "true, false"
                else:
                    values_preview = "[dim](disabled)[/dim]"
                
                table.add_row(
                    str(param_num),
                    status_icon,
                    param_def['name'],
                    values_preview
                )
                param_map[param_num] = param_name
                param_num += 1
        
        self.console.print(table)
        
        # Bereken totaal combinaties
        total_combinations = self.calculate_combinations()
        self.console.print(f"\n[bold]Totaal combinaties:[/bold] [cyan]{total_combinations:,}[/cyan]")
        
        # Quick commands help
        self.console.print(Panel(
            "[bold]Quick Commands:[/bold]\n"
            "  [green]t 1 3 5[/green] : Toggle parameters 1, 3, 5\n"
            "  [green]e 7[/green]     : Edit parameter 7\n"
            "  [green]r[/green]       : Run Grid Search\n"
            "  [green]p[/green]       : Save Preset\n"
            "  [green]l[/green]       : Load Preset\n"
            "  [green]d[/green]       : Load Default\n"
            "  [green]q[/green]       : Quit",
            title="Actions",
            border_style="dim"
        ))
        
        return param_map
    
    def _format_numeric_range(self, min_val: float, max_val: float, step: float) -> str:
        """Format numeriek bereik als preview string."""
        import numpy as np
        values = np.arange(min_val, max_val + step/2, step)
        value_str = ", ".join([f"{v:.1f}" if v % 1 else f"{int(v)}" for v in values])
        if len(values) <= 5:
            return f"{value_str}"
        else:
            return f"{min_val:.1f} - {max_val:.1f} (stap {step:.1f}) [{len(values)} waarden]"
    
    def edit_parameter(self, param_name: str):
        """Bewerk een specifieke parameter."""
        # Vind parameter definitie
        param_def = None
        for category in self.PARAMETERS.values():
            if param_name in category:
                param_def = category[param_name]
                break
        
        if not param_def:
            self.console.print(f"[red]❌ Parameter {param_name} niet gevonden[/red]")
            return
        
        self.console.print(Panel(
            f"[bold]{param_def['name']}[/bold]\n{param_def['description']}",
            title="Edit Parameter",
            border_style="blue"
        ))
        
        # Toggle enabled
        current_enabled = self.config[param_name]['enabled']
        enable_choice = self.console.input(f"Parameter in grid search opnemen? [{'Y' if current_enabled else 'N'}]: ").strip().upper()
        
        if enable_choice == '':
            enable_choice = 'Y' if current_enabled else 'N'
        
        self.config[param_name]['enabled'] = (enable_choice == 'Y')
        
        if not self.config[param_name]['enabled']:
            self.console.print("[yellow]✅ Parameter uitgeschakeld[/yellow]")
            return
        
        # Configureer waarden
        if param_def['type'] == 'numeric':
            self._edit_numeric_parameter(param_name, param_def)
        elif param_def['type'] == 'categorical':
            self._edit_categorical_parameter(param_name, param_def)
        elif param_def['type'] == 'boolean':
            # Boolean is altijd [true, false]
            self.console.print("[green]✅ Boolean parameter: zal [true, false] testen[/green]")
    
    def _edit_numeric_parameter(self, param_name: str, param_def: Dict):
        """Bewerk numerieke parameter (min/max/step)."""
        current = self.config[param_name]
        
        self.console.print(f"\nHuidige waarden: min={current.get('min', param_def['min'])}, "
              f"max={current.get('max', param_def['max'])}, "
              f"step={current.get('step', param_def['step'])}")
        
        try:
            min_val_input = self.console.input(f"Min waarde [{param_def['min']}]: ")
            min_val = float(min_val_input) if min_val_input else param_def['min']
            
            max_val_input = self.console.input(f"Max waarde [{param_def['max']}]: ")
            max_val = float(max_val_input) if max_val_input else param_def['max']
            
            step_input = self.console.input(f"Stapgrootte [{param_def['step']}]: ")
            step = float(step_input) if step_input else param_def['step']
            
            if min_val >= max_val:
                self.console.print("[red]❌ Min moet kleiner zijn dan max[/red]")
                return
            
            if step <= 0:
                self.console.print("[red]❌ Stapgrootte moet positief zijn[/red]")
                return
            
            self.config[param_name]['min'] = min_val
            self.config[param_name]['max'] = max_val
            self.config[param_name]['step'] = step
            
            import numpy as np
            values = np.arange(min_val, max_val + step/2, step)
            self.console.print(f"[green]✅ Zal {len(values)} waarden testen: {self._format_numeric_range(min_val, max_val, step)}[/green]")
            
        except ValueError as e:
            self.console.print(f"[red]❌ Ongeldige invoer: {e}[/red]")
    
    def _edit_categorical_parameter(self, param_name: str, param_def: Dict):
        """Bewerk categorische parameter (checkbox lijst)."""
        self.console.print("\nSelecteer waarden om te testen (spatie-gescheiden nummers):")
        
        available_values = param_def['values']
        for i, val in enumerate(available_values, 1):
            display_val = str(val) if val is not None else 'null'
            self.console.print(f"  [bold]{i}.[/bold] {display_val}")
        
        current_values = self.config[param_name].get('values', param_def.get('default_values', []))
        current_indices = [i+1 for i, v in enumerate(available_values) if v in current_values]
        
        selection = self.console.input(f"Nummers [{' '.join(map(str, current_indices))}]: ").strip()
        
        if not selection:
            # Gebruik defaults
            self.config[param_name]['values'] = current_values
            self.console.print(f"[green]✅ Behouden: {[str(v) if v is not None else 'null' for v in current_values]}[/green]")
            return
        
        try:
            indices = [int(x) for x in selection.split()]
            selected_values = [available_values[i-1] for i in indices if 1 <= i <= len(available_values)]
            
            if not selected_values:
                self.console.print("[red]❌ Geen geldige waarden geselecteerd[/red]")
                return
            
            self.config[param_name]['values'] = selected_values
            self.console.print(f"[green]✅ Geselecteerd: {[str(v) if v is not None else 'null' for v in selected_values]}[/green]")
            
        except (ValueError, IndexError) as e:
            self.console.print(f"[red]❌ Ongeldige selectie: {e}[/red]")
    
    def calculate_combinations(self) -> int:
        """Bereken totaal aantal te testen combinaties."""
        import numpy as np
        total = 1
        
        for param_name, cfg in self.config.items():
            if not cfg['enabled']:
                continue
            
            if cfg['type'] == 'numeric':
                count = len(np.arange(cfg['min'], cfg['max'] + cfg['step']/2, cfg['step']))
            elif cfg['type'] in ['categorical', 'boolean']:
                count = len(cfg.get('values', []))
            else:
                count = 1
            
            total *= count
        
        return total
    
    def generate_grid_json(self) -> Dict:
        """Converteer configuratie naar Grid Search JSON formaat."""
        import numpy as np
        grid = {}
        
        for param_name, cfg in self.config.items():
            if not cfg['enabled']:
                continue
            
            if cfg['type'] == 'numeric':
                values = np.arange(cfg['min'], cfg['max'] + cfg['step']/2, cfg['step'])
                grid[param_name] = [float(v) for v in values]
            elif cfg['type'] in ['categorical', 'boolean']:
                grid[param_name] = cfg.get('values', [])
        
        return grid
    
    def save_preset(self):
        """Sla huidige configuratie op als preset."""
        self.console.print("\n[bold]💾 PRESET OPSLAAN[/bold]")
        
        name = self.console.input("Preset naam: ").strip()
        if not name:
            self.console.print("[red]❌ Naam is verplicht[/red]")
            return
        
        description = self.console.input("Beschrijving (optioneel): ").strip()
        
        try:
            with get_cursor(commit=True) as cur:
                cur.execute("""
                    INSERT INTO qbn.grid_search_presets (preset_name, description, parameters)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (preset_name) DO UPDATE
                    SET description = EXCLUDED.description,
                        parameters = EXCLUDED.parameters,
                        updated_at = NOW()
                """, (name, description, json.dumps(self.config)))
            
            self.current_preset = name
            self.console.print(f"[green]✅ Preset '{name}' opgeslagen[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Fout bij opslaan: {e}[/red]")
    
    def load_preset(self, preset_name: Optional[str] = None):
        """Laad een preset uit de database."""
        self.console.print("\n[bold]📂 PRESET LADEN[/bold]")
        
        try:
            with get_cursor() as cur:
                if preset_name is None:
                    # Toon lijst
                    cur.execute("""
                        SELECT preset_name, description, created_at
                        FROM qbn.grid_search_presets
                        ORDER BY updated_at DESC
                    """)
                    presets = cur.fetchall()
                    
                    if not presets:
                        self.console.print("[red]❌ Geen presets gevonden[/red]")
                        return
                    
                    self.console.print("\nBeschikbare presets:")
                    for i, (name, desc, created) in enumerate(presets, 1):
                        self.console.print(f"  [bold]{i}.[/bold] {name:<30} - {desc or '(geen beschrijving)'}")
                    
                    choice = self.console.input("\nSelecteer nummer (0 = annuleren): ").strip()
                    if not choice or choice == '0':
                        return
                    
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(presets):
                            preset_name = presets[idx][0]
                        else:
                            self.console.print("[red]❌ Ongeldige keuze[/red]")
                            return
                    except ValueError:
                        self.console.print("[red]❌ Ongeldige invoer[/red]")
                        return
                
                # Laad preset
                cur.execute("""
                    SELECT parameters
                    FROM qbn.grid_search_presets
                    WHERE preset_name = %s
                """, (preset_name,))
                
                row = cur.fetchone()
                if not row:
                    self.console.print(f"[red]❌ Preset '{preset_name}' niet gevonden[/red]")
                    return
                
                self.config = row[0]
                self.current_preset = preset_name
                self.console.print(f"[green]✅ Preset '{preset_name}' geladen[/green]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Fout bij laden: {e}[/red]")
    
    def run_grid_search(self):
        """Start Grid Search met huidige configuratie."""
        self.console.print("\n[bold]🚀 START GRID SEARCH[/bold]")
        
        # Controleer of er parameters enabled zijn
        enabled_params = [k for k, v in self.config.items() if v.get('enabled')]
        if not enabled_params:
            self.console.print("[red]❌ Geen parameters geselecteerd. Selecteer eerst parameters.[/red]")
            self.console.input("\nDruk op Enter om terug te gaan...")
            return
        
        total_combinations = self.calculate_combinations()
        
        # Vraag bevestiging
        self.console.print(f"\n📊 Configuratie:")
        self.console.print(f"   Enabled parameters: {len(enabled_params)}")
        self.console.print(f"   Totaal combinaties: {total_combinations:,}")
        
        # Vraag Grid Search parameters
        self.console.print("\n📅 Backtest periode:")
        asset_id = self.console.input("Asset ID [1]: ").strip() or "1"
        
        # Default to previous 3 months
        from datetime import timedelta
        today = datetime.now()
        three_months_ago = today - timedelta(days=90)
        
        default_start = three_months_ago.strftime("%Y-%m-%d")
        default_end = today.strftime("%Y-%m-%d")
        
        start_date = self.console.input(f"Start datum (YYYY-MM-DD) [{default_start}]: ").strip() or default_start
        end_date = self.console.input(f"Eind datum (YYYY-MM-DD) [{default_end}]: ").strip() or default_end
        train_window = self.console.input("Training window (dagen) [365]: ").strip() or "365"
        step_days = self.console.input("Test step (dagen) [7]: ").strip() or "7"
        
        confirm = self.console.input("\n⚠️  Doorgaan met Grid Search? [Y/n]: ").strip().lower()
        if confirm == 'n':
            self.console.print("[yellow]Geannuleerd[/yellow]")
            return
        
        # Genereer grid JSON
        grid_json = self.generate_grid_json()
        
        # Roep run_strategy_finder.py aan
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'scripts' / 'run_strategy_finder.py'),
            '--asset-id', asset_id,
            '--start', start_date,
            '--end', end_date,
            '--window', train_window,
            '--step', step_days,
            '--grid', json.dumps(grid_json)
        ]
        
        self.console.print(f"\n🔄 Start Grid Search met {total_combinations:,} combinaties...")
        self.console.print(f"   Command: {' '.join(cmd)}\n")
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        
        self.console.print("\n[green]✅ Grid Search voltooid[/green]")
        self.console.input("\nDruk op Enter om terug te gaan...")
    
    def run(self):
        """Main loop."""
        while True:
            param_map = self.show_menu()
            user_input = self.console.input("\n[bold]Commando:[/bold] ").strip()
            
            if not user_input:
                continue
                
            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:]
            
            if cmd == 'q':
                break
            elif cmd == 'r':
                self.run_grid_search()
            elif cmd == 'p':
                self.save_preset()
                self.console.input("\nDruk op Enter om door te gaan...")
            elif cmd == 'l':
                self.load_preset()
                self.console.input("\nDruk op Enter om door te gaan...")
            elif cmd == 'd':
                self.load_preset('default_conservative')
                self.console.input("\nDruk op Enter om door te gaan...")
            elif cmd == 'e':
                if not args:
                    self.console.print("[red]❌ Geef een parameter nummer op (bijv. 'e 7')[/red]")
                    self.console.input("\nDruk op Enter...")
                    continue
                try:
                    num = int(args[0])
                    if num in param_map:
                        self.edit_parameter(param_map[num])
                    else:
                        self.console.print("[red]❌ Ongeldig nummer[/red]")
                        self.console.input("\nDruk op Enter...")
                except ValueError:
                    self.console.print("[red]❌ Ongeldige invoer[/red]")
                    self.console.input("\nDruk op Enter...")
            elif cmd == 't':
                if not args:
                    self.console.print("[red]❌ Geef parameter nummers op (bijv. 't 1 3')[/red]")
                    self.console.input("\nDruk op Enter...")
                    continue
                try:
                    nums = [int(x) for x in args]
                    for num in nums:
                        if num in param_map:
                            param_name = param_map[num]
                            self.config[param_name]['enabled'] = not self.config[param_name]['enabled']
                            status = "aan" if self.config[param_name]['enabled'] else "uit"
                            self.console.print(f"✅ Parameter {num} ({param_name}) gezet op {status}")
                        else:
                            self.console.print(f"[red]❌ Nummer {num} niet gevonden[/red]")
                    # No wait needed, just refresh menu
                except ValueError:
                    self.console.print("[red]❌ Ongeldige invoer[/red]")
                    self.console.input("\nDruk op Enter...")
            else:
                self.console.print("[red]❌ Ongeldig commando[/red]")
                self.console.input("\nDruk op Enter...")


def main():
    """Entry point."""
    try:
        configurator = GridSearchConfigurator()
        configurator.run()
        print("\n👋 Tot ziens!\n")
    except KeyboardInterrupt:
        print("\n👋 Afgebroken.\n")


if __name__ == '__main__':
    main()
