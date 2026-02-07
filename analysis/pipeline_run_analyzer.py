"""
Pipeline Run Analyzer - Bundelt alle data van een pipeline run voor LLM analyse.

Verzamelt:
- HANDSHAKE_IN/OUT logs van alle stappen
- ERROR/WARNING logs
- Database statistieken (row counts per tabel)
- Gegenereerde bestanden per stap
- Timing informatie
- Data flow verificatie

Output:
- run_summary.md (hoofdrapport, LLM-optimized)
- handshake_flow.json (gestructureerde data)
- errors_warnings.txt (alle problemen)
- database_stats.json (row counts)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Voeg project root toe aan path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import get_cursor
from core.output_manager import ValidationOutputManager
from core.logging_utils import setup_logging

logger = setup_logging("pipeline_run_analyzer")


@dataclass
class HandshakeLog:
    """Gestructureerde handshake log entry."""
    direction: str  # IN of OUT
    step: str
    run_id: str
    table: str  # source (IN) of target (OUT)
    rows: int
    filter: Optional[str] = None
    operation: Optional[str] = None
    timestamp: Optional[str] = None
    log_file: Optional[str] = None


@dataclass
class StepSummary:
    """Samenvatting van één pipeline stap."""
    step_number: int
    step_name: str
    script: str
    handshake_in: List[HandshakeLog]
    handshake_out: List[HandshakeLog]
    errors: List[str]
    warnings: List[str]
    execution_time_sec: Optional[float]
    output_files: List[str]
    db_tables_written: List[str]


class PipelineRunAnalyzer:
    """Analyseert een volledige pipeline run."""
    
    def __init__(self, run_id: str, asset_id: int, dagster_log_path: Optional[str] = None):
        self.run_id = run_id
        self.asset_id = asset_id
        self.base_dir = Path(__file__).parent.parent
        self.log_dir = self.base_dir / "_log"
        self.validation_dir = self.base_dir / "_validation"
        self.dagster_log_path = Path(dagster_log_path) if dagster_log_path else None

    def scan_dagster_log_for_handshakes(self) -> List[HandshakeLog]:
        """
        Scant Dagster STDOUT log voor HANDSHAKE_IN/OUT entries.

        Dagster logt handshakes naar STDOUT/STDERR wanneer scripts draaien via Dagster.
        Deze logs staan niet in _log/*.log maar in de Dagster terminal output.
        """
        if not self.dagster_log_path or not self.dagster_log_path.exists():
            return []

        handshakes = []
        pattern = re.compile(
            r'HANDSHAKE_(IN|OUT) \| '
            r'step=(\S+) \| '
            r'run_id=(\S+) \| '
            r'(?:source|target)=(\S+) \| '
            r'rows=(\d+)'
            r'(?:\s*\|\s*filter=(\S+))?'
            r'(?:\s*\|\s*operation=(\S+))?'
        )

        try:
            with open(self.dagster_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if self.run_id not in line:
                        continue

                    # Skip duplicate INFO log entries (keep only the first occurrence)
                    if 'INFO:core.step_validation:HANDSHAKE' in line:
                        continue

                    match = pattern.search(line)
                    if match:
                        direction, step, run_id, table, rows, filter_clause, operation = match.groups()

                        # Extract timestamp from Dagster log format
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        timestamp = timestamp_match.group(1) if timestamp_match else None

                        handshakes.append(HandshakeLog(
                            direction=direction,
                            step=step,
                            run_id=run_id,
                            table=table,
                            rows=int(rows),
                            filter=filter_clause if filter_clause and filter_clause != 'none' else None,
                            operation=operation,
                            timestamp=timestamp,
                            log_file="dagster_stdout"
                        ))
        except Exception as e:
            logger.warning(f"Error scanning Dagster log {self.dagster_log_path}: {e}")

        return sorted(handshakes, key=lambda x: (x.timestamp or '', x.step))

    def scan_logs_for_handshakes(self) -> List[HandshakeLog]:
        """Scant alle logs (file logs + Dagster STDOUT) voor HANDSHAKE_IN/OUT entries."""
        handshakes = []
        pattern = re.compile(
            r'HANDSHAKE_(IN|OUT) \| '
            r'step=(\S+) \| '
            r'run_id=(\S+) \| '
            r'(?:source|target)=(\S+) \| '
            r'rows=(\d+)'
            r'(?:\s*\|\s*filter=(\S+))?'
            r'(?:\s*\|\s*operation=(\S+))?'
        )

        # 1. Scan _log directory (individuele script runs)
        log_files = list(self.log_dir.glob("*.log"))
        log_files.extend(self.log_dir.glob("archive/*.log"))

        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if self.run_id not in line:
                            continue

                        match = pattern.search(line)
                        if match:
                            direction, step, run_id, table, rows, filter_clause, operation = match.groups()

                            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            timestamp = timestamp_match.group(1) if timestamp_match else None

                            handshakes.append(HandshakeLog(
                                direction=direction,
                                step=step,
                                run_id=run_id,
                                table=table,
                                rows=int(rows),
                                filter=filter_clause if filter_clause and filter_clause != 'none' else None,
                                operation=operation,
                                timestamp=timestamp,
                                log_file=str(log_file.relative_to(self.base_dir))
                            ))
            except Exception as e:
                logger.warning(f"Error scanning {log_file}: {e}")

        # 2. Scan Dagster STDOUT log (indien beschikbaar)
        dagster_handshakes = self.scan_dagster_log_for_handshakes()
        handshakes.extend(dagster_handshakes)

        # 3. Dedupliceer (sommige handshakes komen voor in beide sources)
        unique_handshakes = {}
        for h in handshakes:
            key = (h.direction, h.step, h.table, h.rows)
            if key not in unique_handshakes:
                unique_handshakes[key] = h

        return sorted(unique_handshakes.values(), key=lambda x: (x.timestamp or '', x.step))
    
    def scan_logs_for_errors_warnings(self) -> Dict[str, List[str]]:
        """Scant logs voor ERROR en WARNING entries."""
        issues = {"errors": [], "warnings": []}
        
        log_files = list(self.log_dir.glob("*.log"))
        log_files.extend(self.log_dir.glob("archive/*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if self.run_id not in line:
                            continue
                        
                        if ", ERROR," in line:
                            issues["errors"].append(f"{log_file.name}: {line.strip()}")
                        elif ", WARNING," in line:
                            issues["warnings"].append(f"{log_file.name}: {line.strip()}")
            except Exception as e:
                logger.warning(f"Error scanning {log_file}: {e}")
        
        return issues
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Haalt row counts op voor alle relevante tabellen.
        
        REASON: Universele tabellen (barrier_outcomes) worden niet gefilterd op run_id,
        run-scoped tabellen wel. Dit voorkomt misleidende "0 rows" rapportage.
        """
        # Universele tabellen (deterministic from market data, geen run_id scoping)
        universal_tables = [
            "qbn.barrier_outcomes",
        ]
        
        # Run-scoped tabellen (houden meerdere runs bij)
        run_scoped_tables = [
            "qbn.composite_threshold_config",
            "qbn.signal_weights",
            "qbn.combination_alpha",
            "qbn.event_windows",
            "qbn.position_delta_threshold_config",
            "qbn.cpt_cache",
        ]
        
        stats = {}
        with get_cursor() as cur:
            # Query universal tables zonder run_id filter
            for table in universal_tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE asset_id = %s", (self.asset_id,))
                    count = cur.fetchone()[0]
                    stats[f"{table} (universal)"] = count
                except Exception as e:
                    logger.warning(f"Could not query {table}: {e}")
                    stats[f"{table} (universal)"] = -1
            
            # Query run-scoped tables met run_id filter
            for table in run_scoped_tables:
                try:
                    cur.execute(f"""
                        SELECT COUNT(*) 
                        FROM {table} 
                        WHERE asset_id = %s AND run_id = %s
                    """, (self.asset_id, self.run_id))
                    count = cur.fetchone()[0]
                    stats[f"{table} (run {self.run_id[:8]})"] = count
                except Exception as e:
                    logger.warning(f"Could not query {table}: {e}")
                    stats[f"{table} (run {self.run_id[:8]})"] = -1
        
        return stats
    
    def find_output_files(self) -> List[str]:
        """Vindt alle output bestanden voor deze run_id."""
        output_dirs = []
        for path in self.validation_dir.glob(f"*-{self.run_id}"):
            if path.is_dir():
                output_dirs.append(str(path.relative_to(self.base_dir)))
        
        return output_dirs
    
    def verify_data_flow(self, handshakes: List[HandshakeLog]) -> List[str]:
        """Verifieert dat output van stap N = input van stap N+1."""
        issues = []
        
        # Groepeer handshakes per stap
        by_step = {}
        for h in handshakes:
            if h.step not in by_step:
                by_step[h.step] = {"in": [], "out": []}
            by_step[h.step][h.direction.lower()].append(h)
        
        # Verificatie logica (simplified example)
        # Check: barrier_backfill OUTPUT → materialize_leading_scores INPUT
        if "barrier_backfill" in by_step and "materialize_leading_scores" in by_step:
            out_rows = next((h.rows for h in by_step["barrier_backfill"]["out"] if h.table == "qbn.barrier_outcomes"), 0)
            in_rows = next((h.rows for h in by_step["materialize_leading_scores"]["in"] if h.table == "qbn.barrier_outcomes"), 0)
            
            if in_rows < out_rows:
                issues.append(f"Data flow warning: barrier_backfill wrote {out_rows} rows, but materialize_leading_scores only read {in_rows}")
        
        # Guards die faalden
        for step, data in by_step.items():
            for h in data["in"]:
                if h.rows == 0:
                    issues.append(f"Guard warning: {step} found 0 rows in {h.table} (expected data from upstream)")
        
        return issues
    
    def generate_step_summaries(self, handshakes: List[HandshakeLog]) -> List[StepSummary]:
        """Genereert samenvatting per pipeline stap."""
        # Pipeline stappen definitie (van eerdere analyse)
        pipeline_steps = [
            (1, "composite_threshold_config", "scripts/run_threshold_analysis.py"),
            (2, "barrier_outcomes", "scripts/barrier_backfill.py"),
            (3, "barrier_outcomes_leading", "scripts/materialize_leading_scores.py"),
            (4, "barrier_outcomes_weights", "scripts/compute_barrier_weights.py"),
            (5, "signal_weights", "alpha-analysis/analyze_signal_alpha.py"),
            (6, "combination_alpha", "scripts/run_combination_analysis.py"),
            (7, "event_windows", "scripts/run_event_window_detection.py"),
            (8, "position_delta_threshold_config", "scripts/run_position_delta_threshold_analysis.py"),
            (9, "cpt_cache", "inference/qbn_v3_cpt_generator.py"),
        ]
        
        summaries = []
        for step_num, step_name, script in pipeline_steps:
            # Filter handshakes voor deze stap
            step_handshakes = [h for h in handshakes if h.step.lower().replace("_", "") in step_name.lower().replace("_", "")]
            
            summary = StepSummary(
                step_number=step_num,
                step_name=step_name,
                script=script,
                handshake_in=[h for h in step_handshakes if h.direction == "IN"],
                handshake_out=[h for h in step_handshakes if h.direction == "OUT"],
                errors=[],  # Extracted separately
                warnings=[],  # Extracted separately
                execution_time_sec=None,  # TODO: calculate from timestamps
                output_files=[],  # TODO: find from validation dir
                db_tables_written=[h.table for h in step_handshakes if h.direction == "OUT"]
            )
            summaries.append(summary)
        
        return summaries
    
    def generate_markdown_report(
        self,
        handshakes: List[HandshakeLog],
        issues: Dict[str, List[str]],
        db_stats: Dict[str, int],
        data_flow_issues: List[str],
        step_summaries: List[StepSummary]
    ) -> str:
        """Genereert compact Markdown rapport voor LLM consumptie."""
        report_lines = [
            f"# Pipeline Run Analysis: {self.run_id}",
            f"",
            f"**Asset ID:** {self.asset_id}  ",
            f"**Run ID:** {self.run_id}  ",
            f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            f"- **Total Steps:** {len(step_summaries)}",
            f"- **HANDSHAKE Logs:** {len(handshakes)} ({sum(1 for h in handshakes if h.direction == 'IN')} IN, {sum(1 for h in handshakes if h.direction == 'OUT')} OUT)",
            f"- **Errors:** {len(issues['errors'])}",
            f"- **Warnings:** {len(issues['warnings'])}",
            f"- **Data Flow Issues:** {len(data_flow_issues)}",
            f"",
        ]
        
        # Data flow graph (simplified)
        if handshakes:
            report_lines.extend([
                f"## Data Flow",
                f"",
                f"```mermaid",
                f"graph TD",
            ])
            
            # Generate mermaid nodes
            for summary in step_summaries:
                node_id = summary.step_name.replace("_", "")
                rows_in = sum(h.rows for h in summary.handshake_in)
                rows_out = sum(h.rows for h in summary.handshake_out)
                report_lines.append(f'    {node_id}["{summary.step_name} IN:{rows_in} OUT:{rows_out}"]')
            
            # Add connections (simplified - based on pipeline order)
            for i in range(len(step_summaries) - 1):
                curr = step_summaries[i].step_name.replace("_", "")
                next_step = step_summaries[i+1].step_name.replace("_", "")
                report_lines.append(f"    {curr} --> {next_step}")
            
            report_lines.extend([
                f"```",
                f"",
            ])
        
        # Database Statistics
        report_lines.extend([
            f"## Database Statistics",
            f"",
            f"| Table | Row Count |",
            f"|-------|-----------|",
        ])
        for table, count in db_stats.items():
            status = "✓" if count > 0 else ("✗" if count == 0 else "?")
            report_lines.append(f"| {status} {table} | {count:,} |")
        report_lines.append(f"")
        
        # Step Details
        report_lines.extend([
            f"## Step Details",
            f"",
        ])
        
        for summary in step_summaries:
            report_lines.extend([
                f"### {summary.step_number}. {summary.step_name}",
                f"",
                f"**Script:** `{summary.script}`  ",
            ])
            
            if summary.handshake_in:
                report_lines.append(f"**Input:**")
                for h in summary.handshake_in:
                    filter_str = f" (filter: {h.filter})" if h.filter else ""
                    report_lines.append(f"- {h.table}: {h.rows:,} rows{filter_str}")
            
            if summary.handshake_out:
                report_lines.append(f"**Output:**")
                for h in summary.handshake_out:
                    op_str = f" ({h.operation})" if h.operation else ""
                    report_lines.append(f"- {h.table}: {h.rows:,} rows{op_str}")
            
            report_lines.append(f"")
        
        # Issues
        if issues["errors"]:
            report_lines.extend([
                f"## Errors",
                f"",
                f"```",
            ])
            for error in issues["errors"][:20]:  # Limit to first 20
                report_lines.append(error)
            if len(issues["errors"]) > 20:
                report_lines.append(f"... and {len(issues['errors']) - 20} more errors")
            report_lines.extend([
                f"```",
                f"",
            ])
        
        if issues["warnings"]:
            report_lines.extend([
                f"## Warnings",
                f"",
                f"```",
            ])
            for warning in issues["warnings"][:20]:  # Limit to first 20
                report_lines.append(warning)
            if len(issues["warnings"]) > 20:
                report_lines.append(f"... and {len(issues['warnings']) - 20} more warnings")
            report_lines.extend([
                f"```",
                f"",
            ])
        
        if data_flow_issues:
            report_lines.extend([
                f"## Data Flow Issues",
                f"",
            ])
            for issue in data_flow_issues:
                report_lines.append(f"- {issue}")
            report_lines.append(f"")
        
        return "\n".join(report_lines)
    
    def analyze(self) -> Path:
        """Voert volledige analyse uit en genereert rapporten."""
        logger.info(f"Analyzing pipeline run: {self.run_id} for asset {self.asset_id}")
        
        # 1. Verzamel data
        logger.info("Scanning logs for HANDSHAKE entries...")
        handshakes = self.scan_logs_for_handshakes()
        logger.info(f"Found {len(handshakes)} HANDSHAKE logs")
        
        logger.info("Scanning logs for errors and warnings...")
        issues = self.scan_logs_for_errors_warnings()
        logger.info(f"Found {len(issues['errors'])} errors, {len(issues['warnings'])} warnings")
        
        logger.info("Querying database statistics...")
        db_stats = self.get_database_stats()
        
        logger.info("Finding output files...")
        output_files = self.find_output_files()
        logger.info(f"Found {len(output_files)} output directories")
        
        logger.info("Verifying data flow...")
        data_flow_issues = self.verify_data_flow(handshakes)
        
        logger.info("Generating step summaries...")
        step_summaries = self.generate_step_summaries(handshakes)
        
        # 2. Genereer output
        output_mgr = ValidationOutputManager()
        output_dir = output_mgr.create_output_dir(
            script_name="pipeline_analysis",
            asset_id=self.asset_id,
            run_id=self.run_id
        )
        
        # Markdown report (hoofdrapport)
        logger.info("Generating markdown report...")
        markdown = self.generate_markdown_report(
            handshakes, issues, db_stats, data_flow_issues, step_summaries
        )
        report_path = output_dir / "run_summary.md"
        report_path.write_text(markdown, encoding='utf-8')
        logger.info(f"Markdown report: {report_path}")
        
        # JSON data (gestructureerd)
        handshake_path = output_dir / "handshake_flow.json"
        handshake_path.write_text(
            json.dumps([asdict(h) for h in handshakes], indent=2),
            encoding='utf-8'
        )
        
        db_stats_path = output_dir / "database_stats.json"
        db_stats_path.write_text(json.dumps(db_stats, indent=2), encoding='utf-8')
        
        # Errors/warnings (plain text)
        issues_path = output_dir / "errors_warnings.txt"
        issues_path.write_text(
            "ERRORS:\n" + "\n".join(issues["errors"]) + "\n\n" +
            "WARNINGS:\n" + "\n".join(issues["warnings"]),
            encoding='utf-8'
        )
        
        logger.info(f"✅ Analysis complete. Output: {output_dir}")
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Analyseert pipeline run resultaten en genereert LLM-optimized rapporten"
    )
    parser.add_argument("--run-id", type=str, help="Specifieke run_id om te analyseren")
    parser.add_argument("--asset-id", type=int, required=True, help="Asset ID")
    parser.add_argument(
        "--dagster-log",
        type=str,
        help="Path to Dagster terminal/STDOUT log file (for handshake detection in Dagster runs)",
    )
    parser.add_argument("--latest", action="store_true", help="Analyseer laatste run voor dit asset")

    args = parser.parse_args()
    
    if not args.run_id and not args.latest:
        parser.error("Specify either --run-id or --latest")
    
    if args.latest:
        # TODO: Query database voor laatste run_id van dit asset
        raise NotImplementedError("--latest not yet implemented")
    
    analyzer = PipelineRunAnalyzer(
        run_id=args.run_id,
        asset_id=args.asset_id,
        dagster_log_path=args.dagster_log,
    )
    output_dir = analyzer.analyze()
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Report: {output_dir / 'run_summary.md'}")
    print(f"📊 Data: {output_dir / 'handshake_flow.json'}")
    print(f"⚠️  Issues: {output_dir / 'errors_warnings.txt'}")


if __name__ == "__main__":
    main()
