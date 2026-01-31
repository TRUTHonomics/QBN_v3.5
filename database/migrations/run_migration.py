#!/usr/bin/env python3
"""
Migration Script: Phase 1.1 Database Schema Migration
Purpose: Execute all schema migrations in correct sequence with validation
Date: 2025-12-10
Priority: CRITICAL
Usage: python run_migration.py [--dry-run]
"""

import sys
import os
import argparse
import psycopg2
from datetime import datetime
from pathlib import Path
import time


class Colors:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class MigrationRunner:
    """Database migration runner"""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.script_dir = Path(__file__).parent
        self.log_dir = self.script_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"migration_1.1_{timestamp}.log"
        
        # Database connection parameters
        self.db_params = {
            'host': '127.0.0.1',
            'port': 15432,
            'database': 'KFLhyper',
            'user': 'pipeline',
            'password': 'pipeline123'
        }
        
        # Migration scripts in execution order
        self.migration_scripts = [
            "003_qbn_v2_outcome_columns.sql",
            "003_qbn_v2_signal_classification.sql",
            "003_qbn_v2_indexes.sql",
            "004_signals_current_expansion.sql",
            "004_mtf_cache_expansion.sql"
        ]
        
        self.conn = None
        
    def log(self, level, message):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        color_map = {
            'INFO': Colors.BLUE,
            'SUCCESS': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED
        }
        
        color = color_map.get(level, Colors.NC)
        console_msg = f"{color}[{level}]{Colors.NC} {message}"
        file_msg = f"[{timestamp}] [{level}] {message}"
        
        print(console_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(file_msg + '\n')
    
    def connect_db(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.conn.autocommit = False
            return True
        except Exception as e:
            self.log('ERROR', f"Cannot connect to database: {e}")
            return False
    
    def disconnect_db(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
    
    def execute_query(self, query, fetch=False):
        """Execute SQL query"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                if fetch:
                    return cur.fetchone()
                return True
        except Exception as e:
            self.log('ERROR', f"Query execution failed: {e}")
            return None if fetch else False
    
    def pre_flight_checks(self):
        """Run pre-flight checks"""
        self.log('INFO', "Running pre-flight checks...")
        
        # Verify all migration scripts exist
        for script in self.migration_scripts:
            script_path = self.script_dir / script
            if not script_path.exists():
                self.log('ERROR', f"Migration script not found: {script}")
                return False
        self.log('SUCCESS', "All migration scripts found")
        
        # Test database connection
        if not self.connect_db():
            self.log('ERROR', f"Cannot connect to database at {self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}")
            return False
        self.log('SUCCESS', "Database connection successful")
        
        # Check PostgreSQL version
        result = self.execute_query("SELECT version();", fetch=True)
        if result:
            self.log('INFO', f"PostgreSQL version: {result[0][:50]}...")
        
        # Check table size
        result = self.execute_query(
            "SELECT pg_size_pretty(pg_total_relation_size('qbn.ml_multi_timeframe_signals'));",
            fetch=True
        )
        if result:
            self.log('INFO', f"Current table size: {result[0]}")
        
        # Check for compressed chunks
        result = self.execute_query(
            """SELECT COUNT(*) FROM timescaledb_information.chunks 
               WHERE hypertable_name = 'ml_multi_timeframe_signals' 
               AND is_compressed = true;""",
            fetch=True
        )
        if result and result[0] > 0:
            self.log('WARNING', f"Found {result[0]} compressed chunks - may need decompression")
        
        # Check active connections
        result = self.execute_query(
            f"""SELECT COUNT(*) FROM pg_stat_activity 
               WHERE datname = '{self.db_params['database']}' 
               AND state != 'idle';""",
            fetch=True
        )
        if result:
            self.log('INFO', f"Active connections: {result[0]}")
        
        self.log('SUCCESS', "Pre-flight checks complete")
        return True
    
    def execute_migration(self, script_name):
        """Execute a single migration script"""
        script_path = self.script_dir / script_name
        
        self.log('INFO', "=" * 50)
        self.log('INFO', f"Executing: {script_name}")
        self.log('INFO', "=" * 50)
        
        start_time = time.time()
        
        try:
            # Read SQL file
            with open(script_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Filter out psql meta-commands (lines starting with \)
            # These are interactive psql commands that don't work via psycopg2
            filtered_lines = []
            for line in sql_content.split('\n'):
                stripped = line.strip()
                # Skip lines that start with psql meta-commands
                if not (stripped.startswith('\\') and len(stripped) > 1 and stripped[1].isalpha()):
                    filtered_lines.append(line)
            
            sql_content = '\n'.join(filtered_lines)
            
            # Execute SQL
            with self.conn.cursor() as cur:
                cur.execute(sql_content)
                self.conn.commit()
            
            duration = int(time.time() - start_time)
            self.log('SUCCESS', f"Completed: {script_name} ({duration}s)")
            return True
            
        except Exception as e:
            duration = int(time.time() - start_time)
            self.log('ERROR', f"Failed: {script_name} ({duration}s)")
            self.log('ERROR', f"Error details: {e}")
            self.conn.rollback()
            return False
    
    def post_migration_validation(self):
        """Run post-migration validation"""
        self.log('INFO', "Running post-migration validation...")
        
        all_valid = True
        
        # Verify outcome columns exist
        result = self.execute_query(
            """SELECT COUNT(*) FROM information_schema.columns 
               WHERE table_schema = 'qbn' 
               AND table_name = 'ml_multi_timeframe_signals' 
               AND column_name LIKE 'outcome_%';""",
            fetch=True
        )
        if result and result[0] >= 3:
            self.log('SUCCESS', f"Outcome columns verified ({result[0]} found)")
        else:
            self.log('ERROR', f"Outcome columns verification failed (expected 3+, found {result[0] if result else 0})")
            all_valid = False
        
        # Verify signal_classification table exists
        result = self.execute_query(
            """SELECT EXISTS (
               SELECT FROM information_schema.tables 
               WHERE table_schema = 'qbn' 
               AND table_name = 'signal_classification'
            );""",
            fetch=True
        )
        if result and result[0]:
            self.log('SUCCESS', "signal_classification table verified")
        else:
            self.log('ERROR', "signal_classification table not found")
            all_valid = False
        
        # Verify indexes created
        result = self.execute_query(
            """SELECT COUNT(*) FROM pg_indexes 
               WHERE schemaname = 'qbn' 
               AND tablename = 'ml_multi_timeframe_signals' 
               AND indexname LIKE '%outcome%';""",
            fetch=True
        )
        if result and result[0] >= 3:
            self.log('SUCCESS', f"Outcome indexes verified ({result[0]} found)")
        else:
            self.log('WARNING', f"Expected 3+ outcome indexes, found {result[0] if result else 0}")
        
        # Verify new signals_current columns
        result = self.execute_query(
            """SELECT COUNT(*) FROM information_schema.columns 
               WHERE table_schema = 'kfl' 
               AND table_name = 'signals_current' 
               AND (column_name LIKE '%adx%' OR column_name LIKE '%cmf%' 
                    OR column_name LIKE '%obv%' OR column_name LIKE '%stoch%' 
                    OR column_name LIKE '%ichimoku%');""",
            fetch=True
        )
        if result and result[0] >= 20:
            self.log('SUCCESS', f"signals_current expansion verified ({result[0]} new columns)")
        else:
            self.log('WARNING', f"Expected 20+ new signal columns, found {result[0] if result else 0}")
        
        # Verify MTF cache expansion
        result = self.execute_query(
            """SELECT COUNT(*) FROM information_schema.columns 
               WHERE table_schema = 'qbn' 
               AND table_name = 'ml_multi_timeframe_signals_cache' 
               AND (column_name LIKE '%adx%' OR column_name LIKE '%cmf%' 
                    OR column_name LIKE '%obv%' OR column_name LIKE '%stoch%' 
                    OR column_name LIKE '%ichimoku%');""",
            fetch=True
        )
        if result and result[0] >= 40:
            self.log('SUCCESS', f"MTF cache expansion verified ({result[0]} new columns)")
        else:
            self.log('WARNING', f"Expected 40+ new cache columns, found {result[0] if result else 0}")
        
        if all_valid:
            self.log('SUCCESS', "Post-migration validation complete")
        else:
            self.log('WARNING', "Post-migration validation completed with warnings")
        
        return all_valid
    
    def run(self):
        """Run the migration"""
        self.log('INFO', "=" * 50)
        self.log('INFO', "Phase 1.1 Database Schema Migration")
        self.log('INFO', "=" * 50)
        self.log('INFO', f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log('INFO', f"Log file: {self.log_file}")
        
        if self.dry_run:
            self.log('INFO', "DRY RUN MODE - No actual migrations will be executed")
        
        # Run pre-flight checks
        if not self.pre_flight_checks():
            self.log('ERROR', "Pre-flight checks failed. Aborting migration.")
            return 1
        
        if self.dry_run:
            self.log('INFO', "DRY RUN: Would execute the following scripts:")
            for script in self.migration_scripts:
                self.log('INFO', f"  - {script}")
            self.log('INFO', "DRY RUN: Validation would be performed after execution")
            self.log('SUCCESS', "DRY RUN complete")
            self.disconnect_db()
            return 0
        
        # Prompt for confirmation
        print()
        self.log('WARNING', "WARNING: This will modify production database schemas")
        confirmation = input("Continue with migration? (yes/y/no): ").strip().lower()
        
        if confirmation not in ['yes', 'y']:
            self.log('INFO', "Migration cancelled by user")
            self.disconnect_db()
            return 0
        
        # Execute migrations sequentially
        start_time = time.time()
        
        for script in self.migration_scripts:
            if not self.execute_migration(script):
                self.log('ERROR', f"Migration failed at script: {script}")
                self.log('ERROR', "Please review errors and consider running rollback script")
                self.disconnect_db()
                return 1
        
        total_duration = int(time.time() - start_time)
        
        # Run post-migration validation
        if not self.post_migration_validation():
            self.log('ERROR', "Post-migration validation failed")
            self.log('WARNING', "Migrations completed but validation detected issues")
            self.disconnect_db()
            return 1
        
        # Final summary
        self.log('INFO', "=" * 50)
        self.log('SUCCESS', "Phase 1.1 Migration Complete")
        self.log('INFO', "=" * 50)
        self.log('INFO', f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log('INFO', f"Total duration: {total_duration}s ({total_duration // 60}m)")
        self.log('INFO', f"Scripts executed: {len(self.migration_scripts)}")
        self.log('INFO', f"Log file: {self.log_file}")
        self.log('INFO', "")
        self.log('INFO', "Next steps:")
        self.log('INFO', "1. Run validation_queries.sql for detailed verification")
        self.log('INFO', "2. Monitor database performance for 24 hours")
        self.log('INFO', "3. Proceed with Phase 1.2 (Signal Classification Mapping)")
        
        self.disconnect_db()
        return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Phase 1.1 Database Schema Migration"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no actual changes)'
    )
    
    args = parser.parse_args()
    
    try:
        runner = MigrationRunner(dry_run=args.dry_run)
        return runner.run()
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

