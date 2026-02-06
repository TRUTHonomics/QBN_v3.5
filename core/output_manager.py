"""Centraal beheer van validation/analyse output directories."""
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ValidationOutputManager:
    """Beheert validation output directories met gestandaardiseerd format."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "_validation"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.archive_dir = self.base_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
    
    def create_output_dir(
        self,
        script_name: str,
        asset_id: int,
        run_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Path:
        """
        Creëert output directory met format:
        {YYMMDD-HHMMSS}-{scriptname}-asset_{id}-{run_id}/
        
        Archiveert bestaande directories voor dezelfde combinatie van
        script_name, asset_id en run_id naar archive/ subdirectory.
        
        Args:
            script_name: Naam van het script (bijv. 'threshold_analysis')
            asset_id: Asset ID
            run_id: Run identifier (default: 'manual')
            timestamp: Timestamp voor directory naam (default: now)
        
        Returns:
            Path object naar de aangemaakte directory
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if run_id is None:
            run_id = "manual"
        
        # Archiveer bestaande directories voor deze combinatie
        self._archive_existing(script_name, asset_id, run_id)
        
        # Format: 260206-114530-threshold_analysis-asset_9889-test-abc123
        dir_name = (
            f"{timestamp.strftime('%y%m%d-%H%M%S')}-"
            f"{script_name}-"
            f"asset_{asset_id}-"
            f"{run_id}"
        )
        
        output_dir = self.base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {output_dir}")
        
        return output_dir
    
    def _archive_existing(self, script_name: str, asset_id: int, run_id: str):
        """
        Archiveert bestaande output directories voor dezelfde script/asset/run_id combinatie.
        
        Pattern: *-{script_name}-asset_{asset_id}-{run_id}
        """
        import shutil
        
        pattern = f"*-{script_name}-asset_{asset_id}-{run_id}"
        existing = list(self.base_dir.glob(pattern))
        
        if not existing:
            return
        
        archive_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for old_dir in existing:
            try:
                # Nieuwe naam in archive: {original_name}
                archive_path = self.archive_dir / old_dir.name
                
                # Als archive path al bestaat, voeg timestamp toe
                if archive_path.exists():
                    archive_path = self.archive_dir / f"{old_dir.name}_{archive_timestamp}"
                
                shutil.move(str(old_dir), str(archive_path))
                logger.info(f"Archived existing output: {old_dir.name} → archive/{archive_path.name}")
            except Exception as e:
                logger.error(f"Failed to archive {old_dir}: {e}")
    
    def find_run_outputs(self, run_id: str) -> List[Path]:
        """
        Vindt alle output directories voor een specifieke run_id.
        
        Args:
            run_id: Run identifier om te zoeken
        
        Returns:
            Gesorteerde lijst van Path objecten
        """
        pattern = f"*-{run_id}"
        matches = list(self.base_dir.glob(pattern))
        return sorted(matches)
    
    def find_latest_run(self, asset_id: int) -> Optional[Path]:
        """
        Vindt de laatste output directory voor een specifiek asset.
        
        Args:
            asset_id: Asset ID om te zoeken
        
        Returns:
            Path naar laatste directory, of None als geen gevonden
        """
        pattern = f"*-asset_{asset_id}-*"
        matches = sorted(self.base_dir.glob(pattern), reverse=True)
        return matches[0] if matches else None
    
    def cleanup_old_outputs(self, days_to_keep: int = 90, keep_latest_per_asset: bool = True):
        """
        Verwijdert outputs ouder dan X dagen.
        
        Args:
            days_to_keep: Aantal dagen om outputs te bewaren
            keep_latest_per_asset: Behoud altijd de laatste run per asset
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Verzamel alle directories met timestamps
        all_dirs = []
        for path in self.base_dir.iterdir():
            if not path.is_dir():
                continue
            
            # Parse timestamp uit directory naam (YYMMDD-HHMMSS)
            try:
                timestamp_str = path.name.split('-')[0] + '-' + path.name.split('-')[1]
                dir_timestamp = datetime.strptime(timestamp_str, '%y%m%d-%H%M%S')
                
                # Extract asset_id
                parts = path.name.split('-')
                asset_part = [p for p in parts if p.startswith('asset_')]
                asset_id = int(asset_part[0].replace('asset_', '')) if asset_part else None
                
                all_dirs.append({
                    'path': path,
                    'timestamp': dir_timestamp,
                    'asset_id': asset_id
                })
            except (ValueError, IndexError):
                logger.warning(f"Could not parse directory name: {path.name}")
                continue
        
        # Bepaal welke te verwijderen
        latest_per_asset = {}
        if keep_latest_per_asset:
            for item in all_dirs:
                if item['asset_id'] is not None:
                    if item['asset_id'] not in latest_per_asset:
                        latest_per_asset[item['asset_id']] = item
                    elif item['timestamp'] > latest_per_asset[item['asset_id']]['timestamp']:
                        latest_per_asset[item['asset_id']] = item
        
        # Verwijder oude directories
        removed_count = 0
        for item in all_dirs:
            # Skip als het de laatste van dit asset is
            if keep_latest_per_asset and item['asset_id'] in latest_per_asset:
                if item['path'] == latest_per_asset[item['asset_id']]['path']:
                    continue
            
            # Verwijder als ouder dan cutoff
            if item['timestamp'] < cutoff_date:
                try:
                    import shutil
                    shutil.rmtree(item['path'])
                    logger.info(f"Removed old output: {item['path'].name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {item['path']}: {e}")
        
        logger.info(f"Cleanup complete: {removed_count} directories removed")
        return removed_count
