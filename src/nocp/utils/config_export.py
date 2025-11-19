"""Configuration export/import utilities"""
import yaml
from pathlib import Path
from typing import Optional
from ..core.config import ProxyConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

def export_config(
    config: ProxyConfig,
    output_path: Optional[Path] = None,
    include_secrets: bool = False
) -> Path:
    """
    Export configuration to YAML file.

    Args:
        config: Configuration to export
        output_path: Where to save (default: .nocp/config.yaml)
        include_secrets: Whether to include API keys (default: False)

    Returns:
        Path to exported config file
    """
    if output_path is None:
        output_path = Path(".nocp/config.yaml")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export config, excluding secrets by default
    exclude_fields = set()
    if not include_secrets:
        exclude_fields = {
            'gemini_api_key',
            'openai_api_key',
            'anthropic_api_key',
        }

    config_dict = config.model_dump(
        exclude=exclude_fields,
        exclude_none=True,
        mode='json'
    )

    with output_path.open("w") as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=True,
            indent=2
        )

    logger.info(f"Configuration exported to {output_path}")
    return output_path

def import_config(config_path: Path) -> ProxyConfig:
    """
    Import configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        ProxyConfig instance
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config_dict = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return ProxyConfig(**config_dict)

def print_config_diff(config1: ProxyConfig, config2: ProxyConfig) -> None:
    """Print differences between two configurations"""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Configuration Diff")
    table.add_column("Setting", style="cyan")
    table.add_column("Config 1", style="yellow")
    table.add_column("Config 2", style="green")

    dict1 = config1.model_dump(exclude_none=True)
    dict2 = config2.model_dump(exclude_none=True)

    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))

    for key in all_keys:
        val1 = dict1.get(key, "—")
        val2 = dict2.get(key, "—")

        if val1 != val2:
            table.add_row(key, str(val1), str(val2))

    console.print(table)
