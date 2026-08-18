"""Rich-based renderers used by the CLI layer.

Keeping presentation here lets ``cli/commands/`` stay thin: command handlers
parse Click arguments, call the service, and forward the result to one of
these helpers.
"""

from rl_cli.render.formatters.a1000_listings import (
    print_extracted_files_table,
    print_samples_panels,
    print_search_results_table,
)
from rl_cli.render.formatters.a1000_operations import (
    print_reanalyze_results_table,
    print_yara_content,
    print_yara_rulesets_table,
)
from rl_cli.render.formatters.a1000_sample_report import (
    print_analysis_status,
    print_report_summary,
    print_summary_panel,
    print_titanium_report,
)
from rl_cli.render.formatters.config_report import (
    print_availability_panel,
    print_profile_names,
    print_service_lines,
)
from rl_cli.render.formatters.ticloud_report import print_file_analysis

__all__ = [
    "print_analysis_status",
    "print_availability_panel",
    "print_extracted_files_table",
    "print_file_analysis",
    "print_profile_names",
    "print_reanalyze_results_table",
    "print_report_summary",
    "print_samples_panels",
    "print_search_results_table",
    "print_service_lines",
    "print_summary_panel",
    "print_titanium_report",
    "print_yara_content",
    "print_yara_rulesets_table",
]
