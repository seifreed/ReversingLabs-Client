"""A1000 platform services, one class per area of the appliance's API.

Each service extends :class:`A1000Service`, which reaches the appliance
over an :class:`A1000Session` — the connection itself. Callers depend on
the narrow service they actually use, and services that need to share one
connection are built from one session::

    session = A1000Session(settings)
    samples = session.service(A1000SampleService)
    reports = session.service(A1000ReportService)

Work spanning two areas is a function over the services it needs, not a
class that unions them: see :func:`upload_and_get_report`.
"""

from rl_cli.services.a1000.metadata import A1000MetadataService
from rl_cli.services.a1000.network import A1000NetworkService
from rl_cli.services.a1000.reports import A1000ReportService
from rl_cli.services.a1000.samples import A1000SampleService
from rl_cli.services.a1000.service import A1000Service
from rl_cli.services.a1000.session import A1000Session
from rl_cli.services.a1000.workflows import upload_and_get_report, wait_for_uploaded_analysis
from rl_cli.services.a1000.yara import A1000YaraService

__all__ = [
    "A1000MetadataService",
    "A1000NetworkService",
    "A1000ReportService",
    "A1000SampleService",
    "A1000Service",
    "A1000Session",
    "A1000YaraService",
    "upload_and_get_report",
    "wait_for_uploaded_analysis",
]
