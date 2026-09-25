"""
Tests for release metadata.
"""

import re
from pathlib import Path

import ai_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVersion:
    """Tests for the version strings that ship to consumers."""

    def test_declared_and_runtime_versions_match(self):
        """Test the packaged version equals the one the package reports.

        0.4.6 shipped with pyproject.toml at 0.4.6 and __init__.py still at 0.4.5, so pip
        and ai_client.__version__ disagreed for everyone who installed it.
        """
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        packaged = re.search(r'^version = "(.+?)"', pyproject, re.M)

        assert packaged is not None
        assert ai_client.__version__ == packaged.group(1)

    def test_changelog_documents_the_current_version(self):
        """Test the release being published has release notes."""
        changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        assert re.search(rf"^## \[{re.escape(ai_client.__version__)}\]", changelog, re.M)
