#!/usr/bin/env python3
"""
Merlina Version Bumping Tool

Automatically bumps version numbers following Semantic Versioning.
Updates version.py, CHANGELOG.md, and creates git tags.

Usage:
    python bump_version.py [major|minor|patch] [--release-name "Name"] [--dry-run]

Examples:
    python bump_version.py patch              # 1.2.0 -> 1.2.1
    python bump_version.py minor              # 1.2.0 -> 1.3.0
    python bump_version.py major              # 1.2.0 -> 2.0.0
    python bump_version.py patch --release-name "Bug Fixes"
    python bump_version.py minor --dry-run    # Preview changes without applying
"""

import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ANSI color codes for pretty output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a styled header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def get_current_version() -> Tuple[int, int, int]:
    """Read current version from version.py"""
    version_file = Path(__file__).parent / "version.py"

    if not version_file.exists():
        print_error("version.py not found!")
        sys.exit(1)

    content = version_file.read_text()
    match = re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', content)

    if not match:
        print_error("Could not parse version from version.py")
        sys.exit(1)

    return tuple(map(int, match.groups()))

def bump_version(current: Tuple[int, int, int], bump_type: str) -> Tuple[int, int, int]:
    """Calculate new version based on bump type"""
    major, minor, patch = current

    if bump_type == "major":
        return (major + 1, 0, 0)
    elif bump_type == "minor":
        return (major, minor + 1, 0)
    elif bump_type == "patch":
        return (major, minor, patch + 1)
    else:
        print_error(f"Invalid bump type: {bump_type}")
        print_info("Valid types: major, minor, patch")
        sys.exit(1)

def update_version_file(new_version: Tuple[int, int, int], release_name: str = None, dry_run: bool = False):
    """Update version.py with new version"""
    version_file = Path(__file__).parent / "version.py"
    content = version_file.read_text()

    version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    today = datetime.now().strftime("%Y-%m-%d")

    # Update version string
    content = re.sub(
        r'__version__ = "[\d\.]+"',
        f'__version__ = "{version_str}"',
        content
    )

    # Update release date
    content = re.sub(
        r'RELEASE_DATE = "[\d-]+"',
        f'RELEASE_DATE = "{today}"',
        content
    )

    # Update release name if provided
    if release_name:
        content = re.sub(
            r'RELEASE_NAME = ".*?"',
            f'RELEASE_NAME = "{release_name}"',
            content
        )

    if dry_run:
        print_info("DRY RUN: Would update version.py")
        print(f"  New content preview:")
        for line in content.split('\n')[:20]:
            if '__version__' in line or 'RELEASE_' in line:
                print(f"    {Colors.GREEN}{line}{Colors.END}")
    else:
        version_file.write_text(content)
        print_success(f"Updated version.py to {version_str}")

def update_changelog(new_version: Tuple[int, int, int], release_name: str = None, dry_run: bool = False):
    """Update CHANGELOG.md with new version"""
    changelog_file = Path(__file__).parent / "CHANGELOG.md"

    if not changelog_file.exists():
        print_warning("CHANGELOG.md not found, skipping")
        return

    content = changelog_file.read_text()
    version_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    today = datetime.now().strftime("%Y-%m-%d")

    release_title = f'[{version_str}] - {today}'
    if release_name:
        release_title += f' "{release_name}"'

    # Create new version entry
    new_entry = f"""## {release_title}

### Added
-

### Changed
-

### Fixed
-

"""

    # Insert after [Unreleased] section
    if "## [Unreleased]" in content:
        content = content.replace(
            "## [Unreleased]",
            f"## [Unreleased]\n\n{new_entry}"
        )
    else:
        # Insert at the top of changelog entries
        match = re.search(r'(## \[\d+\.\d+\.\d+\])', content)
        if match:
            pos = match.start()
            content = content[:pos] + new_entry + content[pos:]

    if dry_run:
        print_info("DRY RUN: Would update CHANGELOG.md")
        print(f"  New entry:\n{Colors.YELLOW}{new_entry}{Colors.END}")
    else:
        changelog_file.write_text(content)
        print_success("Updated CHANGELOG.md")
        print_warning("Don't forget to fill in the changelog entries!")

def check_git_status() -> bool:
    """Check if git working directory is clean"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        return len(result.stdout.strip()) == 0
    except subprocess.CalledProcessError:
        return False

def create_git_tag(version: Tuple[int, int, int], dry_run: bool = False):
    """Create git tag for the new version"""
    version_str = f"{version[0]}.{version[1]}.{version[2]}"
    tag_name = f"v{version_str}"

    if dry_run:
        print_info(f"DRY RUN: Would create git tag {tag_name}")
        return

    try:
        # Check if tag already exists
        result = subprocess.run(
            ["git", "tag", "-l", tag_name],
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout.strip():
            print_warning(f"Git tag {tag_name} already exists")
            return

        # Create annotated tag
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release {version_str}"],
            check=True
        )
        print_success(f"Created git tag {tag_name}")
        print_info(f"Push with: git push origin {tag_name}")

    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create git tag: {e}")

def main():
    """Main entry point"""
    # Parse arguments
    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)

    bump_type = args[0]
    release_name = None
    dry_run = False

    # Parse optional arguments
    i = 1
    while i < len(args):
        if args[i] == "--release-name" and i + 1 < len(args):
            release_name = args[i + 1]
            i += 2
        elif args[i] == "--dry-run":
            dry_run = True
            i += 1
        else:
            print_error(f"Unknown argument: {args[i]}")
            sys.exit(1)

    if bump_type not in ["major", "minor", "patch"]:
        print_error(f"Invalid bump type: {bump_type}")
        print_info("Valid types: major, minor, patch")
        sys.exit(1)

    # Start the bumping process
    print_header("🔮 Merlina Version Bumping Tool")

    if dry_run:
        print_warning("DRY RUN MODE - No changes will be made\n")

    # Get current version
    current = get_current_version()
    current_str = f"{current[0]}.{current[1]}.{current[2]}"
    print_info(f"Current version: {current_str}")

    # Calculate new version
    new_version = bump_version(current, bump_type)
    new_str = f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
    print_info(f"New version: {new_str}")

    if release_name:
        print_info(f"Release name: {release_name}")

    print()

    # Check git status
    if not dry_run:
        if not check_git_status():
            print_warning("Git working directory is not clean")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print_info("Aborted")
                sys.exit(0)

    # Update files
    update_version_file(new_version, release_name, dry_run)
    update_changelog(new_version, release_name, dry_run)

    # Create git tag
    if not dry_run:
        create_git_tag(new_version, dry_run)

    # Summary
    print_header("Summary")
    print_success(f"Version bumped: {current_str} → {new_str}")

    if not dry_run:
        print_info("\nNext steps:")
        print("  1. Review and edit CHANGELOG.md")
        print("  2. Commit changes: git add version.py CHANGELOG.md")
        print(f"  3. Commit: git commit -m 'Bump version to {new_str}'")
        print(f"  4. Push tag: git push origin v{new_str}")
    else:
        print_info("\nThis was a dry run. Run without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
