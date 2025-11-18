"""
Bootstrap script to ensure uv is available.

This module handles automatic installation of uv if not present,
making the setup seamless for users.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


class UVBootstrap:
    """Handles uv installation and verification."""

    def __init__(self):
        self.system = platform.system().lower()
        self.uv_bin = self._find_uv()

    def _find_uv(self) -> Optional[Path]:
        """Try to find uv in PATH or common locations."""
        # Check PATH first
        which_cmd = "where" if self.system == "windows" else "which"
        try:
            result = subprocess.run(
                [which_cmd, "uv"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                uv_path = result.stdout.strip()
                if uv_path:
                    return Path(uv_path)
        except FileNotFoundError:
            pass

        # Check common installation locations
        home = Path.home()
        common_paths = [
            home / ".local" / "bin" / "uv",
            home / ".cargo" / "bin" / "uv",
            Path("/usr/local/bin/uv"),
        ]

        for path in common_paths:
            if path.exists() and path.is_file():
                return path

        return None

    def is_installed(self) -> bool:
        """Check if uv is installed and accessible."""
        return self.uv_bin is not None

    def install(self) -> bool:
        """
        Install uv using the official installation script.

        Returns:
            True if installation successful, False otherwise
        """
        print("🔧 uv not found. Installing uv...")

        if self.system == "windows":
            return self._install_windows()
        else:
            return self._install_unix()

    def _install_unix(self) -> bool:
        """Install uv on Unix-like systems (Linux, macOS)."""
        try:
            # Use the official installation script
            install_cmd = [
                "sh", "-c",
                "curl -LsSf https://astral.sh/uv/install.sh | sh"
            ]

            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                # Update PATH for current session
                uv_bin_dir = Path.home() / ".local" / "bin"
                os.environ["PATH"] = f"{uv_bin_dir}:{os.environ.get('PATH', '')}"

                # Re-check for uv
                self.uv_bin = self._find_uv()

                if self.uv_bin:
                    print(f"✅ uv installed successfully at {self.uv_bin}")
                    self._show_path_message()
                    return True
                else:
                    print("⚠️  uv installation completed but binary not found")
                    return False
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error during installation: {e}")
            return False

    def _install_windows(self) -> bool:
        """Install uv on Windows."""
        try:
            # Use PowerShell installation script
            install_cmd = [
                "powershell", "-c",
                "irm https://astral.sh/uv/install.ps1 | iex"
            ]

            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                self.uv_bin = self._find_uv()
                if self.uv_bin:
                    print(f"✅ uv installed successfully at {self.uv_bin}")
                    return True
                else:
                    print("⚠️  uv installation completed but binary not found")
                    return False
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error during installation: {e}")
            return False

    def _show_path_message(self):
        """Show message about updating PATH if needed."""
        shell = os.environ.get("SHELL", "")

        if "zsh" in shell:
            rc_file = "~/.zshrc"
        elif "bash" in shell:
            rc_file = "~/.bashrc"
        else:
            rc_file = "your shell's RC file"

        print(f"\n💡 Note: You may need to add uv to your PATH by adding this to {rc_file}:")
        print('   export PATH="$HOME/.local/bin:$PATH"')
        print("   Then restart your shell or run: source {}".format(rc_file))

    def get_uv_command(self) -> list[str]:
        """Get the command to run uv."""
        if self.uv_bin:
            return [str(self.uv_bin)]
        else:
            # Fallback to hoping it's in PATH
            return ["uv"]

    def ensure_available(self) -> bool:
        """
        Ensure uv is available, installing if necessary.

        Returns:
            True if uv is available, False otherwise
        """
        if self.is_installed():
            return True

        print("🚀 uv is required but not installed.")
        print("   uv is a fast Python package manager from Astral.")
        print("   More info: https://github.com/astral-sh/uv\n")

        # Ask for permission (in production, could be auto-yes)
        try:
            response = input("Install uv now? [Y/n]: ").strip().lower()
            if response in ("", "y", "yes"):
                return self.install()
            else:
                print("❌ uv is required to run nocp. Installation cancelled.")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Installation cancelled.")
            return False


def bootstrap_uv() -> bool:
    """
    Main bootstrap function to ensure uv is available.

    Returns:
        True if uv is available, False otherwise
    """
    bootstrap = UVBootstrap()
    return bootstrap.ensure_available()


def get_uv_command() -> list[str]:
    """Get the command to run uv."""
    bootstrap = UVBootstrap()
    return bootstrap.get_uv_command()


if __name__ == "__main__":
    # Allow running as standalone script for testing
    success = bootstrap_uv()
    sys.exit(0 if success else 1)
