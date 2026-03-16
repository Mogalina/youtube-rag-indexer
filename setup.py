import os
import sys
import subprocess

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop


def _download_models() -> None:
    """Attempt to download models after installation."""
    print("YouTube Indexer model download")
    
    try:
        env = os.environ.copy()
        src_path = os.path.join(os.getcwd(), "src")
        
        current_pythonpath = env.get("PYTHONPATH", "")
        if current_pythonpath:
            env["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_pythonpath}"
        else:
            env["PYTHONPATH"] = src_path
        
        print("Downloading models...")
        subprocess.check_call([sys.executable, "-m", "cli", "download-models"], env=env)
        print("Models downloaded successfully.")
    except Exception as e:
        print(f"\nWarning: Automatic model download failed: {e}")
        print("You can manually download them later using: tubx download-models")


class PostInstallCommand(install):
    """
    Post-install command to download models.
    """
    
    def run(self) -> None:
        """Run the post-install command."""
        install.run(self)
        _download_models()


class PostDevelopCommand(develop):
    """
    Post-develop command to download models.
    """
    
    def run(self) -> None:
        """Run the post-develop command."""
        develop.run(self)
        _download_models()


if __name__ == "__main__":
    setup(
        cmdclass={
            'install': PostInstallCommand,
            'develop': PostDevelopCommand,
        },
    )
