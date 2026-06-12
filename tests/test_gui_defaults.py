import shutil
import subprocess
from pathlib import Path


def test_gui_launch_scripts_exist_and_target_gui_modules():
    backend = Path("START_BACKEND_GUI.ps1").read_text(encoding="utf-8")
    frontend = Path("START_FRONTEND_GUI.ps1").read_text(encoding="utf-8")

    assert "-m src.backend.gui_app" in backend
    assert "-m src.frontend.gui_client" in frontend


def test_package_script_uses_six_class_model():
    script = Path("scripts/package_backend.ps1").read_text(encoding="utf-8")

    assert "models\\student_behaviour_v6_6cls_img960_e50_best.pt" in script
    assert "START_BACKEND_GUI.ps1" in script
    assert "function Test-IsSubPath" in script
    assert "$modelFullPath.StartsWith" not in script
    assert "$resolvedPackage.StartsWith" not in script


def test_package_script_builds_backend_package_with_gui_files():
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    assert powershell is not None

    model = Path("tmp/test-package-model.pt")
    output_dir = Path("tmp/test-backend-package-dist")
    package_dir = output_dir / "test-backend-package"

    try:
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"fake model")
        shutil.rmtree(output_dir, ignore_errors=True)

        subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts/package_backend.ps1",
                "-OutputDir",
                str(output_dir),
                "-PackageName",
                "test-backend-package",
                "-ModelPath",
                str(model),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert (package_dir / "START_BACKEND_GUI.ps1").exists()
        assert (package_dir / "START_BACKEND.ps1").exists()
        assert (package_dir / "src" / "backend" / "gui_app.py").exists()
        assert (package_dir / "src" / "common" / "protocol.py").exists()
        assert (
            package_dir / "models" / "student_behaviour_v6_6cls_img960_e50_best.pt"
        ).read_bytes() == b"fake model"
        assert (output_dir / "test-backend-package.zip").exists()

        readme = (package_dir / "README_BACKEND.md").read_text(encoding="utf-8")
        assert "START_BACKEND_GUI.ps1" in readme
        assert "START_BACKEND.ps1" in readme
    finally:
        model.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
