from pathlib import Path


def test_gui_launch_scripts_exist_and_target_gui_modules():
    backend = Path("START_BACKEND_GUI.ps1").read_text(encoding="utf-8")
    frontend = Path("START_FRONTEND_GUI.ps1").read_text(encoding="utf-8")

    assert "-m src.backend.gui_app" in backend
    assert "-m src.frontend.gui_client" in frontend


def test_package_script_uses_six_class_model():
    script = Path("scripts/package_backend.ps1").read_text(encoding="utf-8")

    assert "models\\classroom_behaviour_6cls.pt" in script
    assert "START_BACKEND_GUI.ps1" in script
