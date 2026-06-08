from src.frontend.camera_client import build_arg_parser


def test_camera_client_parser_defaults():
    args = build_arg_parser().parse_args(["--host", "192.168.1.20"])

    assert args.host == "192.168.1.20"
    assert args.port == 5001
    assert args.camera == 0
    assert args.width == 640
    assert args.fps == 8.0
    assert args.quality == 80
