from typer.testing import CliRunner
from kreda.main import app

runner = CliRunner()


def test_cli_no_audio_bypass(tmp_path):
    log_file = tmp_path / "run.csv"
    log_file.write_text(
        "t_ms,track,type,changed_or_what,detail,filename\n"
        "10000,0,EVENT,SAVE_slide,0,track_0_10000_slide.png\n"
    )

    grid_file = tmp_path / "grid.json"
    grid_file.write_text(
        '{"_stream_metadata": {"grid_dimensions": [12, 6]}, "frames": {}}'
    )

    result = runner.invoke(
        app,
        [str(tmp_path), "--no-audio", "--debug"],
    )

    assert result.exit_code == 0
    assert "Audio disabled." in result.stdout
