from typer.testing import CliRunner
from kreda.main import app

runner = CliRunner()


def test_cli_no_audio_bypass(tmp_path):
    log_file = tmp_path / "run.csv"
    log_file.write_text("t_ms,track,type,changed_or_what,detail\n")
    log_file.write_text("10000,0,EVENT,SAVE_slide,0\n")

    result = runner.invoke(
        app,
        [str(tmp_path), "--no-audio", "--log-file", "run.csv", "--debug"],
    )

    assert result.exit_code == 0
    assert "Audio disabled." in result.stdout
