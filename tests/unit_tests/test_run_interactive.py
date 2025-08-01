# Standard Library
from unittest.mock import MagicMock, patch

# SynThesisAI Modules
from core.cli.run_interactive import main


@patch("core.cli.run_interactive.run_generation_pipeline")
@patch("core.cli.run_interactive.save_prompts")
@patch("core.cli.run_interactive.get_config_manager")
@patch("core.cli.run_interactive.get_input")
def test_run_interactive_defaults(mock_input, mock_config_mgr, mock_save_prompts, mock_pipeline):
    # Simulate leaving all inputs blank
    mock_input.return_value = ""

    # Dummy config manager
    config = {
        "num_problems": 2,
        "output_dir": "test_output",
        "taxonomy": "taxonomy/sample_math_taxonomy.json",
        "default_batch_id": "batch_test",
    }
    config_manager_mock = MagicMock()
    config_manager_mock.get.side_effect = lambda key, default=None: config.get(key, default)
    config_manager_mock.get_all.return_value = config
    mock_config_mgr.return_value = config_manager_mock

    # Dummy pipeline result
    mock_pipeline.return_value = (["accepted1"], ["rejected1"], MagicMock())

    main()

    mock_pipeline.assert_called_once()
    mock_save_prompts.assert_called_once()
