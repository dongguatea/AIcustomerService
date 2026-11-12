from app.tools import control


def test_suggest_action_tool_shape():
    assert control.suggest_action.name == "suggest_action"
    assert (
        control.SuggestActionArgs.model_validate({"action": "transfer_human"}).action
        == "transfer_human"
    )
