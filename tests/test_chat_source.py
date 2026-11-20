"""ChatSource: lazy creation, one model per backend, project rules for fake."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    factdesk.config                 import  ModelConfig, ServerConfig
from    factdesk.llm.chat_source        import  ChatSource
from    factdesk.llm.fake_chat_model    import  FakeChatModel, FakeRules
from    factdesk.llm.server_chat        import  ServerChat


#==========================================================================
# Server backend
#==========================================================================
def test_server_backend_returns_a_server_chat_on_first_use() -> None:
    """
    Pin lazy creation and the ServerChat settings taken from the config.
    """
    source = ChatSource(ModelConfig(backend="server", max_new_tokens=50), ServerConfig(model_port=8123))
    assert not source.loaded
    chat = source.get()
    assert isinstance(chat, ServerChat)
    assert (chat.base_url, chat.max_tokens) == ("http://127.0.0.1:8123/v1", 50)
    assert source.get() is chat and source.loaded


#==========================================================================
# Fake backend with injected rules
#==========================================================================
def test_fake_backend_uses_the_injected_rules() -> None:
    """
    Pin that rules passed to ChatSource reach the fake model.
    """
    rules  = FakeRules(text=lambda prompt: "from the project rule")
    source = ChatSource(ModelConfig(backend="fake"), ServerConfig(), rules=rules)
    chat   = source.get()
    assert isinstance(chat, FakeChatModel)
    assert chat.invoke("hi").content == "from the project rule"
