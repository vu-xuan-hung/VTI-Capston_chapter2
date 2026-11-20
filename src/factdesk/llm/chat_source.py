"""Create the chat model only when something really needs it."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    langchain_core.language_models.chat_models  import  BaseChatModel

from    factdesk.ai_logger              import  aiLogger
from    factdesk.config                 import  ModelConfig, ServerConfig
from    factdesk.llm.fake_chat_model    import  FakeChatModel, FakeRules
from    factdesk.llm.fake_rules         import  project_fake_rules
from    factdesk.llm.server_chat        import  ServerChat


#==========================================================================
# Lazy chat model
#==========================================================================
class ChatSource:
    """
    Purpose: create one chat model on first use, for the configured backend:
    local (Qwen here), server (the model server) or fake (rules). Commands
    that never call a model never load one.
    """

    #----------------------------------------------------------------------
    # Keep the settings; nothing is loaded here
    #----------------------------------------------------------------------
    def __init__(
        self,
        model: ModelConfig,
        servers: ServerConfig,
        *,
        rules: FakeRules | None = None,
    ) -> None:
        """
        Remember how to create the model.

        Input:
            . model         : backend and model names
            . servers       : where the model server listens (backend "server")
            . rules         : fake-model rules; None = project_fake_rules()
        """
        self._model   = model
        self._servers = servers
        self._rules   = rules if rules is not None else project_fake_rules()
        self._chat: BaseChatModel | None = None

    #----------------------------------------------------------------------
    # Return the chat model
    #----------------------------------------------------------------------
    def get(self) -> BaseChatModel:
        """
        Create the chat model the first time, then return the same one.

        Output:
            . chat          : a LangChain chat model
        """
        if self._chat is None:
            aiLogger.info("Loading chat backend: %s", self._model.backend)
            self._chat = self._create()
        return self._chat

    #----------------------------------------------------------------------
    # Say whether a model was created
    #----------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        """
        Tell whether this process created the model (a resume may not need one).

        Output:
            . loaded        : True after the first get()
        """
        return self._chat is not None

    #----------------------------------------------------------------------
    # Build the configured backend; torch loads only for "local"
    #----------------------------------------------------------------------
    def _create(self) -> BaseChatModel:
        """
        Build the model of the configured backend.

        Output:
            . chat          : FakeChatModel, ServerChat or LocalChatModel
        """
        if self._model.backend == "fake":
            return FakeChatModel(rules=self._rules)
        if self._model.backend == "server":
            return ServerChat(
                base_url    = self._servers.model_url,
                model_name  = self._model.chat_model,
                max_tokens  = self._model.max_new_tokens,
                timeout     = self._servers.request_timeout,
            )
        from    factdesk.llm.local_chat_model   import  LocalChatModel

        return LocalChatModel(
            model_name      = self._model.chat_model,
            max_new_tokens  = self._model.max_new_tokens,
        )
