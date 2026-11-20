"""Create the backends of the model server: the local models, or the fake one."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  time

from    factdesk.ai_logger                   import  aiLogger
from    factdesk.integrations.model_backend  import  ModelBackend
from    factdesk.knowledge.hash_embeddings   import  HashEmbeddings
from    factdesk.llm.fake_chat_model         import  FakeChatModel, FakeRules
from    factdesk.llm.fake_rules              import  project_fake_rules


#==========================================================================
# Local Qwen + sentence-transformers
#==========================================================================
def load_local_backend(
    *,
    chat_name: str,
    embed_name: str,
    max_new_tokens: int,
) -> ModelBackend:
    """
    Load the chat model and the embedding model of this laptop. The code's
    own embeddings use the same HuggingFaceEmbeddings, so a collection built
    by the code and one built through the server hold the same vectors.

    Input:
        . chat_name     : Hugging Face id of the chat model
        . embed_name    : Hugging Face id of the embedding model
        . max_new_tokens: default answer length when a request sends none
    Output:
        . backend       : a ModelBackend for make_server
    """
    from    langchain_huggingface           import  HuggingFaceEmbeddings

    from    factdesk.llm.local_chat_model   import  LocalChatModel

    started    = time.time()
    chat       = LocalChatModel(model_name=chat_name, max_new_tokens=max_new_tokens)
    embeddings = HuggingFaceEmbeddings(model_name=embed_name)
    aiLogger.info("Loaded %s + %s in %.0f s", chat_name, embed_name, time.time() - started)
    return ModelBackend(chat=chat, embeddings=embeddings, chat_name=chat_name, embed_name=embed_name)


#==========================================================================
# Fake model + hashing embeddings
#==========================================================================
def fake_backend(*, rules: FakeRules | None = None) -> ModelBackend:
    """
    Build a backend that needs no download: rule answers, hashing vectors.
    For tests and for checking the HTTP path; never for real results.

    Input:
        . rules         : fake-model rules; None = project_fake_rules()
    Output:
        . backend       : a ModelBackend for make_server
    """
    chat       = FakeChatModel(rules=rules if rules is not None else project_fake_rules())
    embeddings = HashEmbeddings()
    return ModelBackend(
        chat        = chat,
        embeddings  = embeddings,
        chat_name   = "fake-chat",
        embed_name  = embeddings.name,
    )
