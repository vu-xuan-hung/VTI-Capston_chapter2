"""The RAG slice: "what does this number mean?" answered from the notes and indicator chunks.

This is the part of the project that Langflow can express: Chat Input ->
embeddings -> Chroma -> Parser -> Prompt -> model -> Chat Output.
Every value comes from [rag] in configs/app.toml, the same values the flow uses
(flows/FLOW_SPEC.json), so the code and the flow cannot drift.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    langchain_core.language_models.chat_models  import  BaseChatModel
from    langchain_core.output_parsers               import  StrOutputParser
from    langchain_core.prompts                      import  ChatPromptTemplate
from    langchain_core.runnables                    import  RunnableConfig

from    factdesk.config                  import  SliceConfig
from    factdesk.knowledge.vector_store  import  VectorStore


#==========================================================================
# Answer from the notes
#==========================================================================
def answer_from_notes(
    question: str,
    *,
    store: VectorStore,
    chat: BaseChatModel,
    rag: SliceConfig,
    run_config: RunnableConfig | None = None,
) -> tuple[str, list[str]]:
    """
    Search the collection (no filter: the Langflow component has none), join
    the chunks with the parser pattern, and ask the model to answer from them only.

    Input:
        . question      : e.g. "What does GDP per person mean here?"
        . store         : the entity collection
        . chat          : the chat model
        . rag           : k, parser_pattern, system, template
        . run_config    : tags, metadata, tracer
    Output:
        . result        : (answer text, the chunk ids that were given to the model)
    """
    docs    = store.retriever(k=rag.k).invoke(question, config=run_config)
    context = "\n".join(rag.parser_pattern.format(chunk_id=doc.metadata["chunk_id"], text=doc.page_content) for doc in docs)
    prompt  = ChatPromptTemplate.from_messages([("system", rag.system), ("human", rag.template)])
    answer  = (prompt | chat | StrOutputParser()).invoke({"context": context, "question": question}, config=run_config)
    return answer.strip(), [doc.metadata["chunk_id"] for doc in docs]
