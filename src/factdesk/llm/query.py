"""The parsed question: a small schema the model fills, and the chain that fills it."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

from    typing                              import  Literal

from    langchain_core.language_models.chat_models import  BaseChatModel
from    langchain_core.prompts              import  ChatPromptTemplate
from    langchain_core.runnables            import  Runnable
from    pydantic                            import  BaseModel, Field

from    factdesk.llm.prompts                import  PARSE_HUMAN, PARSE_SYSTEM


#==========================================================================
# The parsed question
#==========================================================================
class Query(BaseModel):
    """
    Purpose: what the model reads out of a question: the operation, the
    quantity, the country names and the years. Four flat fields, an enum
    instead of free text, so a small model can fill it.
    """

    operation: Literal["value", "compare", "growth", "rank", "other"] = Field(
        description="Chọn value nếu hỏi giá trị của chỉ số, compare nếu yêu cầu so sánh "
        "giữa các quốc gia hoặc các năm, growth nếu hỏi mức tăng giảm hoặc "
        "tốc độ tăng trưởng theo thời gian, rank nếu hỏi thứ hạng hoặc quốc gia "
        "có giá trị cao nhất hay thấp nhất, và other nếu không thuộc các nhóm trên."
    )
    indicator_text: str = Field(
        description="Tên chỉ số theo cách người dùng viết. Dùng chuỗi rỗng nếu không xác định được."
    )
    country_texts: list[str] = Field(
        description="Danh sách tên quốc gia theo cách người dùng viết. Dùng [] nếu không có quốc gia cụ thể."
    )
    years: list[int] = Field(
        description="Danh sách các năm được nêu rõ trong câu hỏi, dưới dạng số nguyên. Dùng [] nếu không có năm cụ thể."
    )


    # TODO T01 -- add the four fields of Query,
    # in this order: operation (Literal of value, compare, growth, rank, other),
    # indicator_text (str), country_texts (list[str]),
    # years (list[int]); each with Field(description=...) the model can read.
    # CHECK: pytest -k t01 -- the field names are exactly these four and an invalid operation is refused.


#==========================================================================
# The parse chain
#==========================================================================
def build_parse_chain(chat: BaseChatModel) -> Runnable:
    """
    Build prompt | model-with-structured-output; invoke it with {"question": ...}
    and it returns a Query.

    Input:
        . chat          : any LangChain chat model
    Output:
        . chain         : a runnable: {"question": str} -> Query
    """
    # TODO T01 -- build ChatPromptTemplate.from_messages([("system", PARSE_SYSTEM), ("human", PARSE_HUMAN)]) and join it
    # with chat.with_structured_output(Query) using the | operator; return the chain.
    # CHECK: pytest -k t01 -- invoke({"question": ...}) returns a Query with the operation, country texts and years.
    chatTemple = ChatPromptTemplate.from_messages(
        [("system", PARSE_SYSTEM), ("human", PARSE_HUMAN)]
    )
    return chatTemple | chat.with_structured_output(Query)
    # raise NotImplementedError("T01: parse chain")
