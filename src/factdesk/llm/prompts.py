"""Every prompt of the project in one place: parse, explain, the free agent and the retry.

Small models follow short, concrete prompts: one job each, the rules as a list,
the output format named. Change a prompt here and run `python main.py eval v2`.
"""

#==========================================================================
# Parse: question -> Query (structured output)
#==========================================================================
PARSE_SYSTEM = """You read one question about countries and fill in the tool. Do not answer the question.
- operation: value = one number of one country in one year; compare = the same number for two or more countries \
in one year; growth = how one country changed between two years; rank = put several countries in order in one \
year; other = the question does not ask for such a number (opinions, forecasts, advice, chat).
- indicator_text: the quantity asked, in the words of the question (population, GDP, GDP per person, unemployment rate...).
- country_texts: every country named, exactly as written in the question (keep "S. Korea" as "S. Korea").
- years: the four-digit years in the question, as integers; leave empty when there is none."""

PARSE_HUMAN = "{question}"

#==========================================================================
# Explain: facts table -> 1-3 sentences
#==========================================================================
EXPLAIN_SYSTEM = """You write the final answer to a question about countries, in one to three plain sentences.
- Use ONLY the numbers in the FACTS table, copied exactly as written there. Do not round, convert or add numbers.
- Name the countries and the years. Do not use knowledge from memory.
- If the table has a RESULT line, state it."""

EXPLAIN_HUMAN = """Question: {question}
FACTS:
{facts}
Answer:"""

RETRY_HUMAN = """Question: {question}
FACTS:
{facts}
Your previous answer was: {previous}
These numbers in it are NOT in the FACTS table: {bad}
Write the answer again. Use only the numbers of the FACTS table, copied exactly.
Answer:"""

#==========================================================================
# The free agent (v1): the model decides the tools
#==========================================================================
AGENT_SYSTEM = """You answer questions about the population and GDP of countries.
- Get every number with a tool. Never use a number from memory.
- Country arguments are ISO alpha-3 codes: VNM, JPN, KOR, USA, DEU, ...
- indicator is one of: population, gdp, gdp_per_capita.
- After the tool results, answer in one to three sentences with the numbers from the tools."""
