# Sources and licences

Fetched 2026-11-16 from the GitHub organisation `datasets`, branch `main`, at the pinned commits below.
`python scripts/fetch_data.py` downloads the same bytes again from
`https://raw.githubusercontent.com/datasets/<repo>/<SHA>/<path>` into `data/raw/`.
The files in `data/raw/` are data only; nothing in them is ever executed or followed as an instruction.

## 1. population

- Repository: https://github.com/datasets/population
- Pinned SHA: `d4b1a4a9a17e2123baf0ef119de05bceced1ab1b`
- File used: `data/raw/population/data/population.csv` (columns `Country Name,Country Code,Year,Value`;
  17,394 rows; years 1960-2025; 264 codes)
- Upstream: World Bank indicator SP.POP.TOTL (total population). The upstream README lists the World Bank's own
  sources: UN Population Division, UN Statistical Division, national statistical offices, Eurostat, SPC, U.S. Census
  Bureau.
- Licence: `datapackage.json` says `ODC-PDDL-1.0` (Open Data Commons Public Domain Dedication and License 1.0);
  the README says the data package is licensed by its maintainers under the PDDL. That is the maintainers' licence;
  the README names the World Bank as the origin of the numbers.
- Attribution: "Population: World Bank (indicator SP.POP.TOTL), packaged by the `datasets/population` project."

## 2. gdp

- Repository: https://github.com/datasets/gdp
- Pinned SHA: `36d761dc5617982d3e50b213e02b9a1b47c4e958`
- File used: `data/raw/gdp/data/gdp.csv` (columns `Country Name,Country Code,Year,Value`; 13,979 rows;
  years 1960-2023; 262 codes)
- Upstream: World Bank indicator NY.GDP.MKTP.CD (GDP in current US dollars); World Bank and OECD national
  accounts data.
- Licence: `CC-BY-4.0` (https://creativecommons.org/licenses/by/4.0/). The README states the underlying data
  originates from the World Bank, which publishes its open data under CC BY 4.0.
- Attribution (required): "GDP: World Bank - World Development Indicators (NY.GDP.MKTP.CD), packaged by the
  `datasets/gdp` project, CC BY 4.0." Our prepared files are an adapted subset (years from 1990, aggregates
  removed, GDP per person derived by us) and say so here.

## 3. What is NOT used

The `datasets/country-codes` repository is not used and not shipped: its metadata says some columns (UN terminology
names) are restricted and others have an unclear licence. Country names and aliases come from the two files above plus
a short hand-written alias table in `configs/app.toml`.

## What `scripts/prepare_data.py` makes from the raw files

| File | Content |
|---|---|
| `data/facts.csv` | `code,year,population,gdp_usd` for years >= 1990; an empty cell means the source has no value |
| `data/countries.json` | `[{code, name, aliases[]}]`: 216 countries or territories plus `WLD` (World) |
| `data/indicators.json` | `population`, `gdp`, `gdp_per_capita` with unit, description, caveats, synonyms |
| `data/notes.json` | 8 short glossary notes (the "what does this number mean" documents) |
| `data/docs/*.json` | the knowledge base documents: one per country, one per indicator, one per note |

Aggregates (regions, income groups, World Bank lending groups, demographic-dividend groups) are removed with the
explicit code list in `configs/app.toml` (`[facts] aggregate_codes`, hand-checked against the 264 codes of
`population.csv`). `WLD` (World) is the only aggregate kept. `XKX` (Kosovo, a World Bank code that is not an ISO
code) is kept. Territories such as `HKG`, `PRI` or `GIB` stay; `GIB`, `PRK` and `VGB` have population but no GDP at all.
Years 2024 and 2025 have population but no GDP: asking for them is a real "no data" case.
