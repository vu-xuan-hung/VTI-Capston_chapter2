# FactDesk evaluation

## Runs

The saved runs are dated 20 November 2026. I used `python main.py eval v1` and `python main.py eval v2` on the 12 development cases with local Qwen2.5-1.5B-Instruct. The Windows machine has an Intel Core i5-12450HX CPU and 19.7 GiB of usable RAM. I used `python main.py compare v1 v2` on the saved runs and ran `python main.py --config runtime/factdesk-langflow.toml flow-compare` with the same Qwen model and Chroma collection over HTTP.

## Results

| Metric | v1 (free agent) | v2 (graph) |
|---|---:|---:|
| Entities correct | 75.0% | 100% |
| Exact result | 75.0% | 91.7% |
| Answer numbers verified | 66.7% | 100% |
| Correct status | 91.7% | 91.7% |
| Mean seconds per case | 118.22 | 62.01 |
| Model calls per case | 2.00 | 1.75 |

v2 improved on D02, D04, D07, D08, D10 and D11, regressed on D05 and tied on five cases. Accurate entity linking and constrained calculation raised the result score. The overall status score stayed the same because v2 fixed D11 and regressed on D05.

## Errors found

| Case | Observed answer | Expected | Step and trace |
|---|---|---|---|
| D02, v1 | South Korea's 2019 GDP was $165 trillion. | $1,651,422,932,447.77, about $1.65 trillion. | The final wording changed the scale after a correct tool response. `runtime/traces/20261120-001230-v1-D02.json:50` and line 54. |
| D04, v1 | Correct GDP-per-person figures and ratio, followed by an unsupported claim of 83%. | Japan's GDP per person was $40,040.77, South Korea's was $31,721.30 and the Japan/South Korea ratio was 1.26. | The final wording added a percentage the tool never returned and treated GDP per person as earnings. `runtime/traces/20261120-001633-v1-D04.json:55`. |
| D05, v2 | The graph gave an `out_of_scope` response. | Indonesia had more people than Pakistan in 2010. The ratio was 1.24. | The parse step selected `value` for a two-country comparison. `runtime/traces/20261120-003742-v2-D05.json:88`. |
| D07, v1 | Poland's 2020 population was 37,515,748. | Poland's population decreased by 1.94% from 2000 to 2020. | The agent chose `get_value` with one endpoint. `runtime/traces/20261120-002231-v1-D07.json:98`. |

## What verification checked

None of the saved v2 explanations triggered a retry or template. D02 shows the closest contrast. The verifier accepted v2's $1.65 trillion on attempt one, and replaying v1's $165 trillion wording against the same facts flagged that amount. The last digit of $1.65 trillion represents $0.01 trillion, so rounding permits a half-step of $0.005 trillion around the stated amount. The table value falls inside that interval.

## Langflow slice

In `results/flow_compare.json`, both paths included “goods and services” in the GDP definition (F1). For inflation (F2), the Python chain said the GDP was not adjusted, missed the term “current,” and the flow said it did not know. Both paths explained the GDP-per-person calculation (F3) and missed the literal phrase “divided by” because they wrote “dividing ... by.”

For Vietnam's 2015 population (F4), the graph returned 92,823,254 people and the flow said it did not know. The flow's unverified-number list is empty because it supplied no number. The seven visual components retrieve notes and generate text. They provide no table lookup tool, calculation step or numerical verification.

## Limits and tool choice

GDP uses current US dollars and excludes both inflation adjustment and PPP. GDP ends in 2023 and the population series ends in 2025. The scores cover 12 development questions with one small model. The verifier accepts rounding and checks numerical membership, so a correct number attached to the wrong country or direction can pass.

I would use the free agent to draft copy when a researcher can check the source table. It answered D01 correctly with 92,823,254 people. I would use v2 for newsroom figures. It checks numbers against computed facts and averaged 62.01 seconds and 1.75 model calls per case, versus 118.22 seconds and 2.00 for v1. I would still check the comparison direction and units before publication.
