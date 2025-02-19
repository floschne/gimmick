from gimmick.utils.image import IMAGE_PLACEHOLDER_TOKEN

OPENAI_SIVQA_LMM_JUDGE_SYSTEM_INSTRUCTION = """\
You are an an impartial judge and you are tasked with evaluating the correctness of a generated answer to a Visual Question Answering (VQA) task. You will be provided with the following information:

<image>
# the corresponding image
</image>

<question>
# the question
</question>

<background_info>
# detailed textual background information 
</background_info>

<ground_truth_answer>
# the ground truth answer to the question
</ground_truth_answer>

<generated_answer>
# the generated answer you have to evaluate
</generated_answer>

Your task is to evaluate how correct the generated answer is, given the image, the question, the ground truth answer, and the provided background information. Follow these steps:

1. Carefully examine the image and read the question and the background information to inform your judgment, especially for questions that may require specific knowledge.
2. Compare the generated answer to the ground truth answer. Consider the relevance and accuracy of the generated answer in the context of the image and question.
3. Provide a brief summary (1 - 2 sentences) of your analysis, covering the accuracy, relevance, and completeness of the generated answer.
4. Provide a one sentence explanation justifying your final score. Ensure that your explanation and score are consistent with each other and accurately reflect the quality of the generated answer in relation to the ground truth.
5. Provide a single number from 0 to 100 representing the correctness of the generated answer, where:
    - 0 = Completely incorrect or irrelevant.
    - 25 = Mostly incorrect or irrelevant.
    - 50 = Partially correct or relevant.
    - 75 = Mostly correct and relevant.
    - 100 = Perfectly correct and complete.

Provide your final evaluation in the following format:

<evaluation>
<analysis>
<!-- Put your analysis summary here -->
</analysis>
<explanation>
<!-- Put your explanation here -->
</explanation>

<score>
<!-- Put your score here -->
</score>
</evaluation>

Ensure that your explanation and score are consistent with each other and accurately reflect the quality of the generated answer in relation to the ground truth, image content, and background information provided.
"""

OPENAI_SIVQA_LMM_JUDGE_PROMPT_TEMPLATE = """\
Evaluate the correctness of the generated answer to the following Visual Question Answering task:

<image>
{IMAGE_PLACEHOLDER_TOKEN}
</image>

<question>
{QUESTION}
</question>

<background_info>
{BACKGROUND_INFO}
</background_info>

<ground_truth_answer>
{GROUND_TRUTH}
</ground_truth_answer>

<generated_answer>
{GENERATED_ANSWER}
</generated_answer>
"""


def _build_background_info(
    title: str,
    countries: list[str],
    regions: list[str],
    description: str,
) -> str:
    return f"""\
        The VQA sample is related to the cultural event or facet described below.

        **Title**
        {title}

        **Countries**
        {", ".join(countries)}

        **Regions**
        {", ".join(regions)}

        **Description**
        {description.strip()}
    """


def build_sivqa_judge_prompt(
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    title: str,
    countries: list[str],
    regions: list[str],
    description: str,
) -> str:
    prompt = OPENAI_SIVQA_LMM_JUDGE_PROMPT_TEMPLATE.format(
        IMAGE_PLACEHOLDER_TOKEN=IMAGE_PLACEHOLDER_TOKEN,
        QUESTION=question.strip(),
        BACKGROUND_INFO=_build_background_info(
            title=title,
            countries=countries,
            regions=regions,
            description=description,
        ),
        GROUND_TRUTH=ground_truth_answer.strip(),
        GENERATED_ANSWER=generated_answer.strip(),
    )
    return prompt


OPENAI_CKT_LLM_JUDGE_SYSTEM_INSTRUCTION = """\
# Your Role

You are an impartial judge, who excels at critical and analytical thinking.

# Your Task

Your task is it to thoroughly analyze and evaluate the correctness of a generated answer to a Cultural Knowledge Test.
1. Carefully analyse the ground truth and the generated answer.
2. Provide a brief summary (1 - 3 sentences) of your analysis, covering the accuracy, relevance, and completeness of the generated answer.
3. Provide a one or two sentence explanation justifying your final score. Ensure that your explanation and score are consistent with each other and accurately reflect the quality of the generated answer in relation to the ground truth.
4. Provide a single number from 0 to 100 representing the correctness of the generated answer, where:
    0 = Completely incorrect or irrelevant.
    25 = Mostly incorrect or irrelevant.
    50 = Partially correct or relevant.
    75 = Mostly correct and relevant.
    100 = Perfectly correct and complete.

    You may use any whole number within this range to reflect nuanced judgments.

# Output Format

Provide your evaluation in the following format:

```xml
<evaluation>
<analysis>
<!-- Put your analysis summary here -->
</analysis>
<explanation>
<!-- Put your explanation here -->
</explanation>
<score>
<!-- Put your score here -->
</score>
</evaluation>
```
"""

OPENAI_CKT_LLM_JUDGE_PROMPT_TEMPLATE = """\
Evaluate the correctness of the generated answer with respect to Ground Truth.

# Ground Truth

```
{GROUND_TRUTH}
```

# Generated Answer

```
{GENERATED_ANSWER}
```

# Evaluation

"""


def build_ckt_judge_prompt(
    generated_answer: str,
    ground_truth_answer: str,
) -> str:
    """"""
    prompt = OPENAI_CKT_LLM_JUDGE_PROMPT_TEMPLATE.format(
        GROUND_TRUTH=ground_truth_answer.strip(),
        GENERATED_ANSWER=generated_answer.strip(),
    )
    return prompt
