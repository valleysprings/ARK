"""
Evaluation Metrics Module

This module provides a comprehensive collection of evaluation metrics for various NLP tasks,
including question answering, retrieval, classification, code similarity, and summarization.

The module supports both English and Chinese text evaluation with specialized scoring functions
for different task types:
- QA F1 Score: Token-level F1 scores for question answering tasks
- ROUGE Score: Recall-Oriented Understudy for Gisting Evaluation for summarization
- Classification Score: Accuracy metrics for classification tasks
- Retrieval Score: Precision metrics for information retrieval
- Code Similarity Score: Fuzzy matching for code completion tasks

Available Functions:
    - qa_f1_score: English QA F1 score with normalization
    - qa_f1_score_zh: Chinese QA F1 score with jieba tokenization
    - rouge_score: ROUGE-L F1 score for English text
    - rouge_score_zh: ROUGE-L F1 score for Chinese text
    - classification_score: Classification accuracy with fuzzy matching
    - retrieval_score: Retrieval accuracy for English text
    - retrieval_zh_score: Retrieval accuracy for Chinese text
    - code_sim_score: Code similarity using fuzzy ratio
    - count_score: Counting task accuracy
    - scorer: Batch scoring function for datasets

Dataset Mappings:
    - DATASET2METRIC: Maps dataset names to their appropriate metric functions
    - DATASET2PROMPT: Template prompts for different datasets
    - DATASET2MAXNEWTOKENS: Maximum generation tokens per dataset
    - DATASET2CATEGORY: Dataset categorization
"""

import re
import string
import difflib
from typing import List, Dict, Any
from collections import Counter

import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge


def normalize_answer(s: str) -> str:
    """
    Normalize English answer text by lowercasing, removing punctuation, articles, and extra whitespace.

    This function applies a series of text cleaning operations commonly used in QA evaluation:
    1. Converts text to lowercase
    2. Removes articles (a, an, the)
    3. Removes punctuation
    4. Normalizes whitespace

    Args:
        s: Input string to normalize

    Returns:
        Normalized string with lowercase, no articles, no punctuation, and normalized whitespace

    Example:
        >>> normalize_answer("The quick brown fox!")
        "quick brown fox"
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """
    Normalize Chinese answer text by lowercasing, removing punctuation and extra whitespace.

    This function handles both English and Chinese punctuation marks for proper text cleaning.
    Unlike English normalization, it doesn't remove articles as Chinese doesn't use them.

    Args:
        s: Input string to normalize (Chinese text)

    Returns:
        Normalized string with lowercase, no punctuation, and no whitespace

    Example:
        >>> normalize_zh_answer("你好，世界！")
        "你好世界"
    """
    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction: List[str], ground_truth: List[str]) -> float:
    """
    Compute token-level F1 score between prediction and ground truth token lists.

    F1 score is the harmonic mean of precision and recall:
    - Precision = (common tokens) / (prediction tokens)
    - Recall = (common tokens) / (ground truth tokens)
    - F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        prediction: List of tokens from the prediction
        ground_truth: List of tokens from the ground truth

    Returns:
        F1 score as a float between 0 and 1

    Example:
        >>> f1_score(["the", "cat", "sat"], ["the", "cat"])
        0.8
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute F1 score for English question answering tasks.

    This function normalizes both prediction and ground truth using English normalization
    rules (removing articles, punctuation, etc.) and then computes token-level F1.

    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        F1 score as a float between 0 and 1

    Example:
        >>> qa_f1_score("The answer is 42", "42")
        1.0
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_score_zh(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute F1 score for Chinese question answering tasks.

    This function uses jieba for Chinese word segmentation and applies Chinese-specific
    normalization before computing token-level F1 score.

    Args:
        prediction: Predicted answer string (Chinese)
        ground_truth: Ground truth answer string (Chinese)
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        F1 score as a float between 0 and 1

    Example:
        >>> qa_f1_score_zh("答案是42", "42")
        0.67
    """
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute ROUGE-L F1 score for text generation tasks.

    ROUGE-L measures the longest common subsequence between prediction and ground truth.
    This metric is commonly used for summarization and text generation evaluation.

    Args:
        prediction: Generated text
        ground_truth: Reference text
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        ROUGE-L F1 score as a float between 0 and 1, or 0.0 if computation fails

    Example:
        >>> rouge_score("The cat sat on the mat", "A cat sat on a mat")
        0.85
    """
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_score_zh(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute ROUGE-L F1 score for Chinese text generation tasks.

    This function uses jieba to segment Chinese text into words before computing
    ROUGE score, which is necessary for proper evaluation of Chinese text.

    Args:
        prediction: Generated Chinese text
        ground_truth: Reference Chinese text
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        ROUGE-L F1 score as a float between 0 and 1

    Example:
        >>> rouge_score_zh("这是一个测试", "这是测试")
        0.75
    """
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute classification accuracy with exact and fuzzy matching.

    This function first attempts exact matching of class names in the prediction.
    If no exact match is found, it falls back to fuzzy string matching to find
    the most similar class name.

    Args:
        prediction: Model's predicted class/label
        ground_truth: True class/label
        **kwargs: Must contain 'all_classes' - list of all possible class names

    Returns:
        Score as a float:
        - 1.0 / len(matches) if exact match found (to handle multiple matches)
        - 1.0 if fuzzy match is correct
        - 0.0 otherwise

    Example:
        >>> classification_score("The answer is: sports", "sports", all_classes=["sports", "politics"])
        1.0
    """
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if em_match_list != 0:
        if ground_truth in em_match_list:
            score = (1.0 / len(em_match_list))
        else:
            score = 0.0
    else:
        best_match = None
        highest_similarity = 0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == ground_truth)
    return score


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute retrieval accuracy for English paragraph retrieval tasks.

    This function extracts paragraph numbers from both prediction and ground truth,
    then computes the precision of retrieved paragraphs.

    Args:
        prediction: Model's prediction containing paragraph numbers
        ground_truth: Ground truth string containing the correct paragraph (format: "Paragraph N")
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        Precision score: (correct retrievals) / (total retrievals), or 0.0 if no numbers found

    Example:
        >>> retrieval_score("The answer is in Paragraph 3", "Paragraph 3")
        1.0
    """
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute retrieval accuracy for Chinese paragraph retrieval tasks.

    Similar to retrieval_score but for Chinese text using "段落N" format.

    Args:
        prediction: Model's prediction containing paragraph numbers (Chinese)
        ground_truth: Ground truth string containing the correct paragraph (format: "段落N")
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        Precision score: (correct retrievals) / (total retrievals), or 0.0 if no numbers found

    Example:
        >>> retrieval_zh_score("答案在段落3中", "段落3")
        1.0
    """
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute counting task accuracy.

    This function extracts all numbers from the prediction and computes the precision
    of correct counts compared to the ground truth.

    Args:
        prediction: Model's prediction containing numbers
        ground_truth: Ground truth count value
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        Precision score: (correct counts) / (total counts), or 0.0 if no numbers found

    Example:
        >>> count_score("There are 5 unique paragraphs", "5")
        1.0
    """
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Compute code similarity score using fuzzy string matching.

    This function extracts the first line of actual code (excluding comments and markdown)
    from the prediction and computes fuzzy similarity with the ground truth.

    Args:
        prediction: Generated code string
        ground_truth: Reference code string
        **kwargs: Additional keyword arguments (unused, for API compatibility)

    Returns:
        Similarity ratio as a float between 0 and 1

    Example:
        >>> code_sim_score("```python\\nprint('hello')\\n```", "print('hello')")
        1.0
    """
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)


def scorer(
    dataset: str,
    predictions: List[str],
    answers: List[List[str]],
    all_classes: List[str]
) -> float:
    """
    Batch scoring function that applies the appropriate metric for a given dataset.

    This function scores multiple predictions against multiple ground truths,
    taking the maximum score across all ground truth options for each prediction.

    Args:
        dataset: Dataset name (must be a key in DATASET2METRIC)
        predictions: List of model predictions
        answers: List of ground truth answer lists (multiple references per prediction)
        all_classes: List of all possible classes (for classification tasks)

    Returns:
        Average score across all predictions, scaled to 0-100 and rounded to 2 decimals

    Example:
        >>> scorer("hotpotqa", ["Paris"], [["Paris", "paris"]], [])
        100.0
    """
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, DATASET2METRIC[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


# Dataset to metric function mapping
DATASET2METRIC: Dict[str, Any] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_score_zh,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_score_zh,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_score_zh,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Dataset to maximum new tokens mapping
DATASET2MAXNEWTOKENS: Dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

# Dataset to category mapping
DATASET2CATEGORY: Dict[str, str] = {
    "narrativeqa": "EN Single-Doc QA",
    "qasper": "EN Single-Doc QA",
    "multifieldqa_en": "EN Single-Doc QA",
    "multifieldqa_zh": "CN Single-Doc QA",
    "hotpotqa": "EN Multi-Doc QA",
    "2wikimqa": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "dureader": "CN Multi-Doc QA",
    "gov_report": "EN Summarization",
    "qmsum": "EN Summarization",
    "multi_news": "EN Summarization",
    "vcsum": "CN Summarization",
    "trec": "EN Few-Shot Learning",
    "triviaqa": "EN Few-Shot Learning",
    "samsum": "EN Few-Shot Learning",
    "lsht": "CN Few-Shot Learning",
    "passage_retrieval_en": "EN Synthetic Task",
    "passage_count": "EN Synthetic Task",
    "passage_retrieval_zh": "CN Synthetic Task",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion",
}
