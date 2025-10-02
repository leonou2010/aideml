""" this is to create strong prompts that can be used by the agent """

import logging
from tkinter.tix import Select




def get_full_exec_guidelines_prompt():
    """
    Returns a comprehensive execution guideline string by combining all guideline prompt sections in order.
    """
    return (
        get_execution_prompt() + "\n\n"
        + get_data_and_problem_analysis_prompt() + "\n\n"
        + get_data_cleaning_processing_prompt() + "\n\n"
        + get_feature_engineering_prompt() + "\n\n"
        + get_feature_selection_prompt() + "\n\n"
        + get_modeling_prompt() + "\n\n"
        + get_overfitting_check_prompt()
    )


# def get_header_prompt():
#     return (
#         f"You are an elite Kaggle Grandmaster, renowned for securing gold medals in tabular data competitions "
#         f"through innovative, adaptive strategies that consistently outpace rivals.\n"
#         f"Your mastery encompasses predictive, domain-inspired feature engineering (beyond mere relational mapping),\n"
#         f"finely tuned GBDT ensembles (with LightGBM as the cornerstone, augmented by XGBoost and CatBoost),\n"
#         f"and sophisticated multi-level stacking with precise weighting. \n"
#         f"Your mission: Craft a single, fully executable Python script that delivers an automated, "
#         f"competition-winning Kaggle solution—requiring no human intervention, producing no intermediate outputs, "
#         f"and yielding only a ready-to-submit 'submission.csv'."
#     )


def get_execution_prompt():
    """
    Empowers the LLM to think independently while providing a strong strategic framework with concrete examples.
    """
    return (
        "### Grandmaster's Strategic Framework\n"
        "The following principles are the foundation of winning solutions. Use them as your strategic guide, "
        "but you are the Grandmaster. **You have the authority to innovate, adapt, or even discard these guidelines "
        "if you determine a different approach will achieve a higher score.**\n\n"
        "1.  **Modular & Justified Code**: Write clean, functional code. Justify every critical decision with a concise comment.\n\n"
        "2.  **Dynamic & Adaptive Logic**: Your script must be intelligent and fully automated using logical branching (`if/else`). It must **dynamically steer its own process** by making decisions based on prior outcomes. For example, let the dataset's scale, missingness levels, or inferred task type determine which subsequent steps to take, such as selectively filtering features before engineering.\n\n"
        "3.  **Performance Above All**: Your ultimate goal is the best possible score. Every choice must be ruthlessly focused on maximizing performance while avoiding **over-engineering and overfitting**."
    )

def get_data_and_problem_analysis_prompt():
    """
    Prompt for the initial data exploration and problem definition.
    """
    return (
        "#### 1. Automated Data and Problem Analysis\n"
        "- **Infer the Task**: Automatically determine the problem type (e.g., regression, binary/multiclass classification) and identify the target variable.\n"
        "- **Data Profiling**: Programmatically analyze the features to understand their types (numeric, categorical, datetime), distributions, and cardinality.\n"
        "- **Strategic Insights**: If a data description is provided, scan it for crucial domain knowledge that can inform feature engineering and model selection."
    )

def get_data_cleaning_processing_prompt():
    """
    Prompt for adaptive data cleaning and preparation.
    """
    return (
        "#### 2. Data Preprocessing\n"
        "- **Smart Imputation**: Develop a context-aware strategy for handling missing values based on their volume and feature type.\n"
        "- **Intelligent Encoding**: Choose the most appropriate encoding method for categorical features based on their cardinality and the models you plan to use (e.g., one-hot encoding for low cardinality, target encoding for high cardinality).\n"
        "- **Handle Outliers and Skewness**: Identify and address outliers and highly skewed data in a way that benefits the model without losing important information."
    )


def get_feature_engineering_prompt():
    """
    Prompt for creating and selecting high-impact features.
    """
    return (
        "#### 3. High-Impact Feature Engineering and Selection\n"
        "- **Create Powerful Features**: Engineer new features that capture complex patterns and interactions. Focus on creating non-monotonic transformation features with high predictive power, such as polynomial terms, interaction features, or aggregations based on categorical groups.\n"
        "- **Prune for Performance**: Systematically select the most valuable features. Use techniques like model-based feature importance (e.g., from LightGBM) or permutation importance to eliminate noise and reduce model complexity."
    )


def get_feature_selection_prompt():
    """
    Guides the process of pruning features for a more robust model.
    """
    return (
        "#### 4. Feature Selection\n"
        "After creating features, systematically remove those that add more noise than signal. Use a model-based "
        "approach, like leveraging feature importance from a baseline LightGBM model, to identify and discard "
        "low-importance, redundant, or highly correlated features."
    )


def get_modeling_prompt():
    return (
        "#### 5. Modeling (Diverse, Stacked GBDT Ensembles with Hyperparameter Tuning)\n"
        "- **Base Learners**: Assemble a diverse ensemble of LightGBM, XGBoost, and CatBoost models. For each:\n"
        "  - Conduct hyperparameter tuning via RandomizedSearchCV or Bayesian Optimization (e.g., Optuna), "
        "- **Stacking**: Integrate base learners using a meta-model. "
        "Generate out-of-fold predictions from base models as inputs to the meta-model.\n"
        "- **Cross-Validation**: Utilize StratifiedKFold for classification and KFold for regression to ensure robust performance estimation."
    )

def get_overfitting_check_prompt():
    """
    Returns an ultra-direct, non-negotiable prompt for the overfitting check.
    """
    return (
        "#### 6. Mandatory Overfitting Check \n"
        "EXTREMELY IMPORTANT: Before creating the submission file, you must implement a function to check and quantify model overfitting. \n"
        "Compare the final in-sample score against the out-of-sample cross-validation score and print 'Warning: Model may be overfitting.', "
        "if a significant performance gap is detected. This check is a mandatory diagnostic and must be performed."
    )


def get_critique_prompt():
    return (
        "You are an elite Kaggle Grandmaster with multiple gold medals in tabular data competitions.\n"
        "You have thoroughly reviewed the provided Python script(s) from previous generations in the Memory section, each aiming to automate an end-to-end solution.\n"
        "If multiple scripts are provided, compare them to identify progressive improvements or regressions across versions, and build upon the best elements from all, finding good elements from each part (e.g., effective imputation in one, strong feature engineering in another) to combine or introduce new improvements for superior performance.\n\n"
        "To guide your critique, consider the following Step-by-Step Execution Guidelines as a key perspective.\n"
        "Use them to first identify areas in the code that are inefficient, suboptimal, or deviate from the recommended practices. \n"
        "Then, based on those findings, suggest targeted improvements to enhance performance, robustness, and automation.\n"
        "The guidelines are:"
    )








