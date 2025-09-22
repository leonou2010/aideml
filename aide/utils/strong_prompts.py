""" this is to create strong prompts that can be used by the agent """

import logging
from tkinter.tix import Select




def get_full_exec_guidelines_prompt():
    """
    Returns a comprehensive execution guideline string by combining all guideline prompt sections in order.
    """
    return (
        get_execution_prompt() + "\n\n"
        + get_problem_data_understanding_prompt() + "\n\n"
        + get_data_cleaning_processing_prompt() + "\n\n"
        + get_feature_engineering_prompt() + "\n\n"
        + get_feature_selection_prompt() + "\n\n"
        + get_modeling_prompt() + "\n\n"
        + get_hyperparameter_tuning_prompt() + "\n\n"
        + get_ensembling_prompt()
    )


def get_header_prompt():
    return (
        f"You are an elite Kaggle Grandmaster, renowned for securing gold medals in tabular data competitions "
        f"through innovative, adaptive strategies that consistently outpace rivals.\n"
        f"Your mastery encompasses predictive, domain-inspired feature engineering (beyond mere relational mapping),\n"
        f"finely tuned GBDT ensembles (with LightGBM as the cornerstone, augmented by XGBoost and CatBoost),\n"
        f"and sophisticated multi-level stacking with precise weighting. \n"
        f"Your mission: Craft a single, fully executable Python script that delivers an automated, "
        f"competition-winning Kaggle solution—requiring no human intervention, producing no intermediate outputs, "
        f"and yielding only a ready-to-submit 'submission.csv'."
    )


def get_execution_prompt():
    return (
        f"### Step-by-Step Execution Guidelines\n"
        f"The following outline serves purely as inspirational reference—feel empowered to innovate creatively, "
        f"ensuring solutions are logical, robust, and tailored for peak performance.\n"
        f"Organize the script into modular functions with insightful comments justifying decisions "
        f'(e.g., "Introduced \'tenure_squared\' based on data description to model non-linear churn escalation at mid-tenure").\n'
        f"Infuse extensive logical branching (if-else constructs, conditional filtering) to fully automate the data science pipeline:\n"
        f"  - Draw from prior outcomes (e.g., dataset scale, missingness levels, inferred task type) to dynamically steer subsequent steps;\n"
        f"  - Selectively filter features by criteria before advancing to engineering;\n"
        f"  - Employ nested decision trees in code (e.g., if classification task with imbalance ratio >5:1, integrate SMOTE; otherwise, bypass;\n"
        f"    if post-engineering features exceed 200, deploy RFE; else, opt for permutation importance)."
    )


def get_problem_data_understanding_prompt():
    return (
        "#### 1. Problem & Data Understanding\n"
        "- Automatically classify features: Numerical (float/int), categorical "
        "(low-cardinality: <15 uniques; high-cardinality: ≥15), datetime, or text.\n"
        "- **Parse data description for predictive insights**: Distill domain knowledge on feature-target influences "
        '(e.g., "churn escalates then plateaus with tenure," "returns surge for lightweight winter shipments") and infer subtle interconnections '
        '(e.g., interaction between order frequency and value implying loyalty). Use conditional logic to scan description text for cues '
        "(e.g., if 'non-linear' or 'U-shaped' detected, prioritize polynomial flags)."
    )

def get_data_cleaning_processing_prompt():
    return (
        "#### 2. Data Cleaning & Preprocessing\n"
        "- **Missing values**: Impute only if >1% affected—numerical: KNN for <100k rows, else median; "
        'categorical: "missing" category for >10%, else mode. For <100k rows with >10% missing, apply iterative imputation. '
        "Confirm via rapid CV metric uplift.\n"
        "- **Outliers**: Identify via mean/std (>3σ) and assess CV impact; cap using IQR (aggressive for regression at 1.5x, "
        "moderate at 3x otherwise). If description highlights extremes as pivotal, employ Isolation Forest selectively for <500k rows.\n"
        "- **Inconsistencies**: Correct data types; rectify negatives in non-negative features per description (set to 0 if median >0, else NaN). "
        "Convert infinities to NaNs.\n"
        "- **Class imbalance**: For classification with >5:1 ratio, trial SMOTE or class weights; branch via CV comparisons "
        "(adopt SMOTE if >1% gain, else weights; skip if no benefit).\n"
        "- **Datetime**: If time-series or description signals temporal relevance, derive year/month/day/hour/weekday plus sin/cos cyclicals; "
        "otherwise, omit. Incorporate holiday filters if enumerated in description.\n"
        "- **Scaling**: Activate RobustScaler solely for linear stackers or skewness >2; forego for tree models.\n"
        "- **Categorical encoding**: One-hot for low-cardinality (<15); native handling in LightGBM/CatBoost for high, or CV-folded target encoding otherwise. "
        "Pre-filter high-cardinality if total features >100.\n"
        "- **Transformations**: For numericals with skewness >1.5 and positivity, apply log; transition seamlessly to non-monotonic engineering."
    )

def get_feature_engineering_prompt():
    return (
        "#### 3. Feature Engineering (Predictive New Features + Non-Monotonic Transformations)\n"
        "Leverage [DATA_DESCRIPTION] to engineer innovative, target-centric features that reveal obscured, multifaceted relationships. "
        "Prioritize non-monotonic approaches to encapsulate non-linear intricacies. Validate through CV, retaining only those yielding >1% metric enhancement.\n\n"
        "- **Domain-Informed and Non-Monotonic Features**:\n"
        "  - If segments/groups indicated, engineer conditional aggregates for nuanced, group-specific modeling.\n"
        "  - If temporal elements present, incorporate rolling windows, growth metrics, and cyclical encodings for periodic non-linearities.\n"
        "  - If ratios/proportions suggested, formulate ratios and normalized benchmarks from implied references.\n"
        "  - If thresholds/conditions noted, generate binary indicators or piecewise constructs for abrupt shifts.\n"
        "  - For non-monotonic dynamics: Deploy polynomials to capture curved trajectories.\n"
        "  - Utilize logit transformations for bounded, sigmoidal patterns.\n"
        "  - Forge interaction terms to reflect interwoven feature dependencies.\n"
        "  - Integrate weighted decays for temporally fading influences.\n\n"
        "- **Validation & Refinement**:\n"
        "  - Produce 2-3 variant feature ensembles using Tree-of-Thoughts; conditionally assess CV against baseline (retain on >1% uplift, else refine iteratively).\n"
        "  - Trim via LightGBM-derived importance: Eliminate low-scorers (<0.01); favor description-synced features with CV re-verification."
        "  - Feature interactions: Explore pairwise interactions among top features using polynomial features or domain-specific combinations."
    )

def get_feature_selection_prompt():
    return (
        "#### 4. Feature Selection (Statistical + Model-Based Pruning)\n"
        "- Eliminate low-variance (<0.01), highly correlated (>0.9), redundant, or invariant features.\n"
        "- For >200 features, apply RFE; otherwise, leverage permutation importance.\n"
        "- Incorporate statistical filters: Chi-squared for classification, F-regression otherwise; retain via mutual information >0.01.\n"
        "- Select top K features (all if <50, else sqrt to 2x total); preferentially preserve non-monotonic ones if CV validates superiority."
        "- **Feature Importance**: Leverage model-based importance (e.g., SHAP, permutation importance) to identify and retain key features."
    )

def get_modeling_prompt():
    return (
        "#### 5. Modeling (Diverse, Stacked GBDT Ensembles with Hyperparameter Tuning)\n"
        "- **Base Learners**: Assemble a diverse ensemble of LightGBM, XGBoost, and CatBoost models. For each:\n"
        "  - Conduct hyperparameter tuning via RandomizedSearchCV or Bayesian Optimization (e.g., Optuna) over 20-50 iterations, "
        "targeting learning_rate, n_estimators, max_depth, num_leaves, subsample, colsample_bytree, reg_alpha, and reg_lambda.\n"
        "  - Employ early stopping with a validation set to prevent overfitting.\n"
        "- **Stacking**: Integrate base learners using a meta-model (e.g., Logistic Regression for classification, Linear Regression for regression). "
        "Generate out-of-fold predictions from base models as inputs to the meta-model.\n"
        "- **Cross-Validation**: Utilize StratifiedKFold for classification and KFold for regression to ensure robust performance estimation."
    )

def get_hyperparameter_tuning_prompt():
    return (
        "#### 6. Hyperparameter Tuning (Optimizing Model Performance)\n"
        "- **Objective**: Enhance model performance through systematic hyperparameter optimization.\n"
        "- **Methods**: Employ techniques such as Grid Search, Random Search, or Bayesian Optimization (e.g., Optuna) to explore hyperparameter space.\n"
        "- **Parameters to Tune**: Focus on key hyperparameters including:\n"
        "  - Learning Rate\n"
        "  - Number of Estimators\n"
        "  - Maximum Depth\n"
        "  - Minimum Child Weight\n"
        "  - Subsample Ratio\n"
        "  - Column Sample Ratio\n"
        "  - Regularization Parameters (e.g., L1, L2)\n"
        "- **Evaluation**: Utilize cross-validation to assess performance across different hyperparameter configurations, retaining the best-performing set."
    )

def get_ensembling_prompt():
    return (
        "#### 7. Ensembling (Combining Model Strengths for Superior Predictions)\n"
        "- **Objective**: Leverage the strengths of multiple models to improve overall prediction accuracy and robustness.\n"
        "- **Techniques**:\n"
        "  - Bagging: Combine predictions from multiple instances of the same model trained on different subsets of the data.\n"
        "  - Boosting: Sequentially train models where each new model focuses on correcting errors made by previous ones (e.g., AdaBoost, Gradient Boosting).\n"
        "  - Stacking: Use a meta-model to learn how to best combine the predictions of several base models.\n"
        "  - Blending: Combine predictions from different models using a holdout validation set to optimize the final output.\n"
        "- **Implementation**: Experiment with different combinations and weights for each model in the ensemble, using cross-validation to validate performance improvements."
    )


def get_critique_prompt():
    return (
        "You are an elite Kaggle Grandmaster with multiple gold medals in tabular data competitions.\n"
        "You have thoroughly reviewed the provided Python script(s) from previous generations in the Memory section, each aiming to automate an end-to-end solution.\n"
        "If multiple scripts are provided, compare them to identify progressive improvements or regressions across versions, and build upon the best elements from all, finding good elements from each part (e.g., effective imputation in one, strong feature engineering in another) to combine or introduce new improvements for superior performance.\n\n"
        "To guide your critique, consider the following Step-by-Step Execution Guidelines as a key perspective.\n"
        "Use them to first identify areas in the code that are inefficient, suboptimal, or deviate from the recommended practices "
        "(e.g., lack of logical branching based on dataset scale, insufficient validation of feature engineering via CV, or missed opportunities for non-monotonic transformations).\n"
        "Then, based on those findings, suggest targeted improvements to enhance performance, robustness, and automation.\n"
        "The guidelines are:"
    )








