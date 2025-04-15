import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
import re
import nltk
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- NLTK Stopwords --- 
def download_stopwords():
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

# --- Data Loading and Parsing --- 
def load_and_parse_data(filepath):
    solutions_raw = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.lower().replace("nan", "").strip()
                if line:
                    solutions_raw.append(line)
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None

    solution_str = []
    objectives = []
    for dd in solutions_raw:
        parts = dd.rsplit(' ', 1)
        if len(parts) == 2:
            strs = parts[0].strip()
            label_part = parts[1].strip()
        else:
            match = re.search(r'([\s,]*[01](?:,\s*[01]){5})$', dd)
            if match:
                label_part = match.group(1).strip()
                strs = dd[:-len(label_part)].strip()
            else:
                continue

        try:
            int_numbers = [int(nn) for nn in re.findall(r'\b[01]\b', label_part)]
            if len(int_numbers) == 6:
                strs_cleaned = strs.replace(".", "")
                solution_str.append(strs_cleaned)
                objectives.append(np.array(int_numbers))
        except ValueError:
            continue
        except Exception as e:
            continue

    if not solution_str or not objectives:
        print(f"Warning: No valid solutions with 6 labels parsed from {filepath}")
        return None, None

    return solution_str, np.array(objectives)

def create_dataframe(solution_list, objective_array, columns):
    if solution_list is None or objective_array is None:
        return None
    try:
        df = pd.DataFrame({'solution': solution_list})
        if objective_array.shape[1] != len(columns):
            print(f"Error: Mismatch between number of objective columns ({objective_array.shape[1]}) and required column names ({len(columns)})")
            return None
        ob_df = pd.DataFrame(objective_array, columns=columns)
        df = pd.concat([df, ob_df], axis=1)
        return df
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None

# --- Data Visualization --- 
def plot_qmood_counts(dataframe, columns, system_name, save_dir):
    if dataframe is None or dataframe.empty:
        print("DataFrame is empty or None, skipping QMOOD count plot.")
        return
    
    valid_columns = [col for col in columns if col in dataframe.columns]
    if not valid_columns:
        print("No valid QMOOD columns found in DataFrame, skipping QMOOD count plot.")
        return
    
    df_qmood = dataframe[valid_columns]
    counts = []
    for col in valid_columns:
        counts.append((col, df_qmood[col].sum()))
    df_stats = pd.DataFrame(counts, columns=['QMOOD', '# of solutions'])

    plt.figure(figsize=(8, 5))
    sns.barplot(x='QMOOD', y='# of solutions', data=df_stats, ax=plt.gca(), palette="viridis")
    plt.title(f"Number of solutions improving each QMOOD ({system_name})")
    plt.ylabel('# of Solutions')
    plt.xlabel('QMOOD')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{system_name}_num-sol-per-qmood.png')
    try:
        plt.savefig(save_path)
        print(f"Saved QMOOD count plot to {save_path}")
    except Exception as e:
        print(f"Error saving QMOOD count plot: {e}")
    plt.close()

def plot_improved_qmood_distribution(dataframe, columns, system_name, save_dir):
    if dataframe is None or dataframe.empty:
        print("DataFrame is empty or None, skipping improved QMOOD distribution plot.")
        return
    
    valid_columns = [col for col in columns if col in dataframe.columns]
    if not valid_columns:
        print("No valid QMOOD columns found, skipping improved QMOOD distribution plot.")
        return
    
    rowsums = dataframe[valid_columns].sum(axis=1)
    improved_rowsums = rowsums[rowsums > 0]
    if improved_rowsums.empty:
        print("No solutions found improving any QMOODs. Skipping distribution plot.")
        return
    
    x = improved_rowsums.value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=x.index, y=x.values, color='skyblue')
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.title(f"Number of Solutions Improving 1 or more QMOODs ({system_name})")
    plt.ylabel('# of Solutions', fontsize=12)
    plt.xlabel('# of improved QMOODs', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{system_name}_num-solutions-improving-ge1-qmood.png')
    try:
        plt.savefig(save_path)
        print(f"Saved improved QMOOD distribution plot to {save_path}")
    except Exception as e:
        print(f"Error saving QMOOD distribution plot: {e}")
    plt.close()

# --- Feature Importance --- 
def plot_top_features(coef, names, category, system_name, save_dir, top_n=20):
    if len(coef) != len(names):
        print(f"Warning: Coef ({len(coef)})/Names ({len(names)}) length mismatch for {category}. Skipping plot.")
        return
    
    if len(coef) == 0:
        print(f"Warning: No coefficients provided for {category}. Skipping plot.")
        return
    
    try:
        abs_coef_names = sorted(zip(np.abs(coef), names), key=lambda x: x[0], reverse=True)
        top_indices = [names.index(name) for _, name in abs_coef_names[:top_n]]
        
        top_names = [names[i] for i in top_indices]
        top_coef = [coef[i] for i in top_indices]
        
        plot_data = sorted(zip(top_coef, top_names), key=lambda x: x[0])
        plot_coef_sorted, plot_names_sorted = zip(*plot_data)

    except Exception as e:
        print(f"Error sorting or selecting top features for {category}: {e}. Skipping plot.")
        return

    plt.figure(figsize=(10, max(5, len(plot_names_sorted) * 0.4)))
    colors = ['red' if c < 0 else 'blue' for c in plot_coef_sorted]
    plt.barh(range(len(plot_names_sorted)), plot_coef_sorted, align='center', color=colors)
    plt.yticks(range(len(plot_names_sorted)), plot_names_sorted)
    plt.xlabel("Coefficient Value (Importance)")
    plt.title(f"Top {len(plot_names_sorted)} Important Features (by magnitude) for {category} ({system_name})")
    plt.axvline(0, color='grey', lw=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{system_name}_top-{top_n}-features-{category}.png')
    try:
        plt.savefig(save_path)
        print(f"Saved top features plot for {category} to {save_path}")
    except Exception as e:
        print(f"Error saving top features plot: {e}")
    plt.close()

# --- Model Training and Evaluation --- 
def train_evaluate_plot(category, X_train, y_train, X_test, y_test, system_name, save_dir):
    print(f"\n--- Processing QMOOD: {category} ---")
    stop_words_list = list(stopwords.words('english'))

    SVC_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words_list, ngram_range=(1, 2), min_df=3, max_df=0.9)),
        ('clf', OneVsRestClassifier(LinearSVC(max_iter=5000, C=0.5, class_weight='balanced', dual='auto', random_state=42), n_jobs=-1)),
    ])

    results = {}
    if y_train.nunique() < 2:
        print(f"Skipping {category}: Only one class value present in training data.")
        return None

    try:
        print(f"Training model for {category}...")
        SVC_pipeline.fit(X_train, y_train)

        print(f"Evaluating model for {category}...")
        prediction = SVC_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        hamming = hamming_loss(y_test, prediction)
        f1_micro = f1_score(y_test, prediction, average='micro', zero_division=0)
        f1_macro = f1_score(y_test, prediction, average='macro', zero_division=0)

        print(f'Test Metrics for {category}:')
        print(f'  Accuracy:    {accuracy:.4f}')
        print(f'  Hamming Loss:{hamming:.4f}')
        print(f'  F1 (Micro):  {f1_micro:.4f}')
        print(f'  F1 (Macro):  {f1_macro:.4f}')
        results = {'accuracy': accuracy, 'hamming': hamming, 'f1_micro': f1_micro, 'f1_macro': f1_macro}

        classifier = SVC_pipeline.named_steps['clf']
        vectorizer = SVC_pipeline.named_steps['tfidf']
        try:
            feature_names = list(vectorizer.get_feature_names_out())
        except AttributeError:
            feature_names = list(vectorizer.get_feature_names())

        if hasattr(classifier.estimators_[0], 'coef_'):
            coef = classifier.estimators_[0].coef_[0]
            plot_top_features(coef, feature_names, category, system_name, save_dir)
        else:
            print(f"Classifier for {category} lacks 'coef_' attribute.")

    except Exception as e:
        print(f"Error during training/evaluation/plotting for {category}: {e}")
        results = None

    return results

# --- Main Execution --- 
def main(system_name):
    download_stopwords()

    # Config
    processed_data_dir = "processed_data"
    figs_dir = "figs"
    input_filename = f"FinalSolutions-multilabel-{system_name}.txt"
    input_filepath = os.path.join(processed_data_dir, input_filename)
    output_fig_dir = os.path.join(figs_dir, system_name)
    os.makedirs(output_fig_dir, exist_ok=True)
    qmood_cols = ["Effectiveness", "Extendibility", "Flexibility", "Functionality", "Reusability", "Understandability"]

    # Load Data
    print(f"=== Loading data for {system_name} ===")
    solution_list, objective_array = load_and_parse_data(input_filepath)
    df = create_dataframe(solution_list, objective_array, qmood_cols)

    if df is None or df.empty:
        print("Exiting: Failed to load or parse data, or DataFrame is empty.")
        return

    print(f"Loaded DataFrame shape: {df.shape}")

    # Visualize Data
    print(f"\n=== Visualizing data for {system_name} ===")
    plot_qmood_counts(df.copy(), qmood_cols, system_name, output_fig_dir)
    plot_improved_qmood_distribution(df.copy(), qmood_cols, system_name, output_fig_dir)

    # Split Data
    print(f"\n=== Splitting data for {system_name} ===")
    try:
        if 'solution' not in df.columns:
            print("Error: 'solution' column missing. Cannot split.")
            return
        valid_qmood_cols = [col for col in qmood_cols if col in df.columns]
        if not valid_qmood_cols:
            print("Error: No QMOOD columns found for splitting/stratification.")
            return

        try:
            train, test = train_test_split(df, random_state=42, test_size=0.30, shuffle=True, stratify=df[valid_qmood_cols])
            print("Using stratified split for train/test data.")
        except ValueError:
            print("Warning: Stratified split failed. Using non-stratified split.")
            train, test = train_test_split(df, random_state=42, test_size=0.30, shuffle=True)

        X_train = train['solution']
        X_test = test['solution']
        print(f"Training data size: {len(X_train)}")
        print(f"Test data size: {len(X_test)}")
    except Exception as e:
        print(f"Error during data splitting: {e}. Exiting.")
        return

    # Train and Evaluate Models
    print(f"\n=== Training and evaluating models for {system_name} ===")
    all_results = {}
    for category in qmood_cols:
        if category in train.columns and category in test.columns:
            train_category = train[category]
            test_category = test[category]
            category_results = train_evaluate_plot(category, X_train, train_category, X_test, test_category, system_name, output_fig_dir)
            all_results[category] = category_results
        else:
            print(f"Warning: Category '{category}' not found in DataFrame.")
            all_results[category] = None

    # Print Summary
    print(f"\n=== Evaluation Summary for {system_name} ===")
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        results_df = pd.DataFrame.from_dict(valid_results, orient='index')
        print(results_df.round(4))
        summary_path = os.path.join(output_fig_dir, f'{system_name}_evaluation_summary.csv')
        try:
            results_df.to_csv(summary_path)
            print(f"\nSaved evaluation summary to {summary_path}")
        except Exception as e:
            print(f"\nError saving evaluation summary: {e}")
    else:
        print("No valid evaluation results to summarize.")

    print(f"\n=== Processing for {system_name} complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification pipeline for refactoring suggestions.')
    parser.add_argument('system_name', type=str, help='Name of the system to process (e.g., ant, jhotdraw).')
    args = parser.parse_args()

    main(args.system_name) 