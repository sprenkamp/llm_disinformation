import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from classifier_new import fact_check
from multiprocessing import Pool, freeze_support
from typing import Tuple
from functools import partial

def process_row(row: pd.Series, llm: ChatOpenAI, perform_search: bool) -> Tuple[str, str]:
    """
    Process a single row of data and return the predicted label and reasoning.

    Parameters:
    row (pd.Series): A single row of data from the DataFrame.
    llm (ChatOpenAI): The ChatOpenAI model for prediction.
    perform_search (bool): Flag indicating whether to perform a search.

    Returns:
    predicted_label (str): The predicted label for the row.
    reasoning (str): The reasoning behind the predicted label.
    """
    statement, author, date = row["statement"], row["author"], row["date"]
    output = fact_check(statement=statement, author=author, date=date, llm=llm, perform_search=perform_search)
    try:
        predicted_label, reasoning = output.split(" -> ")
    except ValueError:
        print(output)
        predicted_label = "error"
        reasoning = "error"
    return predicted_label, reasoning

def process_and_save_df(file_path: str, gpt_model: str, perform_search: str, number_of_samples: int = None) -> None:
    """
    This function reads a CSV file into a DataFrame, makes predictions on the data, 
    and saves a new DataFrame with true labels and predicted labels.

    Parameters:
    file_path (str): The file path of the CSV file.
    gpt_model (str): The GPT model to use for predictions.
    search (str): The search parameter.
    number_of_samples (int, optional): The number of samples to process. If None, the entire DataFrame is processed.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter the DataFrame based on the label
    df = df[df.label.isin(["true", "mostly-true", "half-true", "barely-true", "false-true", "pants-fire"])]

    # If number_of_samples is not None, limit the number of samples
    if number_of_samples is not None:
        df = df[:number_of_samples]

    # Reset the DataFrame index
    df.reset_index(inplace=True)

    # Initialize an empty list for predicted labels
    predicted_labels = []
    reasonings = []

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, model=gpt_model, request_timeout=60)

    # Create a partial function with the fixed arguments
    process_row_partial = partial(process_row, llm=llm, perform_search=perform_search)

    # Create a multiprocessing Pool
    with Pool() as p:
        results = list(tqdm(p.imap(process_row_partial, [row for _, row in df.iterrows()]), total=df.shape[0]))

    predicted_labels, reasonings = zip(*results)

     # Calculate and print accuracy, precision, recall, F1 score, confusion matrix, and classification report
    accuracy = accuracy_score(df['label'], predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(df['label'], predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(df['label'], predicted_labels)
    class_report = classification_report(df['label'], predicted_labels)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: \n{conf_matrix}")
    print(f"Classification Report: \n{class_report}")


    # Add the predicted labels as a new column to the DataFrame
    df['predicted_label'] = predicted_labels
    df['reasoning'] = reasonings

    # Construct the DataFrame name
    df_name = f"{gpt_model}_{perform_search}_{df.shape[0]}"

    # Save the DataFrame to a CSV file
    df.to_csv(f"data/results/{df_name}.csv", index=False)

if __name__ == "__main__":
    freeze_support()
    # Usage
    process_and_save_df(file_path="data/politifact/sample.csv", gpt_model="gpt-3.5-turbo", perform_search=True, number_of_samples=1000)