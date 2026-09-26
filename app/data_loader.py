from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Processed data directory
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_customer_voice_data():
    """
    Load all processed Customer Voice Intelligence datasets.
    """

    corpus_path = DATA_DIR / "customer_voice_corpus.csv"
    aspects_path = DATA_DIR / "customer_voice_aspects.csv"
    sentiment_path = DATA_DIR / "aspect_level_sentiment.csv"

    # Validate files
    required_files = [
        corpus_path,
        aspects_path,
        sentiment_path,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing dashboard data files:\n"
            + "\n".join(missing_files)
        )

    # Load datasets
    corpus_df = pd.read_csv(corpus_path)
    aspects_df = pd.read_csv(aspects_path)
    sentiment_df = pd.read_csv(sentiment_path)

    # Basic validation
    if len(corpus_df) != 21021:
        raise ValueError(
            f"Unexpected corpus row count: {len(corpus_df)}"
        )

    if len(aspects_df) != 21021:
        raise ValueError(
            f"Unexpected aspects row count: {len(aspects_df)}"
        )

    if len(sentiment_df) != 21021:
        raise ValueError(
            f"Unexpected sentiment row count: {len(sentiment_df)}"
        )

    return {
        "corpus": corpus_df,
        "aspects": aspects_df,
        "sentiment": sentiment_df,
    }


if __name__ == "__main__":
    data = load_customer_voice_data()

    print("Customer Voice Intelligence data loaded successfully.")
    print()

    for name, df in data.items():
        print(f"{name}: {df.shape}")