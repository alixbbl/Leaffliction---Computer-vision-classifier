import os, argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from config import OUTPUT_DIR


def create_pie_chart(folder_name: str, stats: Dict) -> None:

    categories = list(stats.keys())
    counts = list(stats.values())
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=categories, autopct='%1.1f%%')
    plt.title(f"{folder_name} distribution")
    plt.savefig(f"{OUTPUT_DIR}/{folder_name}_Pie_chart.png")
    plt.show()


def create_bar_chart(folder_name: str, stats: Dict) -> None:

    categories = list(stats.keys())
    counts = list(stats.values())
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts)
    plt.xlabel(f'{folder_name} categories')
    plt.ylabel(f'{folder_name} images count')
    plt.title(f'{folder_name} images count per category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{folder_name}_Bar_chart.png")
    plt.show()


def stats_from_dir(path_dir: str) -> Dict:
    """
        Runs the directory to count the files.
        input : directory path
        output: dict of the folders and files count
    """
    stats = {}
    for root, dirs, files in os.walk(path_dir):
        file_count = len(files)   
        if file_count > 0:
            stats[os.path.basename(root)] = file_count
    
    return stats


# ===================================== MAIN ======================================

def main(parsed_args):
    
    try:
        if parsed_args.directory is None:
            print("Usage: python Distribution.py <directory>")
            print("Example: python Distribution.py ./Apple")
            sys.exit(1)

        if Path(parsed_args.directory).is_dir():
            folder_name = os.path.basename(parsed_args.directory)
            stats = stats_from_dir(parsed_args.directory)
            print(f"Distribution in {folder_name}: {stats}")
            
            create_pie_chart(folder_name, stats)
            create_bar_chart(folder_name, stats)

        else:
            print("No directory provided. Exiting...")
            sys.exit(1)

    except Exception as e:
        print(f"Error occured during processing directory : {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        type=str,
                        nargs='?',
                        default=None,
                        help="Directory to be opened and analyzed.")
    parsed_args = parser.parse_args()
    main(parsed_args)