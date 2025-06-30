import os, argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from config import OUTPUT_DIR


def create_pie_chart(stats: Dict) -> None:

    categories = list(stats.keys())
    counts = list(stats.values())
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=categories, autopct='%1.1f%%')
    plt.title(f"Images distribution")
    plt.savefig(f"{OUTPUT_DIR}/Pie_chart.png")
    plt.show()


def create_bar_chart(stats: Dict) -> None:

    categories = list(stats.keys())
    counts = list(stats.values())
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts)
    plt.xlabel('Categories')
    plt.ylabel('Images count')
    plt.title('Images count per category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Bar_chart.png")
    plt.show()


def stats_from_dir(path_dir: str) -> Dict:
    
    stats = {}
    for root, dirs, files in os.walk(path_dir):
        if root == path_dir:
            continue
        folder_name = os.path.basename(root)
        file_count = len(files)   
        if file_count > 0:
            stats[folder_name] = file_count 
    
    return stats


def main(parsed_args):
    
    try:
        if Path(parsed_args.directory).is_dir():
            
            stats = stats_from_dir(parsed_args.directory)
            print(f"Here is my distribution : {stats}")
            
            create_pie_chart(stats)
            create_bar_chart(stats)

        else:
            print("Not a directory.")
    
    except Exception as e:
        print(f"Error occured during processing directory : {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default=None,
                        help="Directory to be opened and analyzed.")
    parsed_args = parser.parse_args()
    main(parsed_args)