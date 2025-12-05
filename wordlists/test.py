"""
交叉检查occup_category.json
1. 是否存在重复
2. 打印不存在于occupations_zh.txt的职位
3. 打印occupations_zh.txt中不存在于occup_category.json的职位
"""
import json
from pathlib import Path
from collections import Counter

# Load files
base_dir = Path(__file__).parent
occup_category_path = base_dir / "occup_category.json"
occupations_path = base_dir / "occupations_zh.txt"

# Load occup_category.json
with open(occup_category_path, "r", encoding="utf-8") as f:
    occup_category = json.load(f)

# Load occupations_zh.txt
with open(occupations_path, "r", encoding="utf-8") as f:
    occupations_list = [line.strip() for line in f if line.strip()]

# Convert to set for faster lookup
occupations_set = set(occupations_list)

# 1. Check for duplicates within occup_category.json
print("=" * 80)
print("1. Checking for duplicates in occup_category.json")
print("=" * 80)

all_occupations_in_category = []
occupation_to_categories = {}

for category, occupations in occup_category.items():
    for occ in occupations:
        all_occupations_in_category.append(occ)
        if occ not in occupation_to_categories:
            occupation_to_categories[occ] = []
        occupation_to_categories[occ].append(category)

# Count occurrences
occupation_counts = Counter(all_occupations_in_category)
duplicates = {occ: count for occ, count in occupation_counts.items() if count > 1}

if duplicates:
    print(f"\nFound {len(duplicates)} duplicate occupations:")
    for occ, count in sorted(duplicates.items()):
        categories = occupation_to_categories[occ]
        print(f"  '{occ}' appears {count} times in categories: {', '.join(categories)}")
else:
    print("\nNo duplicates found in occup_category.json")

# 2. Find occupations in occup_category.json that are not in occupations_zh.txt
print("\n" + "=" * 80)
print("2. Occupations in occup_category.json but NOT in occupations_zh.txt")
print("=" * 80)

unique_occupations_in_category = set(all_occupations_in_category)
missing_in_occupations_txt = unique_occupations_in_category - occupations_set

if missing_in_occupations_txt:
    print(f"\nFound {len(missing_in_occupations_txt)} occupations:")
    for occ in sorted(missing_in_occupations_txt):
        categories = occupation_to_categories[occ]
        print(f"  '{occ}' (in categories: {', '.join(categories)})")
else:
    print("\nAll occupations in occup_category.json are present in occupations_zh.txt")

# 3. Find occupations in occupations_zh.txt that are not in occup_category.json
print("\n" + "=" * 80)
print("3. Occupations in occupations_zh.txt but NOT in occup_category.json")
print("=" * 80)

missing_in_category = occupations_set - unique_occupations_in_category

if missing_in_category:
    print(f"\nFound {len(missing_in_category)} occupations:")
    for occ in sorted(missing_in_category):
        print(f"  '{occ}'")
else:
    print("\nAll occupations in occupations_zh.txt are present in occup_category.json")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Total unique occupations in occup_category.json: {len(unique_occupations_in_category)}")
print(f"Total occupations in occupations_zh.txt: {len(occupations_set)}")
print(f"Overlap: {len(unique_occupations_in_category & occupations_set)}")
print(f"Only in occup_category.json: {len(missing_in_occupations_txt)}")
print(f"Only in occupations_zh.txt: {len(missing_in_category)}")
print(f"Duplicates in occup_category.json: {len(duplicates)}")
