# Edison Data Acquisition

This script is used to create a simple dataset to train a keyword spotting algorithm.

To get statistics, run
```bash
find out -type d -print0 | while read -d '' -r dir; do
    files=("$dir"/*)
    printf "%5d files in directory %s\n" "${#files[@]}" "$dir"
done
```