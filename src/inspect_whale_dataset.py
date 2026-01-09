from datasets import load_dataset

def main():
    ds = load_dataset("janak2/whale-songs", streaming=True)

    print(ds)

    split = list(ds.keys())[0]
    print("Using split:", split)

    row = next(iter(ds[split]))
    print("\nKeys in one example:")
    for k, v in row.items():
        print(" ", k, "->", type(v))

    print("\nExample values (truncated):")
    for k, v in row.items():
        if isinstance(v, (str, int, float)):
            print(f"{k}: {v}")
        elif isinstance(v, dict):
            print(f"{k}: dict keys={list(v.keys())}")
        else:
            print(f"{k}: type={type(v)}")

if __name__ == "__main__":
    main()
