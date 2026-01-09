from huggingface_hub import hf_hub_download

def main():
    path = hf_hub_download(
        repo_id="peng7554/DS3500",
        repo_type="dataset",    
        filename="DS3500.zip",
        token=False               
    )
    print("Downloaded to:", path)

if __name__ == "__main__":
    main()
