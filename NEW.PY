import kagglehub

# Download latest version
path = kagglehub.dataset_download("montassarba/mimic-iv-clinical-database-demo-2-2")

print("Path to dataset files:", path)