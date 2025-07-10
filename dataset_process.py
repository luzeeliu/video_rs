import kagglehub

# Download latest version
path = kagglehub.dataset_download("anushabellam/trending-videos-on-youtube")

print("Path to dataset files:", path)