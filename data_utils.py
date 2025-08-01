from datasets import load_dataset, concatenate_datasets, Image
import torchvision.transforms as transforms
import torch # Import torch for tensor operations

def preprocess_images(examples):
    preprocess = transforms.Compose([
        # Resize the image to a fixed size (512x512).
        transforms.Resize((512, 512)),
        # Convert the image to a PyTorch tensor and scale pixel values to [0, 1].
        transforms.ToTensor(),
        # Normalize the pixel values to the range [-1, 1] using mean and standard deviation of 0.5.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # Apply the preprocess transformations to each image in the 'image' column
    # and stack the resulting tensors into a single batch tensor.
    # Ensure images are converted to RGB before processing.
    processed_images = torch.stack([preprocess(image.convert("RGB")) for image in examples["image"]])
    # Return the processed images as a dictionary with the key "image".
    return {"image": processed_images}

def load_and_preprocess_data(Dataset1_dir: str, Dataset2_dir: str):
    # I used two datasets for this project; users can use them as they wish.
    # Load the "Dataset1" from the specified directory using the imagefolder format.
    print(f"Loading Dataset1 from {Dataset1_dir}...")
    Dataset1 = load_dataset("imagefolder", data_dir=Dataset1_dir)
    print("Dataset1 loaded.")

    # Load the "Dataset2" from the specified directory using the imagefolder format.
    print(f"Loading Dataset2 from {Dataset2_dir}...")
    Dataset2 = load_dataset("imagefolder", data_dir=Dataset2_dir)
    print("Dataset2 loaded.")


    # Concatenate the training splits of the two datasets into a single dataset.
    # This creates a combined dataset for training.
    print("Concatenating datasets")
    combined_dataset = concatenate_datasets([Dataset1["train"], Dataset2["train"]])
    print(f"Combined dataset: {combined_dataset}")

    # Apply the preprocessing function to the combined dataset.
    # `batched=True` allows processing images in batches for efficiency.
    # `remove_columns=["image"]` removes the original image column to save memory.
    print("Preprocessing images...")
    preprocessed_dataset = combined_dataset.map(preprocess_images, batched=True, remove_columns=["image"])
    print("Image preprocessing complete.")

    # Display the preprocessed dataset information.
    # Note: In a standalone script, display() might not work directly.
    # You might use print(preprocessed_dataset) instead.
    # display(preprocessed_dataset)

    return preprocessed_dataset

if __name__ == "__main__":
    Dataset1_dir = "path/to/your/Dataset1"
    Dataset2_dir = "path/to/your/Dataset2"

    print("Demonstrating data loading and preprocessing function call:")
    preprocessed_data = load_and_preprocess_data(Dataset1_dir, Dataset2_dir)
    print(preprocessed_data)
    