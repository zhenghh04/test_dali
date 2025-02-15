import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

def create_dali_pipeline(
    data_dir, 
    batch_size=16, 
    num_threads=4, 
    device_id=0,
    shuffle=True
):
    """
    Creates a DALI pipeline that reads PNG images from `data_dir`, 
    decodes them, and outputs (images, labels).
    
    :param data_dir: Path to the image directory.
    :param batch_size: Number of samples in each batch.
    :param num_threads: Number of CPU threads to use for image loading and decoding.
    :param device_id: GPU device id.
    :param shuffle: Whether to shuffle the data.
    """
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id
    )
    
    with pipe:
        # Read image files from data_dir
        jpegs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=shuffle,
            name="Reader"
        )
        
        # Decode PNG images (device="mixed" uses GPU acceleration for decode)
        images = fn.decoders.image(
            jpegs,
            device="mixed",
            output_type=types.RGB
        )
        
        # Set pipeline outputs: images and labels
        pipe.set_outputs(images, labels)
    
    return pipe

def main():
    data_dir = "./data/0/"
    batch_size = 16
    num_threads = 4
    device_id = 0

    # Create the DALI pipeline
    pipe = create_dali_pipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id
    )
    
    # Build the pipeline
    pipe.build()

    # Run the pipeline for one epoch (or multiple times as needed)
    num_samples = 100  # For demonstration, say we have 100 samples
    total_iterations = (num_samples + batch_size - 1) // batch_size
    
    for iteration in range(total_iterations):
        # Run one iteration of the pipeline
        outputs = pipe.run()
        # outputs is a tuple/list of DALI tensors (images, labels)
        images, labels = outputs
        
        # Convert DALI tensors to CPU if needed
        # (For further CPU-based processing or saving)
        images_cpu = images.as_cpu().as_array()
        labels_cpu = labels.as_cpu().as_array()
        
        print(f"Iteration {iteration} - images shape: {images_cpu.shape}, labels shape: {labels_cpu.shape}")

if __name__ == "__main__":
    main()
