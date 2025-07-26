from diffusers import StableDiffusionPipeline
import torch

def generate_image_from_text(prompt, output_path="generated_image.png"):
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate image
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved at: {output_path}")

# Example usage
generate_image_from_text("a scenic view of mountains during sunset with birds flying")
