#!/usr/bin/env python3
"""
Google Cloud Image Generation Script using Vertex AI SDK
Generates images using Google's Imagen 4.0 model and saves them locally.
Enhanced version with support for text-to-image generation and concurrent processing.
"""

import json
import time
import os
import sys
from datetime import datetime
import re
from PIL import Image
import io
import base64
import concurrent.futures
import threading
from queue import Queue

# Import Vertex AI SDK
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    print("✓ Vertex AI SDK imported successfully")
except ImportError as e:
    print(f"❌ Error importing Vertex AI SDK: {e}")
    print("Please install it with: pip install --upgrade --user google-cloud-aiplatform")
    sys.exit(1)

class ImageGenerator:
    def __init__(self, project_id=None):
        # Project ID must be provided
        self.project_id = project_id
        self.location_id = "us-central1"
        self.model_id = "imagen-4.0-generate-preview-06-06"
        self.image_folder = "generated_images"
        
        # Thread-safe lock for printing
        self.print_lock = threading.Lock()
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location_id)
            print(f"✓ Vertex AI initialized with project: {self.project_id}")
        except Exception as e:
            error_message = f"❌ Error initializing Vertex AI: {e}"
            # Print with bold formatting and red color for visibility
            print(f"\033[1;31m{error_message}\033[0m")
            print("\033[1;31mPlease make sure you're authenticated with gcloud auth login\033[0m")
            print("\033[1;31mand that the project ID is correct.\033[0m")
            sys.exit(1)
        
        # Initialize the model
        try:
            self.generation_model = ImageGenerationModel.from_pretrained(self.model_id)
            print(f"✓ Model loaded: {self.model_id}")
        except Exception as e:
            error_message = f"❌ Error loading model: {e}"
            # Print with bold formatting and red color for visibility
            print(f"\033[1;31m{error_message}\033[0m")
            sys.exit(1)
        
        # Create image folder if it doesn't exist
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
            print(f"Created folder: {self.image_folder}")

    def safe_print(self, message):
        """Thread-safe printing"""
        with self.print_lock:
            print(message)

    # Method removed as prompts are now provided directly through the web interface

    def generate_images(self, prompt, seed=None, sample_count=4, aspect_ratio="16:9", 
                       negative_prompt="", person_generation="allow_all", 
                       safety_filter_level="block_few", add_watermark=True, max_retries=3):
        """Generate images using Vertex AI SDK with retry logic for authentication errors"""
        self.safe_print(f"Generating {sample_count} images for prompt: '{prompt}'")
        if seed:
            self.safe_print(f"Using seed: {seed}")
        self.safe_print(f"Aspect ratio: {aspect_ratio}")
        self.safe_print(f"Person generation: {person_generation}")
        self.safe_print(f"Safety filter: {safety_filter_level}")
        self.safe_print(f"Watermark: {'Yes' if add_watermark else 'No'}")
        
        for attempt in range(1, max_retries + 1):
            try:
                # Call the Vertex AI SDK
                response = self.generation_model.generate_images(
                    prompt=prompt,
                    number_of_images=sample_count,
                    aspect_ratio=aspect_ratio,
                    negative_prompt=negative_prompt,
                    person_generation=person_generation,
                    safety_filter_level=safety_filter_level,
                    add_watermark=add_watermark,
                )
                images = response.images  # Extract the list of PIL images
                self.safe_print(f"✓ Successfully generated {len(images)} images")
                return images
                
            except Exception as e:
                error_str = str(e)
                self.safe_print(f"❌ Error generating images (attempt {attempt}/{max_retries}): {e}")
                
                # Check if it's an authentication error
                if "401" in error_str and ("authentication" in error_str.lower() or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in error_str):
                    if attempt < max_retries:
                        self.safe_print(f"🔄 Authentication error detected. Retrying in 5 seconds... (attempt {attempt}/{max_retries})")
                        time.sleep(5)
                        continue
                    else:
                        self.safe_print(f"❌ Authentication error persisted after {max_retries} attempts. Stopping script.")
                        print("\n=== AUTHENTICATION ERROR ===")
                        print("The script encountered persistent authentication errors.")
                        print("Please check your Google Cloud authentication:")
                        print("1. Run: gcloud auth login")
                        print("2. Run: gcloud auth application-default login")
                        print("3. Verify your project ID and permissions")
                        print("4. Restart the script")
                        sys.exit(1)
                else:
                    # For non-authentication errors, don't retry
                    self.safe_print(f"❌ Non-authentication error. Not retrying.")
                    return None
        
        return None

    def save_images(self, images, prompt, seed=None):
        """Save generated images to local files"""
        if not images:
            self.safe_print("No images to save")
            return []
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean prompt for filename (remove special characters)
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt).strip()
        clean_prompt = re.sub(r'[-\s]+', '_', clean_prompt)[:50]  # Limit length
        
        # Add seed to filename if available
        seed_suffix = f"_seed_{seed}" if seed is not None else ""
        
        for i, image in enumerate(images):
            try:
                filename = f"{clean_prompt}_{timestamp}{seed_suffix}_image_{i+1}.png"
                filepath = os.path.join(self.image_folder, filename)
                image.save(filepath)  # Use the SDK's save method directly
                self.safe_print(f"Saved image {i+1}: {filepath}")
                saved_files.append(filepath)
            except Exception as e:
                self.safe_print(f"Error saving image {i+1}: {e}")
        
        return saved_files

    def process_single_prompt(self, prompt_data):
        """Process a single prompt with given parameters"""
        prompt, seed, sample_count, aspect_ratio, person_generation, safety_filter_level, add_watermark = prompt_data
        
        try:
            # Generate images
            images = self.generate_images(
                prompt, seed, sample_count, aspect_ratio, 
                person_generation=person_generation, 
                safety_filter_level=safety_filter_level, 
                add_watermark=add_watermark
            )
            
            if images:
                # Save images
                saved_files = self.save_images(images, prompt, seed)
                if saved_files:
                    self.safe_print(f"Successfully generated and saved {len(saved_files)} images for prompt: '{prompt}'")
                    self.safe_print(f"Seed used: {seed}")
                    return len(saved_files)
                else:
                    self.safe_print(f"No images were saved for prompt: '{prompt}'")
                    return 0
            else:
                self.safe_print(f"No images were generated for prompt: '{prompt}'")
                return 0
        except SystemExit:
            # Re-raise SystemExit to stop the entire script
            raise
        except Exception as e:
            self.safe_print(f"Unexpected error processing prompt '{prompt}': {e}")
            return 0

    def run_concurrent_batch_generation(self, prompts_data, max_workers=5):
        """Process multiple prompts concurrently with 5-second delays between batches"""
        print("\n=== Concurrent Batch Generation from Text Files ===")
        
        # Get common settings for all prompts
        print("\nCommon settings for all prompt generations:")
        
        # Ask for number of images to generate per prompt
        print()
        sample_count = 4  # Default for batch
        sample_input = input("How many images per prompt? (1-8, default: 4): ").strip()
        if sample_input:
            try:
                sample_count = int(sample_input)
                if sample_count < 1 or sample_count > 8:
                    print("Invalid number, using default: 4 images per prompt")
                    sample_count = 4
                else:
                    print(f"Will generate {sample_count} images per prompt")
            except ValueError:
                print("Invalid number, using default: 4 images per prompt")
        else:
            print("Using default: 4 images per prompt")
        
        # Ask for aspect ratio
        print()
        print("Choose aspect ratio:")
        print("1. Landscape (16:9) - default")
        print("2. Portrait (9:16)")
        print("3. Square (1:1)")
        ratio_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        aspect_ratio = "16:9"  # Default
        if ratio_choice == "2":
            aspect_ratio = "9:16"
            print("Using portrait (9:16) aspect ratio")
        elif ratio_choice == "3":
            aspect_ratio = "1:1"
            print("Using square (1:1) aspect ratio")
        else:
            print("Using landscape (16:9) aspect ratio")
        
        # Ask for person generation setting
        print()
        print("Person generation setting:")
        print("1. Allow all (default)")
        print("2. Allow adult only")
        print("3. Block all")
        person_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        person_generation = "allow_all"  # Default
        if person_choice == "2":
            person_generation = "allow_adult"
            print("Using allow adult only")
        elif person_choice == "3":
            person_generation = "block_all"
            print("Using block all")
        else:
            print("Using allow all")
        
        # Ask for safety filter level
        print()
        print("Safety filter level:")
        print("1. Block few (default)")
        print("2. Block some")
        print("3. Block most")
        safety_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        safety_filter_level = "block_few"  # Default
        if safety_choice == "2":
            safety_filter_level = "block_some"
            print("Using block some")
        elif safety_choice == "3":
            safety_filter_level = "block_most"
            print("Using block most")
        else:
            print("Using block few")
        
        # Ask for watermark setting
        print()
        watermark_choice = input("Add watermark? (y/n, default: y): ").strip().lower() or "y"
        add_watermark = watermark_choice == "y"
        print(f"Watermark: {'Yes' if add_watermark else 'No'}")
        
        # Ask for seed option
        print()
        print("Seed options:")
        print("1. Use random seed for each prompt (default)")
        print("2. Use the same seed for all prompts")
        print("3. Use incremental seeds starting from base seed")
        seed_option = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        base_seed = None
        if seed_option in ["2", "3"]:
            seed_input = input("Enter base seed number: ").strip()
            try:
                base_seed = int(seed_input)
                print(f"Using base seed: {base_seed}")
            except ValueError:
                print("Invalid seed number, using random seeds instead")
                seed_option = "1"
        
        print(f"\n=== Starting Concurrent Batch Generation ===")
        print(f"Processing with {max_workers} concurrent workers")
        print(f"5-second delay between batches")
        
        # Prepare all prompts with their parameters
        all_prompts = []
        prompt_counter = 0
        
        for file_name, prompts in prompts_data:
            print(f"\nPreparing prompts from file: {file_name}")
            
            for i, prompt in enumerate(prompts):
                prompt_counter += 1
                
                # Determine seed based on option
                if seed_option == "1":  # Random seed for each prompt
                    import random
                    seed = random.randint(1, 2147483647)
                elif seed_option == "2":  # Same seed for all prompts
                    seed = base_seed
                elif seed_option == "3":  # Incremental seeds
                    seed = base_seed + prompt_counter - 1
                
                # Create prompt data tuple
                prompt_data = (prompt, seed, sample_count, aspect_ratio, 
                             person_generation, safety_filter_level, add_watermark)
                all_prompts.append(prompt_data)
        
        print(f"\nTotal prompts to process: {len(all_prompts)}")
        
        # Process prompts in batches of max_workers
        total_successful_images = 0
        batch_number = 1
        
        for i in range(0, len(all_prompts), max_workers):
            batch = all_prompts[i:i + max_workers]
            print(f"\n=== Processing Batch {batch_number} ===")
            print(f"Processing {len(batch)} prompts concurrently...")
            
            # Process batch concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks in the batch
                future_to_prompt = {
                    executor.submit(self.process_single_prompt, prompt_data): prompt_data[0] 
                    for prompt_data in batch
                }
                
                # Collect results as they complete
                batch_successful_images = 0
                for future in concurrent.futures.as_completed(future_to_prompt):
                    prompt = future_to_prompt[future]
                    try:
                        images_generated = future.result()
                        batch_successful_images += images_generated
                    except SystemExit:
                        # Re-raise SystemExit to stop the entire script
                        raise
                    except Exception as e:
                        self.safe_print(f"Error processing prompt '{prompt}': {e}")
            
            total_successful_images += batch_successful_images
            print(f"Batch {batch_number} complete: {batch_successful_images} images generated")
            
            # 5-second delay before next batch (except for the last batch)
            if i + max_workers < len(all_prompts):
                print("Waiting 5 seconds before next batch...")
                time.sleep(5)
            
            batch_number += 1
        
        print("\n=== Concurrent Batch Generation Complete ===")
        print(f"Processed {len(all_prompts)} prompts in {batch_number - 1} batches")
        print(f"Successfully generated {total_successful_images} images")
        print(f"All images saved to '{self.image_folder}' folder")

    def run_batch_generation(self, prompts_data):
        """Process multiple prompts from text files for batch generation (sequential version)"""
        print("\n=== Sequential Batch Generation from Text Files ===")
        
        # Get common settings for all prompts
        print("\nCommon settings for all prompt generations:")
        
        # Ask for number of images to generate per prompt
        print()
        sample_count = 4  # Default for batch
        sample_input = input("How many images per prompt? (1-8, default: 4): ").strip()
        if sample_input:
            try:
                sample_count = int(sample_input)
                if sample_count < 1 or sample_count > 8:
                    print("Invalid number, using default: 4 images per prompt")
                    sample_count = 4
                else:
                    print(f"Will generate {sample_count} images per prompt")
            except ValueError:
                print("Invalid number, using default: 4 images per prompt")
        else:
            print("Using default: 4 images per prompt")
        
        # Ask for aspect ratio
        print()
        print("Choose aspect ratio:")
        print("1. Landscape (16:9) - default")
        print("2. Portrait (9:16)")
        print("3. Square (1:1)")
        ratio_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        aspect_ratio = "16:9"  # Default
        if ratio_choice == "2":
            aspect_ratio = "9:16"
            print("Using portrait (9:16) aspect ratio")
        elif ratio_choice == "3":
            aspect_ratio = "1:1"
            print("Using square (1:1) aspect ratio")
        else:
            print("Using landscape (16:9) aspect ratio")
        
        # Ask for person generation setting
        print()
        print("Person generation setting:")
        print("1. Allow all (default)")
        print("2. Allow adult only")
        print("3. Block all")
        person_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        person_generation = "allow_all"  # Default
        if person_choice == "2":
            person_generation = "allow_adult"
            print("Using allow adult only")
        elif person_choice == "3":
            person_generation = "block_all"
            print("Using block all")
        else:
            print("Using allow all")
        
        # Ask for safety filter level
        print()
        print("Safety filter level:")
        print("1. Block few (default)")
        print("2. Block some")
        print("3. Block most")
        safety_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        safety_filter_level = "block_few"  # Default
        if safety_choice == "2":
            safety_filter_level = "block_some"
            print("Using block some")
        elif safety_choice == "3":
            safety_filter_level = "block_most"
            print("Using block most")
        else:
            print("Using block few")
        
        # Ask for watermark setting
        print()
        watermark_choice = input("Add watermark? (y/n, default: y): ").strip().lower() or "y"
        add_watermark = watermark_choice == "y"
        print(f"Watermark: {'Yes' if add_watermark else 'No'}")
        
        # Ask for seed option
        print()
        print("Seed options:")
        print("1. Use random seed for each prompt (default)")
        print("2. Use the same seed for all prompts")
        print("3. Use incremental seeds starting from base seed")
        seed_option = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        base_seed = None
        if seed_option in ["2", "3"]:
            seed_input = input("Enter base seed number: ").strip()
            try:
                base_seed = int(seed_input)
                print(f"Using base seed: {base_seed}")
            except ValueError:
                print("Invalid seed number, using random seeds instead")
                seed_option = "1"
        
        print("\n=== Starting Batch Generation ===")
        
        # Process each prompt file
        total_prompts = sum(len(prompts) for _, prompts in prompts_data)
        successful_images = 0
        prompt_counter = 0
        
        for file_name, prompts in prompts_data:
            print(f"\nProcessing prompts from file: {file_name}")
            
            for i, prompt in enumerate(prompts):
                prompt_counter += 1
                print(f"\n[{prompt_counter}/{total_prompts}] Processing prompt: '{prompt}'")
                
                # Determine seed based on option
                if seed_option == "1":  # Random seed for each prompt
                    import random
                    seed = random.randint(1, 2147483647)
                    print(f"Generated random seed: {seed}")
                elif seed_option == "2":  # Same seed for all prompts
                    seed = base_seed
                    print(f"Using same seed for all: {seed}")
                elif seed_option == "3":  # Incremental seeds
                    seed = base_seed + prompt_counter - 1
                    print(f"Using incremental seed: {seed}")
                
                try:
                    # Generate images
                    images = self.generate_images(
                        prompt, seed, sample_count, aspect_ratio, 
                        person_generation=person_generation, 
                        safety_filter_level=safety_filter_level, 
                        add_watermark=add_watermark
                    )
                    
                    if images:
                        # Save images
                        saved_files = self.save_images(images, prompt, seed)
                        if saved_files:
                            successful_images += len(saved_files)
                            print(f"Successfully generated and saved {len(saved_files)} images for this prompt")
                            print(f"Seed used: {seed}")
                        else:
                            print("No images were saved for this prompt")
                    else:
                        print("No images were generated for this prompt")
                except SystemExit:
                    # Re-raise SystemExit to stop the entire script
                    raise
                except Exception as e:
                    print(f"Unexpected error processing prompt '{prompt}': {e}")
                
                # Brief pause between requests to avoid rate limiting
                if prompt_counter < total_prompts:
                    print("Pausing briefly before next generation...")
                    time.sleep(5)
        
        print("\n=== Batch Generation Complete ===")
        print(f"Processed {total_prompts} prompts")
        print(f"Successfully generated {successful_images} images")
        print(f"All images saved to '{self.image_folder}' folder")

    def run(self):
        """Main execution method"""
        print("=== Google Cloud Image Generation Script ===")
        print()
        
        # Choose the primary generation mode
        print("Choose primary generation mode:")
        print("1. Manual prompt entry (default)")
        print("2. Load prompts from text files (sequential)")
        print("3. Load prompts from text files (concurrent - 5 requests simultaneously)")
        primary_mode = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        if primary_mode in ["2", "3"]:
            # Automated mode: Load prompts from text files
            prompts_data = self.get_prompts_from_files()
            
            if not prompts_data:
                print("\nNo prompts found in text files. Switching to manual mode.")
                primary_mode = "1"
            else:
                if primary_mode == "3":
                    self.run_concurrent_batch_generation(prompts_data)
                else:
                    self.run_batch_generation(prompts_data)
                return
        
        # Manual mode: Ask for generation parameters
        print("\n=== Text-to-Image Generation ===")
        
        # Get prompt from user
        prompt = input("Enter your image generation prompt: ").strip()
        if not prompt:
            print("Error: Prompt cannot be empty")
            return
        
        # Ask for number of images to generate
        print()
        sample_count = 4  # Default
        sample_input = input("How many images do you want to generate? (1-8, default: 4): ").strip()
        if sample_input:
            try:
                sample_count = int(sample_input)
                if sample_count < 1 or sample_count > 8:
                    print("Invalid number, using default: 4 images")
                    sample_count = 4
                else:
                    print(f"Will generate {sample_count} images")
            except ValueError:
                print("Invalid number, using default: 4 images")
        else:
            print("Using default: 4 images")
        
        # Ask for aspect ratio
        print()
        print("Choose aspect ratio:")
        print("1. Landscape (16:9) - default")
        print("2. Portrait (9:16)")
        print("3. Square (1:1)")
        ratio_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        aspect_ratio = "16:9"  # Default
        if ratio_choice == "2":
            aspect_ratio = "9:16"
            print("Using portrait (9:16) aspect ratio")
        elif ratio_choice == "3":
            aspect_ratio = "1:1"
            print("Using square (1:1) aspect ratio")
        else:
            print("Using landscape (16:9) aspect ratio")
        
        # Ask for person generation setting
        print()
        print("Person generation setting:")
        print("1. Allow all (default)")
        print("2. Allow adult only")
        print("3. Block all")
        person_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        person_generation = "allow_all"  # Default
        if person_choice == "2":
            person_generation = "allow_adult"
            print("Using allow adult only")
        elif person_choice == "3":
            person_generation = "block_all"
            print("Using block all")
        else:
            print("Using allow all")
        
        # Ask for safety filter level
        print()
        print("Safety filter level:")
        print("1. Block few (default)")
        print("2. Block some")
        print("3. Block most")
        safety_choice = input("Enter choice (1, 2, or 3): ").strip() or "1"
        
        safety_filter_level = "block_few"  # Default
        if safety_choice == "2":
            safety_filter_level = "block_some"
            print("Using block some")
        elif safety_choice == "3":
            safety_filter_level = "block_most"
            print("Using block most")
        else:
            print("Using block few")
        
        # Ask for watermark setting
        print()
        watermark_choice = input("Add watermark? (y/n, default: y): ").strip().lower() or "y"
        add_watermark = watermark_choice == "y"
        print(f"Watermark: {'Yes' if add_watermark else 'No'}")
        
        # Get optional seed from user
        print()
        seed_input = input("Enter seed number (optional, press Enter to skip): ").strip()
        seed = None
        if seed_input:
            try:
                seed = int(seed_input)
                print(f"Using provided seed: {seed}")
            except ValueError:
                print("Invalid seed number, generating random seed instead")
                import random
                seed = random.randint(1, 2147483647)  # Max 32-bit signed integer
                print(f"Generated random seed: {seed}")
        else:
            # Generate a random seed if none provided
            import random
            seed = random.randint(1, 2147483647)  # Max 32-bit signed integer
            print(f"Generated random seed: {seed}")
        
        print()
        
        try:
            # Generate images
            images = self.generate_images(
                prompt, seed, sample_count, aspect_ratio, 
                person_generation=person_generation, 
                safety_filter_level=safety_filter_level, 
                add_watermark=add_watermark
            )
            
            if images:
                # Save images
                saved_files = self.save_images(images, prompt, seed)
                
                if saved_files:
                    print("\n=== Generation Complete ===")
                    print(f"Generated {len(saved_files)} images and saved to '{self.image_folder}' folder:")
                    for file in saved_files:
                        print(f"  - {file}")
                    
                    # Display seed information prominently
                    print("\n=== SEED INFORMATION ===")
                    print(f"🌱 SEED USED: {seed}")
                    print("💡 Tip: Use this seed with different prompts to get similar visual styles!")
                    print(f"💡 Command: Enter '{seed}' when prompted for seed number")
                else:
                    print("\n=== Generation Failed ===")
                    print("No images were saved.")
            else:
                print("\n=== Generation Failed ===")
                print("No images were generated.")
        except SystemExit:
            # Re-raise SystemExit to stop the entire script
            raise
        except Exception as e:
            print(f"\nUnexpected error during generation: {e}")

def main():
    """Entry point of the script"""
    try:
        generator = ImageGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 