#!/usr/bin/env python3
"""
Flask server for Google Cloud Image Generation Script
Provides a web interface to generate images using Vertex AI
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import time
import json
from queue import Queue
from image_gen_script import ImageGenerator
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for tracking progress
progress_data = {
    "total_prompts": 0,
    "processed_prompts": 0,
    "remaining_prompts": 0,
    "total_images": 0,
    "status": "idle",
    "current_prompt": "",
    "batch_number": 0,
    "total_batches": 0,
    "is_processing": False
}

# Reset progress data to initial state
def reset_progress_data():
    global progress_data
    progress_data = {
        "total_prompts": 0,
        "processed_prompts": 0,
        "remaining_prompts": 0,
        "total_images": 0,
        "status": "idle",
        "current_prompt": "",
        "batch_number": 0,
        "total_batches": 0,
        "is_processing": False
    }

# Thread-safe lock for updating progress data
progress_lock = threading.Lock()

def update_progress(key, value):
    """Thread-safe update of progress data"""
    with progress_lock:
        progress_data[key] = value

def process_batch(prompts, sample_count, aspect_ratio, person_generation, 
                safety_filter_level, add_watermark, seed_option, base_seed, max_workers, project_id=None):
    """Process batch of prompts in background thread"""
    try:
        # Set initial status
        update_progress("status", "initializing")
        update_progress("is_processing", True)
        
        # Print bold progress header
        print("\033[1m" + "=" * 80 + "\033[0m")
        print("\033[1mIMAGE GENERATION STARTED\033[0m")
        print(f"\033[1mTotal prompts: {len(prompts)}\033[0m")
        print(f"\033[1mImages per prompt: {sample_count}\033[0m")
        print(f"\033[1mAspect ratio: {aspect_ratio}\033[0m")
        if project_id:
            print(f"\033[1mProject ID: {project_id}\033[0m")
        print("\033[1m" + "=" * 80 + "\033[0m")
        
        # After a short delay, change to processing status
        time.sleep(1)
        update_progress("status", "processing")
        
        # Initialize generator with project ID if provided
        generator = ImageGenerator(project_id=project_id)
        
        # Override the safe_print method to update progress
        original_safe_print = generator.safe_print
        
        def progress_tracking_print(message):
            original_safe_print(message)
            
            # Update current prompt if detected
            if "Processing prompt: '" in message:
                current_prompt = message.split("Processing prompt: '")[1].split("'")[0]
                update_progress("current_prompt", current_prompt)
                print(f"\033[1m[PROGRESS] Processing: '{current_prompt}'\033[0m")
            elif "prompt: '" in message:
                current_prompt = message.split("prompt: '")[1].split("'")[0]
                update_progress("current_prompt", current_prompt)
            
            # Update batch information
            if "=== Processing Batch" in message:
                batch_num = int(message.split("Batch ")[1].split(" ")[0])
                update_progress("batch_number", batch_num)
                print(f"\033[1m[BATCH] Starting batch {batch_num}\033[0m")
            
            # Update image count when images are saved
            if "Successfully generated and saved" in message:
                try:
                    count = int(message.split("Successfully generated and saved ")[1].split(" ")[0])
                    with progress_lock:
                        progress_data["total_images"] += count
                        print(f"\033[1m[COMPLETED] Generated {count} images. Total: {progress_data['total_images']}\033[0m")
                except Exception as e:
                    pass
            elif "Saved image" in message:
                with progress_lock:
                    progress_data["total_images"] += 1
            
            # Update processed prompts count
            if "Successfully generated" in message and ("images for prompt:" in message or "for this prompt" in message):
                with progress_lock:
                    progress_data["processed_prompts"] += 1
                    progress_data["remaining_prompts"] = progress_data["total_prompts"] - progress_data["processed_prompts"]
                    print(f"\033[1m[PROGRESS] Completed {progress_data['processed_prompts']}/{progress_data['total_prompts']} prompts. Remaining: {progress_data['remaining_prompts']}\033[0m")
            elif "Generating" in message and "images for prompt:" in message:
                # Capture when processing starts for a prompt
                try:
                    current_prompt = message.split("images for prompt: '")[1].split("'")[0]
                    update_progress("current_prompt", current_prompt)
                except:
                    pass
        
        # Replace the safe_print method
        generator.safe_print = progress_tracking_print
        
        # Prepare prompts data in the format expected by the generator
        prompts_data = [("web_interface.txt", prompts)]
        
        # Set total prompts
        update_progress("total_prompts", len(prompts))
        update_progress("remaining_prompts", len(prompts))
        update_progress("processed_prompts", 0)
        update_progress("total_images", 0)
        
        # Calculate total batches
        total_batches = (len(prompts) + max_workers - 1) // max_workers
        update_progress("total_batches", total_batches)
        
        update_progress("status", "processing")
        
        # Process prompts based on concurrent or sequential mode
        if max_workers > 1:
            # Use concurrent processing
            all_prompts = []
            prompt_counter = 0
            
            for file_name, prompt_list in prompts_data:
                for i, prompt in enumerate(prompt_list):
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
            
            # Process prompts in batches
            batch_number = 1
            
            for i in range(0, len(all_prompts), max_workers):
                batch = all_prompts[i:i + max_workers]
                update_progress("batch_number", batch_number)
                
                # Process batch concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks in the batch
                    future_to_prompt = {
                        executor.submit(generator.process_single_prompt, prompt_data): prompt_data[0] 
                        for prompt_data in batch
                    }
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_prompt):
                        prompt = future_to_prompt[future]
                        try:
                            future.result()
                        except Exception as e:
                            generator.safe_print(f"Error processing prompt '{prompt}': {e}")
                
                # 5-second delay before next batch (except for the last batch)
                if i + max_workers < len(all_prompts):
                    time.sleep(5)
                
                batch_number += 1
        else:
            # Use sequential processing
            total_prompts = sum(len(prompts) for _, prompts in prompts_data)
            prompt_counter = 0
            
            for file_name, prompt_list in prompts_data:
                for i, prompt in enumerate(prompt_list):
                    prompt_counter += 1
                    update_progress("current_prompt", prompt)
                    
                    # Determine seed based on option
                    if seed_option == "1":  # Random seed for each prompt
                        import random
                        seed = random.randint(1, 2147483647)
                    elif seed_option == "2":  # Same seed for all prompts
                        seed = base_seed
                    elif seed_option == "3":  # Incremental seeds
                        seed = base_seed + prompt_counter - 1
                    
                    try:
                        # Generate images
                        images = generator.generate_images(
                            prompt, seed, sample_count, aspect_ratio, 
                            person_generation=person_generation, 
                            safety_filter_level=safety_filter_level, 
                            add_watermark=add_watermark
                        )
                        
                        if images:
                            # Save images
                            saved_files = generator.save_images(images, prompt, seed)
                            if saved_files:
                                with progress_lock:
                                    progress_data["total_images"] += len(saved_files)
                    except Exception as e:
                        generator.safe_print(f"Unexpected error processing prompt '{prompt}': {e}")
                    
                    # Update processed count
                    with progress_lock:
                        progress_data["processed_prompts"] += 1
                        progress_data["remaining_prompts"] = progress_data["total_prompts"] - progress_data["processed_prompts"]
                    
                    # Brief pause between requests to avoid rate limiting
                    if prompt_counter < total_prompts:
                        time.sleep(5)
        
        # Print completion message with bold formatting
        print("\033[1m" + "=" * 80 + "\033[0m")
        print("\033[1mIMAGE GENERATION COMPLETED\033[0m")
        print(f"\033[1mProcessed {progress_data['total_prompts']} prompts\033[0m")
        print(f"\033[1mGenerated {progress_data['total_images']} images\033[0m")
        print(f"\033[1mImages saved to '{generator.image_folder}' folder\033[0m")
        print("\033[1m" + "=" * 80 + "\033[0m")
        
        update_progress("status", "completed")
    except Exception as e:
        update_progress("status", f"error: {str(e)}")
    finally:
        update_progress("is_processing", False)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Handle image generation request"""
    if progress_data["is_processing"]:
        return jsonify({"error": "A batch is already processing"}), 400
    
    try:
        # Reset progress data before starting new batch
        reset_progress_data()
        
        data = request.json
        
        # Extract parameters from request
        prompts = data.get('prompts', '').strip().split('\n')
        prompts = [p.strip() for p in prompts if p.strip()]
        
        if not prompts:
            return jsonify({"error": "No valid prompts provided"}), 400
        
        # Update total prompts count immediately
        with progress_lock:
            progress_data["total_prompts"] = len(prompts)
            progress_data["remaining_prompts"] = len(prompts)
        
        sample_count = int(data.get('sample_count', 4))
        aspect_ratio = data.get('aspect_ratio', '16:9')
        person_generation = data.get('person_generation', 'allow_all')
        safety_filter_level = data.get('safety_filter_level', 'block_few')
        add_watermark = data.get('add_watermark', True)
        seed_option = data.get('seed_option', '1')
        base_seed = int(data.get('base_seed', 0)) if data.get('base_seed') else None
        max_workers = int(data.get('max_workers', 1))
        project_id = data.get('project_id')
        
        # Validate project_id
        if not project_id:
            return jsonify({"error": "Project ID is required"}), 400
        
        # Validate parameters
        if sample_count < 1 or sample_count > 8:
            return jsonify({"error": "Sample count must be between 1 and 8"}), 400
            
        if max_workers < 1 or max_workers > 10:
            return jsonify({"error": "Max workers must be between 1 and 10"}), 400
        
        if seed_option in ['2', '3'] and base_seed is None:
            return jsonify({"error": "Base seed is required for selected seed option"}), 400
        
        # Calculate total batches
        total_batches = (len(prompts) + max_workers - 1) // max_workers
        with progress_lock:
            progress_data["total_batches"] = total_batches
        
        # Start processing in background thread
        processing_thread = threading.Thread(
            target=process_batch,
            args=(prompts, sample_count, aspect_ratio, person_generation, 
                  safety_filter_level, add_watermark, seed_option, base_seed, max_workers, project_id)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({"message": "Batch processing started", "prompt_count": len(prompts)}), 200
    
    except Exception as e:
        print(f"\033[1m[ERROR] {str(e)}\033[0m")
        return jsonify({"error": str(e)}), 500

@app.route('/progress')
def get_progress():
    """Get current progress data"""
    with progress_lock:
        return jsonify(progress_data)

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

# Create necessary folders
def create_folders():
    """Create necessary folders if they don't exist"""
    folders = ['templates', 'static', 'generated_images']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

if __name__ == '__main__':
    # Create necessary folders
    create_folders()
    
    # Import here to avoid circular imports
    import concurrent.futures
    from flask import send_from_directory
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)