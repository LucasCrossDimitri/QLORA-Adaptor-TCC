import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = "TheBloke/StableBeluga-7B-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

while True:
    try:
        # Load the user's message from an external text file
        with open("user_message.txt", "r", encoding="utf-8") as user_message_file:
            message = user_message_file.read().strip()  # Read and remove leading/trailing whitespace

        system_prompt = """### System:\n Generate a dialogue between a user and an industrial company attendant. The user initiates the conversation, and the assistant responds accordingly. Please create a few examples of such interactions.\n\n"""

        # Tokenize the input text
        tokens = tokenizer(message, return_tensors="pt")

        # Get the number of tokens
        num_tokens = tokens["input_ids"].shape[1]
        
        # Display the number of tokens
        print(f"Number of tokens in input text: {num_tokens}")

        # Record the start time
        start_time = time.time()

        prompt = f"""{system_prompt}### User: {message}\n\n### Assistant:\n"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=1024) #128, 155, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, and 4096

        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        # Split the response based on "### Assistant:" and keep the part after it
        response = response.split("### Assistant:")[-1].strip()

        # Create a dictionary with just the assistant's response
        conversations = "\"\"" + response

        # Specify the output file path
        output_file = "conversation.txt"

        # Write the assistant's response to the plain text file
        with open(output_file, "a", encoding="utf-8") as file:
            file.write(conversations)

        print("Time taken by tokenizer (in seconds):", elapsed_time)
        
        torch.cuda.empty_cache()

        # Sleep for 5 seconds before running the model again
        time.sleep(5)

    except Exception as e:
        print("Error:", e)
