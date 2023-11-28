import gradio as gr
import transformers
from torch import bfloat16
# from dotenv import load_dotenv  # if you wanted to adapt this for a repo that uses auth
from threading import Thread
from gradio.themes.utils.colors import Color
from peft import PeftModel
from peft.tuners.lora import LoraLayer


#HF_AUTH = os.getenv('HF_AUTH')
#model_id = "stabilityai/StableBeluga2" # 70B parm model based off Llama 2 70B
model_id = "TheBloke/StableBeluga-7B-GPTQ" # the lil guy.
adapter_path = './StableBeluga-7B-CrossAdpt10k/' #StableBeluga-7B-CrossAdpt_10k_estiloStable

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map='auto',
    #low_cpu_mem_usage=True
    #use_auth_token=HF_AUTH
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True
    #use_auth_token=HF_AUTH
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

text_color = "#FFFFFF"
app_background = "#0A0A0A"
user_inputs_background = "#193C4C"#14303D"#"#091820"
widget_bg = "#000100"
button_bg = "#141414"

dark = Color(
    name="dark",
    c50="#F4F3EE",  # not sure
    # all text color:
    c100=text_color, # Title color, input text color, and all chat text color.
    c200=text_color, # Widget name colors (system prompt and "chatbot")
    c300="#F4F3EE", # not sure
    c400="#F4F3EE", # Possibly gradio link color. Maybe other unlicked link colors.
    # suggestion text color...
    c500=text_color, # text suggestion text. Maybe other stuff.
    c600=button_bg,#"#444444", # button background color, also outline of user msg.
    # user msg/inputs color:
    c700=user_inputs_background, # text input background AND user message color. And bot reply outline.
    # widget bg.
    c800=widget_bg, # widget background (like, block background. Not whole bg), and bot-reply background.
    c900=app_background, # app/jpage background. (v light blue)
    c950="#F4F3EE", # not sure atm. 
)

DESCRIPTION = """
# Cross StableBeluga2 7B Chat 🗨️
This is a streaming Chat Interface implementation of [StableBeluga2](https://huggingface.co/stabilityai/StableBeluga2) 
Hosted on My house :D

Sometimes you will get an empty reply, just hit the "Retry" button.
Also sometimes model wont stop itself from generating. Again, try a retry here.
"""

SYS_PROMPT_EXPLAIN = """# System Prompt
A system prompt can be used to guide model behavior. See the examples for an idea of this, but feel free to write your own!"""

prompts = [
    "You are an AI that follows instructions extremely well. Help as much as you can",
    "You are an Assistant that follows instructions extremely well. Help as much as you can"
]

def prompt_build(system_prompt, user_input, history):
    prompt = f"""### System:\n{system_prompt}\n\n"""
    
    for pair in history:
        prompt += f"""### User:\n{pair[0]}\n\n### Assistant:\n{pair[1]}\n\n"""

    prompt += f"""### User:\n{user_input}\n\n### Assistant:"""
    return prompt

def chat(user_input, history, system_prompt):

    prompt = prompt_build(system_prompt, user_input, history)
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    streamer = transformers.TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256, # will override "max_len" if set. 128, 155, 160, 192, 256, 384, 512, 768, 1024, 1536, 2048, and 4096
        #max_length=1024,
        do_sample=True,
        top_p=0.95,
        temperature=0.6,
        top_k=50
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    model_output = ""
    for new_text in streamer:
        model_output += new_text
        if "### END." in model_output:
            break
        yield model_output
    return model_output

with gr.Blocks(theme=gr.themes.Monochrome(
               font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
               primary_hue="sky",  # when loading
               secondary_hue="sky", # something with links
               neutral_hue="dark"),) as demo:  #main.

    gr.Markdown(DESCRIPTION)
    gr.Markdown(SYS_PROMPT_EXPLAIN)
    dropdown = gr.Dropdown(choices=prompts, label="Type your own or select a system prompt", value="You are a helpful AI.", allow_custom_value=True)
    chatbot = gr.ChatInterface(fn=chat, additional_inputs=[dropdown])

demo.queue(api_open=False).launch(show_api=False,share=True)