# %%
## Dependencies
# pip install gradio

# %%
## Importing libraries 

import gradio as gr
from chatbot_tools import chat  # your chat() function from chatbot_tools.py
import pprint

# %% [markdown]
# ## Initializing memory

# %%
initial_history = [
    {
        "role": "assistant",
        "content": "Hi, Welcome to the chatbot for CompTIA Security 701+ Exam preparation."
    },
    {
        "role": "assistant",
        "content": "I am here to help you with your exam preparation."
    },
    {
        "role": "assistant",
        "content": "How may I help you?"
    },
]

# %% [markdown]
# ## Retriving chat funtion using memory

# %%
def chat_fn(user_input, history):
    """
    Gradio callback for Chatbot in messages format.
    Each message is a dict: {"role": "user"/"assistant", "content": "<string>"}
    """
    if history is None:
        history = []

    if not user_input.strip():
        return history

    print(f"\n--- Chat Function Called ---")
    print(f"User Input: {user_input}")

    # Append user message
    history.append({"role": "user", "content": user_input})

    # Call your existing chat function
    answer = chat(user_input)
    print(f"Raw Answer from chat(): {answer}")

    # Append assistant message
    if isinstance(answer, str) and answer.strip():
        history.append({"role": "assistant", "content": answer})
    else:
        # Crucial Debug: If the answer is not a string, we need to know why.
        print(f"Warning: chat() did not return a valid string answer. Type: {type(answer)}")
        history.append({"role": "assistant", "content": "An error occurred fetching the response."})

    print("New History to be returned:")
    pprint.pprint(history) #Pretty print the history for inspection
    print("----------------------------\n")

    return history



with gr.Blocks(title="Security+ SY0-701 Tutor Bot") as demo:
    gr.Markdown(
        """
        # Security+ SY0-701 Tutor Bot

        Ask anything about CompTIA Security+ (SY0-701), including concepts,
        exam objectives, and content from your course videos (`clean_text.csv`)
        and PDFs (`pdf.csv`) indexed in Pinecone.
        """
    )

    # Gradio Chatbot (messages format)
    chatbot = gr.Chatbot(label="Security+ Tutor", height=500, value=initial_history)
    state = gr.State(initial_history.copy())  # holds message history including greeting

    # Input textbox and Send button
    txt = gr.Textbox(
        label="Your question",
        placeholder="Type your Security+ question here and press Enter or click Send...",
        lines=2,
        autofocus=True,
    )
    send_btn = gr.Button("Send", variant="primary")

    # Wrapper to update chatbot and clear textbox
    def wrapper(user_input, history):
        new_history = chat_fn(user_input, history)
        return "", new_history, new_history

    # Send button click
    send_btn.click(
        fn=wrapper,
        inputs=[txt, state],
        outputs=[txt, chatbot, state],
    )

    # Press Enter in textbox
    txt.submit(
        fn=wrapper,
        inputs=[txt, state],
        outputs=[txt, chatbot, state],
    )

if __name__ == "__main__":
    demo.launch()



