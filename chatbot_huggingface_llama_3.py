import os

import replicate
import requests
from dotenv import load_dotenv
from streamlit_chat import message
import streamlit as st
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def initialize_session_state():
    load_dotenv()
    # setup streamlit page
    st.set_page_config(
        page_title="llama-3 Chatbot",
        page_icon="🦙"
    )

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []


def messages_to_prompt(system_message, messages):
    return_messages = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
    for msg in messages:
        if msg.type == "human":
            return_messages += f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
        elif msg.type == "ai":
            return_messages += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    return_messages += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return return_messages


def generate_output(prompt_template):
    API_URL = os.getenv("HF_API_URL")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt_template,
        "parameters": {
            "return_full_text": False
        },

    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Error in generate_output: {response.text}")
    else:
        return response.json()[0]['generated_text']


def generate_chat_response():
    chat_container = st.container()
    container = st.container()

    with st.sidebar:
        system_message = st.text_input("system message", value="You are a helpful assistant.")
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask anything", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            st.session_state.messages.append(
                HumanMessage(content=user_input))
            message_prompt = messages_to_prompt(system_message, st.session_state.messages)
            with st.spinner("Thinking..."):
                output = generate_output(message_prompt)
            st.session_state.messages.append(
                AIMessage(content=output))

        if st.session_state['generated']:
            with chat_container:
                messages = st.session_state['messages']
                # Iterate over messages in chunks of 3
                for i, msg in enumerate(messages):
                    if i % 2 == 0:
                        message(msg.content, is_user=True, key=str(i) + '_user')
                    else:
                        message(msg.content, is_user=False, key=str(i) + '_ai')


def main():
    initialize_session_state()
    st.header("llama-3 Chatbot 🦙🤖")
    generate_chat_response()


if __name__ == '__main__':
    main()
