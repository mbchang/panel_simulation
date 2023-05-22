import os
from PIL import Image, ImageDraw, ImageOps
import streamlit as st


def list_to_string(l):
    return ", ".join(l[:-1]) + ", and " + l[-1]


def about_text():
    return f"""
    # About

    The challenge faced by generating multi-agent simulations is determining the order in which the agents speak.
    To address this challenge, this application uses a "director" agent to decide the order in which the agents speak.
    You can find an example of how to implement this in the [LangChain docs](https://python.langchain.com/en/latest/use_cases/agent_simulations/multiagent_authoritarian.html).

    Created by [Michael Chang](https://mbchang.github.io/) ([@mmmbchang](https://twitter.com/mmmbchang)) at [LangChain](https://python.langchain.com/en/latest/).
    """


def dislaimer_text():
    return f"""
    # Disclaimer

    This application simulates a fake dialogue among fictitious renditions of real people.
    To protect the privacy of the individuals represented, note that:

    1. the content generated by this application is entirely synthetic and is not actual statements or opinions expressed by the individuals represented;
    2. the voices used in this application are synthetically generated by [Eleven Labs](https://eleven-labs.com/en/) and do not reflect the voices of the individuals represented.

    This application is intended for demonstration and entertainment purposes only.
    The creator of this application takes no responsibility for any actions taken based on its content.
    """


def acknowledgements_text():
    return f"""
    # Acknowledgements

    This application is powered by OpenAI and created with
    [LangChain](https://python.langchain.com/en/latest/),
    [Streamlit](https://streamlit.io/),
    and
    [Eleven Labs](https://beta.elevenlabs.io/).
    The UI design was inspired by [WhatIfGPT](https://github.com/realminchoi/whatifgpt).
    Thank you to the LangChain team for their support and feedback.
    """


def config(with_sound):
    openai_api_key = st.sidebar.text_input("Your OpenAI API KEY", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    if with_sound:
        sound_on = st.sidebar.checkbox("Enable Sound")
        if sound_on:
            eleven_api_key = st.sidebar.text_input(
                "Your Eleven Labs API Key", type="password"
            )
            os.environ["ELEVEN_API_KEY"] = eleven_api_key
        else:
            st.sidebar.write("Sound is turned off.")

        debug_sound = st.sidebar.checkbox("Debug Sound")
    else:
        sound_on = False
        debug_sound = False

    termination_probability = st.sidebar.slider(
        label="Termination Probability",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=0.01,
    )

    openai_api_model = st.sidebar.selectbox(
        "Model name", options=["gpt-3.5-turbo", "gpt-4"]
    )
    return sound_on, debug_sound, termination_probability, openai_api_model


def site_description(agent_summaries):
    directors = [agent for agent in agent_summaries if agent.role == "moderator"]
    assert len(directors) == 1
    director_name = directors[0].name

    return f"""Generate a fake panel discussion about any topic among fictitious renditions of
        {list_to_string([agent.url_markdown() for agent in agent_summaries])}. {director_name} will moderate.
        """


def initialize_gui(title, agent_summaries, with_sound):
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_title=title,
        layout="centered",
    )
    st.title(title)
    st.write(site_description(agent_summaries))
    st.write("---")

    with st.sidebar:
        st.write(about_text())
        st.write("---")
        sound_on, debug_sound, termination_probability, openai_api_model = config(
            with_sound
        )
        st.write("---")
        st.write(dislaimer_text())
        st.write(acknowledgements_text())

    return sound_on, debug_sound, termination_probability, openai_api_model


class Message:
    def __init__(self, panelist):
        self.panelist = panelist

        icon_col, message_col = st.columns([1, 4], gap="medium")
        self.icon = self.get_icon(self.panelist.name)
        caption = f"{self.panelist.name}, {self.panelist.role}"
        icon_col.image(self.icon, caption=caption)
        self.markdown = message_col.markdown

    def get_icon(self, name):
        # Open the image file
        img = Image.open(self.panelist.icon_path)
        # Create a circular mask
        mask = Image.new("L", img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0) + img.size, fill=255)
        # Apply the mask and save the result
        icon = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
        icon.putalpha(mask)

        return icon

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def write(self, content):
        self.markdown(content)
