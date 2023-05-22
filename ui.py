from dataclasses import dataclass
import os
from PIL import Image, ImageDraw, ImageOps
import re

import elevenlabs
import streamlit as st

import panel
import ui

##########
# VISUAL #
##########


@dataclass
class UserConfig:
    openai_api_key: str = ""
    eleven_api_key: str = ""
    sound_on: bool = False
    debug_sound: bool = False
    termination_probability: float = 0.01
    gpt_model: str = "gpt-3.5-turbo"


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


def configure():
    openai_api_key = st.sidebar.text_input("Your OpenAI API KEY", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    sound_on = st.sidebar.checkbox("Enable Sound")
    if sound_on:
        eleven_api_key = st.sidebar.text_input(
            "Your Eleven Labs API Key", type="password"
        )
        os.environ["ELEVEN_API_KEY"] = eleven_api_key
    else:
        st.sidebar.write("Sound is turned off.")
        eleven_api_key = ""

    debug_sound = st.sidebar.checkbox("Debug Sound")

    termination_probability = st.sidebar.slider(
        label="Termination Probability",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=0.01,
    )

    gpt_model = st.sidebar.selectbox("Model name", options=["gpt-3.5-turbo", "gpt-4"])
    return UserConfig(
        openai_api_key=openai_api_key,
        eleven_api_key=eleven_api_key,
        sound_on=sound_on,
        debug_sound=debug_sound,
        termination_probability=termination_probability,
        gpt_model=gpt_model,
    )


def site_description(agent_cfgs):
    director_name = panel.get_director(agent_cfgs).name
    return f"""Generate a fake panel discussion about any topic among fictitious renditions of
        {list_to_string([agent.url_markdown() for agent in agent_cfgs])}. {director_name} will moderate.
        """


def initialize_gui(title, agent_cfgs):
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_title=title,
        layout="centered",
    )
    st.title(title)
    st.write(site_description(agent_cfgs))
    st.write("---")

    with st.sidebar:
        st.write(about_text())
        st.write("---")
        user_config = configure()
        st.write("---")
        st.write(dislaimer_text())
        st.write(acknowledgements_text())
    return user_config


class Message:
    def __init__(self, panelist):
        self.panelist = panelist

        icon_col, message_col = st.columns([1, 4], gap="medium")
        self.icon = self.get_icon(self.panelist.name)
        title = "Moderator" if self.panelist.role == "director" else "Panelist"
        caption = f"{self.panelist.name}, {title}"
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


#########
# AUDIO #
#########


def create_voice_if_nonexistent(name, gender, age, accent, accent_strength):
    """Create a new voice if it does not exist."""
    if not any(voice.name == name for voice in elevenlabs.voices()):
        design = elevenlabs.VoiceDesign(
            name="Phoenix",
            text="The quick brown foxes jump over the lazy dogs, showcasing their agility and speed in a playful ways.",
            gender=gender,
            age=age,
            accent=accent,
            accent_strength=accent_strength,
        )
        elevenlabs.Voice.from_design(design)
    return name


class Speaker:
    def __init__(self, name, voice="Rachel"):
        self.name = name
        self.voice = voice

    def speak(self, message):
        message = re.sub(r"\*.*?\*", "", message)
        audio = elevenlabs.generate(message, voice=self.voice)
        st.audio(audio)

    def write(self, message):
        st.markdown("*" + message + "*")
        print(f"({self.name}): {message}")
        print("\n")

    def output(self, message, sound_on, debug_sound):
        self.write(message)
        if sound_on:
            if debug_sound:
                message = message[:20]
            self.speak(message)


class VisibleSpeaker(Speaker):
    def __init__(self, name, role, icon_path, voice):
        Speaker.__init__(self, name, voice)
        self.role = role
        self.icon_path = icon_path

    @classmethod
    def from_config(cls, cfg: panel.PanelistConfig):
        return cls(
            name=cfg.name, role=cfg.role, voice=cfg.voice, icon_path=cfg.icon_path
        )

    def write(self, message):
        with ui.Message(self) as m:
            m.write(f"{message}")
        print(f"({self.name}): {message}")
        print("\n")
