import asyncio
from collections import OrderedDict
import elevenlabs
from enum import Enum
import functools
import os
from PIL import Image, ImageDraw, ImageOps
from playsound import playsound
import random
import re
import streamlit as st
import tenacity
from typing import List, Callable


from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

WITH_SOUND = True


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message


class IntegerOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."


class DirectorDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        model: ChatOpenAI,
        speakers: List[DialogueAgent],
        stopping_probability: float,
    ) -> None:
        super().__init__(name, system_message, model)
        self.speakers = speakers
        self.next_speaker = ""

        self.stop = False
        self.stopping_probability = stopping_probability
        self.termination_clause = "Finish the conversation by stating a concluding message and thanking everyone."
        self.continuation_clause = "Do not end the conversation. Keep the conversation going by adding your own ideas."

        # 1. have a prompt for generating a response to the previous speaker
        self.response_prompt_template = PromptTemplate(
            input_variables=["message_history", "termination_clause"],
            template=f"""{{message_history}}

Follow up with an insightful comment.
{{termination_clause}}
{self.prefix}
        """,
        )

        # 2. have a prompt for deciding who to speak next
        self.choice_parser = IntegerOutputParser(
            regex=r"<(\d+)>", output_keys=["choice"], default_output_key="choice"
        )
        self.choose_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "speaker_names"],
            template=f"""{{message_history}}

Given the above conversation, select a new speaker by choosing index next to their name:
{{speaker_names}}

{self.choice_parser.get_format_instructions()}

Do nothing else.
        """,
        )

        # 3. have a prompt for prompting the next speaker to speak
        self.prompt_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "next_speaker"],
            template=f"""{{message_history}}

The next speaker is {{next_speaker}}.
Prompt the next speaker to speak with an insightful question.
{self.prefix}
        """,
        )

    def _generate_response(self):
        # if self.stop = True, then we will inject the prompt with a termination clause
        sample = random.uniform(0, 1)
        self.stop = sample < self.stopping_probability

        print(f"\tStop? {self.stop}\n")

        response_prompt = self.response_prompt_template.format(
            message_history="\n".join(self.message_history),
            termination_clause=self.termination_clause if self.stop else "",
        )

        self.response = self.model(
            [
                self.system_message,
                HumanMessage(content=response_prompt),
            ]
        ).content

        return self.response

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),  # No waiting time between retries
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
        ),
        retry_error_callback=lambda retry_state: 0,
    )  # Default value when all retries are exhausted
    def _choose_next_speaker(self) -> str:
        speaker_names = "\n".join(
            [f"{idx}: {name}" for idx, name in enumerate(self.speakers)]
        )
        choice_prompt = self.choose_next_speaker_prompt_template.format(
            message_history="\n".join(
                self.message_history + [self.prefix] + [self.response]
            ),
            speaker_names=speaker_names,
        )

        choice_string = self.model(
            [
                self.system_message,
                HumanMessage(content=choice_prompt),
            ]
        ).content
        choice = int(self.choice_parser.parse(choice_string)["choice"])

        return choice

    def select_next_speaker(self):
        return self.chosen_speaker_id

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        # 1. generate and save response to the previous speaker
        self.response = self._generate_response()

        if self.stop:
            message = self.response
        else:
            # 2. decide who to speak next
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            print(f"\tNext speaker: {self.next_speaker}\n")

            # 3. prompt the next speaker to speak
            next_prompt = self.prompt_next_speaker_prompt_template.format(
                message_history="\n".join(
                    self.message_history + [self.prefix] + [self.response]
                ),
                next_speaker=self.next_speaker,
            )
            message = self.model(
                [
                    self.system_message,
                    HumanMessage(content=next_prompt),
                ]
            ).content
            message = " ".join([self.response, message])

        return message


def select_next_speaker(
    step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent
) -> int:
    """
    If the step is even, then select the director
    Otherwise, the director selects the next speaker.
    """
    # the director speaks on odd steps
    if step % 2 == 1:
        idx = 0
    else:
        # here the director chooses the next speaker
        idx = director.select_next_speaker() + 1  # +1 because we excluded the director
    return idx


def initialize_simulation(
    topic, agent_summaries, director_name, openai_api_model, termination_probability
):
    agent_summary_string = "\n- ".join(
        [""]
        + [f"{name}: {role}" for name, (role, description) in agent_summaries.items()]
    )

    conversation_description = f"""This is a panel discussion at the AI San Francisco Summit focusing on the topic: {topic}.

    The panel features {agent_summary_string}."""

    def generate_agent_header(agent_name, agent_role, agent_description):
        return f"""{conversation_description}

    Your name is {agent_name}, your role is {agent_role}.

    Your description is as follows: {agent_description}

    You are discussing the topic: {topic}.

    Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your background.
    """

    def generate_agent_system_message(agent_name, agent_header):
        return SystemMessage(
            content=(
                f"""{agent_header}
    Speak in the first person with the style and perspective of {agent_name}.
    For describing your own body movements, wrap your description in '*'.
    Be concise and limit your response to 30 words.
    """
            )
        )

    agent_headers = [
        generate_agent_header(name, role, description)
        for name, (role, description) in agent_summaries.items()
    ]
    agent_system_messages = [
        generate_agent_system_message(name, header)
        for name, header in zip(agent_summaries, agent_headers)
    ]

    for name, header, system_message in zip(
        agent_summaries, agent_headers, agent_system_messages
    ):
        print(f"\n\n{name}:")
        print(f"\nHeader:\n{header}")
        print(f"\nSystem Message:\n{system_message.content}")

    topic_specifier_prompt = [
        SystemMessage(content="You can make a topic more specific."),
        HumanMessage(
            content=f"""{conversation_description}

            Please elaborate on the topic to generate hype about communication and human-AI interaction.
            Frame the topic as a single question to be answered.
            Be creative and imaginative.
            Please reply with the specified topic in 50 words or less.
            Do not add anything else."""
        ),
    ]
    specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

    print(f"Original topic:\n{topic}\n")
    print(f"Detailed topic:\n{specified_topic}\n")
    # How can multi-agent chatbot simulations be used to revolutionize the way we interact with AI, and what implications might this have on the future of communication and human-AI collaboration?

    director = DirectorDialogueAgent(
        name=director_name,
        system_message=agent_system_messages[0],
        model=ChatOpenAI(model_name=openai_api_model, temperature=0.5),
        speakers=[name for name in agent_summaries if name != director_name],
        stopping_probability=termination_probability,
    )

    agents = [director]
    for name, system_message in zip(
        list(agent_summaries.keys())[1:], agent_system_messages[1:]
    ):
        agents.append(
            DialogueAgent(
                name=name,
                system_message=system_message,
                model=ChatOpenAI(model_name=openai_api_model, temperature=0.2),
            )
        )
    return agents, director, specified_topic


def initialize_gui(agent_summaries, director_name):
    title = "[AISF](https://aisf.co/) Panel Simulation"
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_title=title,
        layout="centered",
    )
    st.title(title)
    st.write(
        f"""Generate a fake panel discussion about any topic among fictitious renditions of
        [Hubert Thieblot](https://en.wikipedia.org/wiki/Hubert_Thieblot),
        [Edward Saatchi](https://en.wikipedia.org/wiki/Edward_Saatchi),
        [Jim Fan](https://jimfan.me/),
        [Joon Park](http://www.joonsungpark.com/),
        [Jack Soslow](https://twitter.com/JackSoslow), and
        [Michael Chang](https://mbchang.github.io/).
        {director_name} will moderate.
        """
    )

    st.sidebar.write(
        f"""
    # About

    The challenge faced by generating multi-agent simulations is determining the order in which the agents speak.
    To address this challenge, this application uses a "director" agent to decide the order in which the agents speak.
    You can find an example of how to implement this in the [LangChain docs](https://python.langchain.com/en/latest/use_cases/agent_simulations/multiagent_authoritarian.html).

    Created by [Michael Chang](https://mbchang.github.io/) ([@mmmbchang](https://twitter.com/mmmbchang)) at [LangChain](https://python.langchain.com/en/latest/).
    """
    )
    st.sidebar.write("---")
    openai_api_key = st.sidebar.text_input("Your OpenAI API KEY", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    if WITH_SOUND:
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

    st.sidebar.write("---")

    st.write("---")
    disclaimer_text = f"""
    # Disclaimer

    This application simulates a fake dialogue among fictitious renditions of real people.
    To protect the privacy of the individuals represented, note that:

    1. the content generated by this application is entirely synthetic and is not actual statements or opinions expressed by the individuals represented;
    2. the voices used in this application are synthetically generated by [Eleven Labs](https://eleven-labs.com/en/) and do not reflect the voices of the individuals represented.

    This application is intended for demonstration and entertainment purposes only.
    The creator of this application takes no responsibility for any actions taken based on its content.
    """
    st.sidebar.write(disclaimer_text)

    acknowledgements_text = f"""
    # Acknowledgements

    This application is powered by OpenAI and created with
    [LangChain](https://python.langchain.com/en/latest/),
    [Streamlit](https://streamlit.io/),
    and
    [Eleven Labs](https://beta.elevenlabs.io/).
    The UI design was inspired by [WhatIfGPT](https://github.com/realminchoi/whatifgpt).
    Thank you to the LangChain team for their support and feedback.
    """
    st.sidebar.write(acknowledgements_text)
    return sound_on, debug_sound, termination_probability, openai_api_model


def list_to_string(l):
    return ", ".join(l[:-1]) + ", and " + l[-1]


class Message:
    def __init__(self, name: str):
        self.name = name

        if self.name != "Audience member":
            icon_col, message_col = st.columns([1, 4], gap="medium")
            self.icon = self.get_icon(name)
            if name == "Hubert Thieblot":
                caption = f"{name}, Moderator"
            else:
                caption = f"{name}, Panelist"
            icon_col.image(self.icon, caption=caption)
            self.markdown = message_col.markdown

    def get_icon(self, name):
        if name == "Hubert Thieblot":
            icon_path = "images/hubert.jpg"
        elif name == "Edward Saatchi":
            icon_path = "images/edward.jpeg"
        elif name == "Jim Fan":
            icon_path = "images/jim.jpg"
        elif name == "Joon Park":
            icon_path = "images/joon.jpg"
        elif name == "Jack Soslow":
            icon_path = "images/jack.jpg"
        elif name == "Michael Chang":
            icon_path = "images/michael.jpg"
        else:
            raise ValueError(f"No picture for name: {name}")

        # Open the image file
        img = Image.open(icon_path)
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
        if self.name == "Audience member":
            st.markdown("*" + content + "*")  # Italicize the content
        else:
            self.markdown(content)


# async def output(name, message):
def output(name, message, sound_on=False, debug_sound=True):
    with Message(name) as m:
        m.write(f"{message}")
    print(f"({name}): {message}")
    print("\n")

    # remove text delimited by asterisks from message
    if sound_on and WITH_SOUND:
        message = re.sub(r"\*.*?\*", "", message)
        if debug_sound:
            play_voice(message[:20], name=name)
        else:
            play_voice(message, name=name)


def play_voice(text, name):
    if name == "Hubert Thieblot":
        voice = "Adam"
    elif name == "Edward Saatchi":
        voice = "Arnold"
    elif name == "Jim Fan":
        voice = "Josh"
    elif name == "Joon Park":
        voice = "Sam"
    elif name == "Jack Soslow":
        voice = "Antoni"
    elif name == "Michael Chang":
        voice = "Phoenix"
    elif name == "Audience member":
        voice = "Rachel"
    else:
        raise ValueError(f"No picture for name: {name}")

    voice = "Phoenix"

    # elevenlabs.play(elevenlabs.generate(text, voice=voice))
    audio = elevenlabs.generate(text, voice=voice)
    st.audio(audio)
    # elevenlabs.play(audio)
    # elevenlabs.save(audio, f"{name}.mpeg")
    # playsound(f"{name}.mpeg")
    # os.remove(f"{name}.mpeg")


def create_new_voice():
    if not any(voice.name == "Phoenix" for voice in elevenlabs.voices()):
        print("Creating new voice!!!!!!!")
        print("*" * 100)
        # Build a voice deisgn object
        design = elevenlabs.VoiceDesign(
            name="Phoenix",
            text="The quick brown fox jumps over the lazy dog, showcasing its agility and speed in a playful manner.",
            gender=elevenlabs.Gender.male,
            age=elevenlabs.Age.young,
            accent=elevenlabs.Accent.american,
            accent_strength=1.0,
        )
        elevenlabs.Voice.from_design(design)
    else:
        print("Not creating a new voice!")
        print("*" * 100)


def main():
    director_name = "Hubert Thieblot"
    agent_summaries = OrderedDict(
        {
            "Hubert Thieblot": (
                "Partner, Founders Inc",
                "You are an entrepreneur best known for founding Curse Inc., a global multimedia and technology company that provides content and services related to video games, including community forums, video game databases, live streaming, and eSports team management.",
            ),
            "Edward Saatchi": (
                "Founder, Fable Simulation",
                "You co-founded Fable Studio, which focuses on creating 'virtual beings' - characters that use AI and VR to provide interactive and immersive storytelling experiences, and you are interested in using agent simulations as a new paradigm for training AGIs.",
            ),
            "Jim Fan": (
                "AI Scientist, Nvidia",
                "You built MineDojo, which is a multitask benchamrking suite 1000s of open-ended and language-prompted tasks, where the AI agents can freely explore a procedurally generated 3D world with diverse terrains to roam, materials to mine, tools to craft, structures to build, and wonders to discover.",
            ),
            "Joon Park": (
                "PhD student, Stanford",
                "You are known for your research on generative agents, which are computational software agents based on LLMs that simulate believable human behavior, demonstrating a virtual societ of 25 NPCs that exhibit various emergent social behaviors.",
            ),
            "Jack Soslow": (
                "Partner, a16z",
                "You are an investor at Andreessen Horowitz who has made investments in games, AR, VR, and AI, and you are interested in multi-agent simulations from the perspective of alignment, specialization, colective intelligence, and cultural impacts.",
            ),
            "Michael Chang": (
                "Technical Staff, LangChain",
                "You are an AI researcher studying reinforcement learning and recursive self-improvement, and you have built various demonstrations of how to implement various multi-agent dialogue simulations using LangChain, a popualar framework for composing LLMs into powerful applications.",
            ),
        }
    )

    sound_on, debug_sound, termination_probability, openai_api_model = initialize_gui(
        agent_summaries, director_name
    )

    # create new voice
    create_new_voice()

    topic = st.text_input(
        "Enter the topic for debate", "multi-agent chatbot simulations"
    )
    button = st.button("Run")

    if button:
        with st.spinner("Initializing simulation..."):
            agents, director, specified_topic = initialize_simulation(
                topic,
                agent_summaries,
                director_name,
                openai_api_model,
                termination_probability,
            )

        with st.spinner("Running simulation..."):
            # initialize voices
            elevenlabs.voices()

            simulator = DialogueSimulator(
                agents=agents,
                selection_function=functools.partial(
                    select_next_speaker, director=director
                ),
            )
            simulator.reset()
            simulator.inject("Audience member", specified_topic)
            output("Audience member", specified_topic, sound_on, debug_sound)

            while True:
                name, message = simulator.step()
                # await output(name, message)
                output(name, message, sound_on, debug_sound)
                if director.stop:
                    break
            st.write("*Finished discussion.*")


if __name__ == "__main__":
    # asyncio.run(main())
    main()
