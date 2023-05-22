import elevenlabs
import functools
import re
import streamlit as st

import gui
import simulation


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


# maybe have a mixin?
# how do I mkae the init method initalize from both super classes?
class Speaker:
    def __init__(self, name, voice="Rachel"):
        self.name = name
        self.voice = voice

    def speak(self, message, sound_on, debug_sound):
        if sound_on:
            message = re.sub(r"\*.*?\*", "", message)
            text = message[:20] if debug_sound else message
            audio = elevenlabs.generate(text, voice=self.voice)
            st.audio(audio)

    def write(self, message):
        st.markdown("*" + message + "*")
        print(f"({self.name}): {message}")
        print("\n")

    def output(self, message, sound_on, debug_sound):
        self.write(message)
        self.speak(message, sound_on, debug_sound)


# this should be in a new file.
class Panelist(Speaker):
    def __init__(
        self,
        name,
        role,
        title,
        bio,
        url,
        icon_path,
        voice,
    ):
        super().__init__(name, voice)
        self.role = role
        self.title = title
        self.bio = bio
        self.url = url
        self.icon_path = icon_path

    def url_markdown(self):
        return f"[{self.name}]({self.url})"

    def write(self, message):
        with gui.Message(self) as m:
            m.write(f"{message}")
        print(f"({self.name}): {message}")
        print("\n")


def main():
    title = "[AISF](https://aisf.co/) Panel Simulation"
    director_name = "Hubert Thieblot"
    agent_summaries = [
        Panelist(
            name="Hubert Thieblot",
            role="moderator",
            title="Partner, Founders Inc",
            bio="You are an entrepreneur best known for founding Curse Inc., a global multimedia and technology company that provides content and services related to video games, including community forums, video game databases, live streaming, and eSports team management.",
            url="https://en.wikipedia.org/wiki/Hubert_Thieblot",
            icon_path="images/hubert.jpg",
            voice="Adam",  # premade by elevenlabs
        ),
        Panelist(
            name="Edward Saatchi",
            role="panelist",
            title="Founder, Fable Simulation",
            bio="You co-founded Fable Studio, which focuses on creating 'virtual beings' - characters that use AI and VR to provide interactive and immersive storytelling experiences, and you are interested in using agent simulations as a new paradigm for training AGIs.",
            url="https://en.wikipedia.org/wiki/Edward_Saatchi",
            icon_path="images/edward.jpeg",
            voice="Arnold",  # premade by elevenlabs
        ),
        Panelist(
            name="Jim Fan",
            role="panelist",
            title="AI Scientist, Nvidia",
            bio="You built MineDojo, which is a multitask benchamrking suite 1000s of open-ended and language-prompted tasks, where the AI agents can freely explore a procedurally generated 3D world with diverse terrains to roam, materials to mine, tools to craft, structures to build, and wonders to discover.",
            url="https://jimfan.me/",
            icon_path="images/jim.jpg",
            voice="Josh",  # premade by elevenlabs
        ),
        Panelist(
            name="Joon Park",
            role="panelist",
            title="PhD student, Stanford",
            bio="You are known for your research on generative agents, which are computational software agents based on LLMs that simulate believable human behavior, demonstrating a virtual societ of 25 NPCs that exhibit various emergent social behaviors.",
            url="http://www.joonsungpark.com/",
            icon_path="images/joon.jpg",
            voice="Sam",  # premade by elevenlabs
        ),
        Panelist(
            name="Jack Soslow",
            role="panelist",
            title="Partner, a16z",
            bio="You are an investor at Andreessen Horowitz who has made investments in games, AR, VR, and AI, and you are interested in multi-agent simulations from the perspective of alignment, specialization, colective intelligence, and cultural impacts.",
            url="https://twitter.com/JackSoslow",
            icon_path="images/jack.jpg",
            voice="Antoni",  # premade by elevenlabs
        ),
        Panelist(
            name="Michael Chang",
            role="panelist",
            title="Technical Staff, LangChain",
            bio="You are an AI researcher studying reinforcement learning and recursive self-improvement, and you have built various demonstrations of how to implement various multi-agent dialogue simulations using LangChain, a popualar framework for composing LLMs into powerful applications.",
            url="https://mbchang.github.io/",
            icon_path="images/michael.jpg",
            voice=create_voice_if_nonexistent(
                "Phoenix",
                gender=elevenlabs.Gender.male,
                age=elevenlabs.Age.young,
                accent=elevenlabs.Accent.american,
                accent_strength=1.0,
            ),
        ),
    ]

    names_to_panelists = {p.name: p for p in agent_summaries}

    (
        sound_on,
        debug_sound,
        termination_probability,
        openai_api_model,
    ) = gui.initialize_gui(title, agent_summaries, with_sound=True)

    topic = st.text_input(
        "Enter the topic for debate", "multi-agent chatbot simulations"
    )
    button = st.button("Run")

    if button:
        with st.spinner("Initializing simulation..."):
            agents, director, specified_topic = simulation.initialize_simulation(
                topic,
                agent_summaries,
                director_name,
                openai_api_model,
                termination_probability,
            )

        with st.spinner("Running simulation..."):
            elevenlabs.voices()  # initialize voices

            simulator = simulation.DialogueSimulator(
                agents=agents,
                selection_function=functools.partial(
                    simulation.select_next_speaker, director=director
                ),
            )
            simulator.reset()
            simulator.inject("Audience member", specified_topic)
            Speaker("Audience member").output(specified_topic, sound_on, debug_sound)

            while True:
                name, message = simulator.step()
                names_to_panelists[name].output(message, sound_on, debug_sound)
                if director.stop:
                    break
            st.write("*Finished discussion.*")


if __name__ == "__main__":
    main()
