from collections import OrderedDict
import functools
import re
import streamlit as st

import voices
import gui
import simulation


def output(name, message, sound_on=False, debug_sound=True):
    with gui.Message(name) as m:
        m.write(f"{message}")
    print(f"({name}): {message}")
    print("\n")

    # remove text delimited by asterisks from message
    if sound_on:
        message = re.sub(r"\*.*?\*", "", message)
        if debug_sound:
            voices.play_voice(message[:20], name=name)
        else:
            voices.play_voice(message, name=name)


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

    (
        sound_on,
        debug_sound,
        termination_probability,
        openai_api_model,
    ) = gui.initialize_gui(director_name, with_sound=True)

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
            voices.initialize_voices()

            simulator = simulation.DialogueSimulator(
                agents=agents,
                selection_function=functools.partial(
                    simulation.select_next_speaker, director=director
                ),
            )
            simulator.reset()
            simulator.inject("Audience member", specified_topic)
            output("Audience member", specified_topic, sound_on, debug_sound)

            while True:
                name, message = simulator.step()
                output(name, message, sound_on, debug_sound)
                if director.stop:
                    break
            st.write("*Finished discussion.*")


if __name__ == "__main__":
    main()
