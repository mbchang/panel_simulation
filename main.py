import elevenlabs
import streamlit as st
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

import config
import ui
import dialogue


def generate_detailed_topic(conversation_description):
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
    return specified_topic


def select_next_speaker(step: int, agents: List[dialogue.DialogueAgent]) -> int:
    """
    If the step is even, then select the director (index 0)
    Otherwise, the director selects the next speaker.
    """
    # validate that the first agent is instance of DirectorDialogueAgent
    assert isinstance(agents[0], dialogue.DirectorDialogueAgent)
    director = agents[0]
    # validate all other agents are not the director
    assert all([agent != director for agent in agents[1:]])

    # the director speaks on odd steps
    if step % 2 == 1:
        idx = 0
    else:
        # here the director chooses the next speaker
        idx = director.select_next_speaker() + 1  # +1 because we excluded the director
    return idx


def main():
    title = "[AISF](https://aisf.co/) Panel Simulation"
    agent_cfgs = [
        config.PanelistConfig(
            name="Hubert Thieblot",
            role="director",
            title="Partner, Founders Inc",
            bio="You are an entrepreneur best known for founding Curse Inc., a global multimedia and technology company that provides content and services related to video games, including community forums, video game databases, live streaming, and eSports team management.",
            url="https://en.wikipedia.org/wiki/Hubert_Thieblot",
            icon_path="images/hubert.jpg",
            voice="Adam",  # premade by elevenlabs
        ),
        config.PanelistConfig(
            name="Edward Saatchi",
            role="panelist",
            title="Founder, Fable Simulation",
            bio="You co-founded Fable Studio, which focuses on creating 'virtual beings' - characters that use AI and VR to provide interactive and immersive storytelling experiences, and you are interested in using agent simulations as a new paradigm for training AGIs.",
            url="https://en.wikipedia.org/wiki/Edward_Saatchi",
            icon_path="images/edward.jpeg",
            voice="Arnold",  # premade by elevenlabs
        ),
        config.PanelistConfig(
            name="Jim Fan",
            role="panelist",
            title="AI Scientist, Nvidia",
            bio="You built MineDojo, which is a multitask benchamrking suite 1000s of open-ended and language-prompted tasks, where the AI agents can freely explore a procedurally generated 3D world with diverse terrains to roam, materials to mine, tools to craft, structures to build, and wonders to discover.",
            url="https://jimfan.me/",
            icon_path="images/jim.jpg",
            voice="Josh",  # premade by elevenlabs
        ),
        config.PanelistConfig(
            name="Joon Park",
            role="panelist",
            title="PhD student, Stanford",
            bio="You are known for your research on generative agents, which are computational software agents based on LLMs that simulate believable human behavior, demonstrating a virtual societ of 25 NPCs that exhibit various emergent social behaviors.",
            url="http://www.joonsungpark.com/",
            icon_path="images/joon.jpg",
            voice="Sam",  # premade by elevenlabs
        ),
        config.PanelistConfig(
            name="Jack Soslow",
            role="panelist",
            title="Partner, a16z",
            bio="You are an investor at Andreessen Horowitz who has made investments in games, AR, VR, and AI, and you are interested in multi-agent simulations from the perspective of alignment, specialization, colective intelligence, and cultural impacts.",
            url="https://twitter.com/JackSoslow",
            icon_path="images/jack.jpg",
            voice="Antoni",  # premade by elevenlabs
        ),
        config.PanelistConfig(
            name="Michael Chang",
            role="panelist",
            title="Technical Staff, LangChain",
            bio="You are an AI researcher studying reinforcement learning and recursive self-improvement, and you have built various demonstrations of how to implement various multi-agent dialogue simulations using LangChain, a popualar framework for composing LLMs into powerful applications.",
            url="https://mbchang.github.io/",
            icon_path="images/michael.jpg",
            voice=ui.create_voice_if_nonexistent(
                "Phoenix",
                gender=elevenlabs.Gender.male,
                age=elevenlabs.Age.young,
                accent=elevenlabs.Accent.american,
                accent_strength=1.0,
            ),
        ),
    ]

    user_cfg = ui.initialize_gui(title, agent_cfgs)
    topic = st.text_input(
        "Enter the topic for debate", "multi-agent chatbot simulations"
    )
    button = st.button("Run")

    if button:
        with st.spinner("Initializing simulation..."):
            description = f"""This is a panel discussion at the AI San Francisco Summit focusing on the topic: {topic}.

The panel features {config.get_summary(agent_cfgs)}."""

            specified_topic = generate_detailed_topic(description)
            print(f"Original topic:\n{topic}\n")
            print(f"Detailed topic:\n{specified_topic}\n")

            model = ChatOpenAI(model_name=user_cfg.gpt_model, temperature=0.5)

            director_config = config.get_director(agent_cfgs)
            director = dialogue.DirectorDialogueAgent(
                name=director_config.name,
                system_message=director_config.generate_system_message(description),
                model=model,
                speakers=[agent.name for agent in config.get_panelists(agent_cfgs)],
                stopping_probability=user_cfg.termination_probability,
            )

            agents = [director]
            for agent_cfg in config.get_panelists(agent_cfgs):
                agents.append(
                    dialogue.DialogueAgent(
                        name=agent_cfg.name,
                        system_message=agent_cfg.generate_system_message(description),
                        model=model,
                    )
                )

            for agent in agents:
                print(f"\n\n{agent.name}:")
                print(f"\nSystem Message:\n{agent.system_message.content}")

            speakers = {}
            for agent_cfg in agent_cfgs:
                speakers[agent_cfg.name] = ui.VisibleSpeaker(
                    name=agent_cfg.name,
                    role=agent_cfg.role,
                    icon_path=agent_cfg.icon_path,
                    voice=agent_cfg.voice,
                )

        with st.spinner("Running simulation..."):
            elevenlabs.voices()  # initialize voices

            simulator = dialogue.DialogueSimulator(
                agents=agents, selection_function=select_next_speaker
            )
            simulator.reset()
            simulator.inject("Audience member", specified_topic)
            ui.Speaker("Audience member").output(
                specified_topic, user_cfg.sound_on, user_cfg.debug_sound
            )

            while True:
                name, message = simulator.step()
                speakers[name].output(message, user_cfg.sound_on, user_cfg.debug_sound)
                if director.stop:
                    break
            st.write("*Finished discussion.*")


if __name__ == "__main__":
    main()
