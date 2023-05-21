import random
import tenacity
from typing import List, Callable

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)


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
