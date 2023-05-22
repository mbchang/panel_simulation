import elevenlabs


def create_new_voice():
    if not any(voice.name == "Phoenix" for voice in elevenlabs.voices()):
        # Build a voice deisgn object
        design = elevenlabs.VoiceDesign(
            name="Phoenix",
            text="The quick brown foxes jump over the lazy dogs, showcasing their agility and speed in a playful ways.",
            gender=elevenlabs.Gender.male,
            age=elevenlabs.Age.young,
            accent=elevenlabs.Accent.american,
            accent_strength=1.0,
        )
        elevenlabs.Voice.from_design(design)


def initialize_voices():
    # create new voice
    create_new_voice()

    # initialize voices
    elevenlabs.voices()
