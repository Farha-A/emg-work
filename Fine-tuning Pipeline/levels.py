import pygame


class Level:
    def __init__(
        self,
        title,
        body,
    ):
        self.title = title
        self.body = body
        self.start_time = pygame.time.get_ticks() / 1000.0


def build_levels():
    return [
        Level(
            title="Do Nothing",
            body=(
                "Relax and do nothing with your bite. Keep your jaw relaxed for the whole interval."
            ),
        ),
        Level(
            title="Strong Bite",
            body=(
                "Bite down as firmly as you can for the duration. Try to maintain a strong, steady force."
            ),
        ),
        Level(
            title="Teeth in Contact",
            body=(
                "Just make sure your teeth are touching."
            ),
        ),
        Level(
            title="Preferred trigger bite strength",
            body=(
                "Use the bite strength you would normally use to trigger the device. Try to make it natural."
            ),
        ),
        Level(
            title="Smile",
            body=(
                "Smile as wide as you can for the duration. Try to maintain a strong, steady force."
            ),
        ),
        Level(
            title="Raise both eyebrows",
            body=(
                "Raise both eyebrows as high as you can for the duration. Try to stay in place."
            ),
        ),
        Level(
            title="Gulp",
            body=(
                "Swallow three times, take as long as you want."
            ),
        ),
    ]
