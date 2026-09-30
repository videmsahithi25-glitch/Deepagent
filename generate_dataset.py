import json
from pathlib import Path

BASE_DIR = Path(__file__).parent


def load_json(filename):

    with open(
        BASE_DIR / filename,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def generate_examples(topics, templates):

    examples = []

    for _, topic_list in topics.items():

        for topic in topic_list:

            for _, template_list in templates.items():

                for template in template_list:

                    examples.append(
                        template.format(
                            topic=topic
                        )
                    )

    return sorted(
        list(
            set(examples)
        )
    )


def save_json(filename, data):

    with open(
        BASE_DIR / filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def main():

    topics = load_json(
        "topics.json"
    )

    question_templates = load_json(
        "question_templates.json"
    )

    topic_templates = load_json(
        "topic_templates.json"
    )

    question_examples = generate_examples(
        topics,
        question_templates
    )

    topic_examples = generate_examples(
        topics,
        topic_templates
    )

    save_json(
        "question_mode.json",
        question_examples
    )

    save_json(
        "topic_mode.json",
        topic_examples
    )

    print(
        f"Generated {len(question_examples)} question examples."
    )

    print(
        f"Generated {len(topic_examples)} topic examples."
    )


if __name__ == "__main__":

    main()