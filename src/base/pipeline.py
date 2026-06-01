from src.transformers.registry import TRANSFORMERS


def build_pipeline(config):

    processors = []

    for step_cfg in config["pipeline"]:
        op = step_cfg["op"]

        cls = TRANSFORMERS[op]

        kwargs = {k: v for k, v in step_cfg.items() if k != "op"}

        processors.append(cls(**kwargs))

    return processors


def run_pipeline(streams, pipeline):

    data = streams

    for processor in pipeline:
        data = processor.run(data)

    return data
