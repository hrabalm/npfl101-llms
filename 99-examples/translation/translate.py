import click
from tqdm import tqdm

TEMPLATE = """### Instruction
Translate Input from English to Japanese
### Input
{text}
### Response
"""


def apply_template(text: str, src_lang: str, tgt_lang: str, template: str):
    return template.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang)


class HFModel:
    def __init__(self, model_name_or_path, model_kwargs={}, tokenizer_kwargs={}):
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs
        )

    def _generate(self, prompts_batch, max_new_tokens, generate_kwargs={}):
        input_ids = self.tokenizer(
            prompts_batch,
            return_tensors="pt",
            padding=True,
            padding_side="left",  # type: ignore
        ).to(self.model.device)

        outputs = self.model.generate(
            **input_ids, max_new_tokens=max_new_tokens, **generate_kwargs
        )

        # slice prefix (prompt)
        outputs = [
            out[inputs.shape[0] :]
            for out, inputs in zip(outputs, input_ids["input_ids"])
        ]

        import torch

        @torch.jit.script
        def end_index(tensor, eos_token_id: int):
            """Returns index of first eos token in tensor or its length."""
            # not really sure whether the JIT helps here
            # TODO: hyperfine this versus the naive python variant
            for idx, token in enumerate(tensor):
                if token == eos_token_id:
                    return idx
            return tensor.shape[-1]

        # print(end_index.code)

        # slice suffix (eos)
        outputs = [
            out[: end_index(out, self.tokenizer.eos_token_id)] for out in outputs
        ]

        outputs_str = [self.tokenizer.decode(x) for x in outputs]

        return outputs_str

    def generate(
        self, prompts: list[str], max_new_tokens=128, max_batch_size=16, seed=42
    ):
        from transformers import set_seed

        outputs = []

        prompts_queue = prompts.copy()
        batch_size = max_batch_size

        with tqdm(total=len(prompts), desc="Translating") as pbar:
            while len(prompts_queue) > 0:
                try:
                    batch = prompts_queue[:batch_size]
                    set_seed(seed)
                    partial_outputs = self._generate(
                        batch, max_new_tokens=max_new_tokens
                    )
                    outputs.extend(partial_outputs)
                    prompts_queue = prompts_queue[batch_size:]
                    pbar.update(batch_size)
                    # batch_size = max_batch_size  # reset the batch_size, this could be optimized with some memory to see whether we ever succeed
                    # FIXME: raise batch size sometimes
                except:
                    batch_size //= 2
                    if batch_size == 0:
                        raise ValueError("Failed to generate output with batch_size=1")

        assert len(prompts) == len(outputs)
        return outputs


class Model:
    def __init__(self, model_path, template):
        import torch

        self._model = HFModel(
            model_path,
            model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "cuda"},
        )
        self._template = template

    def translate(self, inputs: list[str]):
        formatted_inputs = [
            apply_template(text, "", "", self._template) for text in inputs
        ]
        outputs = self._model.generate(
            formatted_inputs, max_new_tokens=128, max_batch_size=512
        )
        return outputs


@click.command()
@click.option("--input", "-i", type=click.File("r"), required=True)
@click.option("--output", "-o", type=click.File("w", lazy=True), required=True)
@click.option("--model", "-m", required=True)
def cli(input, output, model):
    model = Model(model, TEMPLATE)

    inputs = [x.rstrip("\n") for x in input.readlines()]
    outputs = model.translate(inputs)

    for tgt in outputs:
        output.write(f"{tgt}\n")


if __name__ == "__main__":
    cli()
