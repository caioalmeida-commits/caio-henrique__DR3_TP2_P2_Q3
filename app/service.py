import re
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

MODELO = "Helsinki-NLP/opus-mt-en-de"
LIMITE_TOKENS = 400


@dataclass
class Tradutor:
    cadeia: object
    tokenizer: object


def carregar_tradutor() -> Tradutor:
    llm = HuggingFacePipeline.from_model_id(
        model_id=MODELO,
        task="translation",
        pipeline_kwargs={"max_length": 512, "truncation": True},
    )
    cadeia = PromptTemplate.from_template("{texto}") | llm | StrOutputParser()
    return Tradutor(cadeia=cadeia, tokenizer=llm.pipeline.tokenizer)


def dividir_em_trechos(texto: str, tokenizer, limite: int = LIMITE_TOKENS) -> list[str]:
    frases = re.split(r"(?<=[.!?])\s+", texto.strip())
    trechos, atual, tamanho = [], [], 0
    for frase in frases:
        n = len(tokenizer.tokenize(frase))
        if atual and tamanho + n > limite:
            trechos.append(" ".join(atual))
            atual, tamanho = [], 0
        atual.append(frase)
        tamanho += n
    if atual:
        trechos.append(" ".join(atual))
    return trechos


def traduzir(tradutor: Tradutor, texto: str) -> tuple[str, int]:
    linhas, total_trechos = [], 0
    for linha in texto.split("\n"):
        if not linha.strip():
            linhas.append("")
            continue
        trechos = dividir_em_trechos(linha, tradutor.tokenizer)
        total_trechos += len(trechos)
        traducoes = tradutor.cadeia.batch([{"texto": t} for t in trechos])
        linhas.append(" ".join(t.strip() for t in traducoes))
    return "\n".join(linhas), total_trechos
