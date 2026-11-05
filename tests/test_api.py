import os

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app import service
from app.main import app


class TokenizerFalso:
    def tokenize(self, texto):
        return texto.split()


@pytest.fixture
def client():
    llm = FakeListLLM(responses=["Guten Morgen", "Wie geht es dir?"])
    cadeia = PromptTemplate.from_template("{texto}") | llm | StrOutputParser()
    app.state.tradutor = service.Tradutor(cadeia=cadeia, tokenizer=TokenizerFalso())
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["modelo"] == "Helsinki-NLP/opus-mt-en-de"


def test_traduz_com_a_cadeia_do_langchain(client):
    corpo = client.post("/traduzir", json={"texto": "Good morning."}).json()
    assert corpo["traducao"] == "Guten Morgen"
    assert corpo["trechos"] == 1


def test_texto_longo_vira_varios_trechos_no_batch(client):
    texto = " ".join(f"Sentence number {i} has some words." for i in range(100))
    corpo = client.post("/traduzir", json={"texto": texto}).json()
    assert corpo["trechos"] > 1
    assert corpo["traducao"].count("Guten Morgen") + corpo["traducao"].count("Wie geht es dir?") == corpo["trechos"]


@pytest.mark.parametrize("corpo", [{}, {"texto": ""}, {"texto": "x" * 10001}])
def test_entradas_invalidas(client, corpo):
    assert client.post("/traduzir", json=corpo).status_code == 422


@pytest.mark.skipif(os.getenv("RODAR_MODELO_REAL") != "1", reason="define RODAR_MODELO_REAL=1 para usar o modelo de verdade")
def test_modelo_real():
    with TestClient(app) as client:
        corpo = client.post("/traduzir", json={"texto": "Good morning, my friend."}).json()
    assert "morgen" in corpo["traducao"].lower()
