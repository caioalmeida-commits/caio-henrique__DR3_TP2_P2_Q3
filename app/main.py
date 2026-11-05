import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app import service
from app.schemas import TraducaoRequest, TraducaoResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tradutor = service.carregar_tradutor()
    yield


app = FastAPI(
    title="Tradução inglês-alemão com Hugging Face e LangChain",
    description="Recebe um texto em inglês e devolve a tradução para o alemão com o modelo Helsinki-NLP/opus-mt-en-de, gerenciado pelo LangChain.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", summary="Estado da API")
def health():
    return {"status": "ok", "modelo": service.MODELO}


@app.post("/traduzir", response_model=TraducaoResponse, summary="Traduz um texto do inglês para o alemão")
def traduzir(dados: TraducaoRequest, request: Request):
    inicio = time.perf_counter()
    traducao, trechos = service.traduzir(request.app.state.tradutor, dados.texto)
    return TraducaoResponse(
        texto_original=dados.texto,
        traducao=traducao,
        trechos=trechos,
        tempo_ms=round((time.perf_counter() - inicio) * 1000, 1),
    )
