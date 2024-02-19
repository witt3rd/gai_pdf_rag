# System imports
import os

# Third-party imports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema.runnable import RunnableLambda
from langserve import CustomUserType, add_routes
from langchain.pydantic_v1 import Field

# Local imports
from app.pdf_rag import question_pdf

# Environment variables

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

#

app = FastAPI(
    title="PDF RAG Server",
    version="1.0",
    description="Standalone RAG server for PDF documents.",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class AskUrlRequest(CustomUserType):
    """Represents a request to ask a question about a resource specified by a URL.

    ATTENTION: Inherit from CustomUserType instead of BaseModel otherwise
        the server will decode it into a dict instead of a pydantic model.

    Attributes:
        url (str): The URL of the resource to ask the question about.
        question (str): The question to ask about the resource.
    """

    url: str = Field(..., extra={"widget": {"type": "text"}})
    question: str = Field(..., extra={"widget": {"type": "text"}})


def _ask_url(request: AskUrlRequest) -> str:
    return question_pdf(request.url, request.question)


add_routes(
    app,
    RunnableLambda(_ask_url).with_types(input_type=AskUrlRequest),
    config_keys=["configurable"],
    path="/pdf",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", "8000")),
    )
