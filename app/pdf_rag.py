# System imports
import base64
from datetime import datetime, timedelta
import json
import os
import requests
import uuid

# Third party imports
from dotenv import load_dotenv
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
import redis
from unstructured.partition.pdf import partition_pdf

#

load_dotenv()

# Constants
PROJECT_NAME = os.environ["LANGCHAIN_PROJECT"]

ID_KEY = "doc_id"

DATA_DIR = os.getenv("DATA_DIR", "data")

SUMMARIZE_TEXT_PROMPT_TEXT = """Summarize the following text: {element} """
SUMMARIZE_TABLE_PROMPT_TEXT = """Summarize the following table: {element} """

# Globals

_db = redis.Redis(host="localhost", port=6379, decode_responses=True)

_text_model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    max_tokens=1024,
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
_vision_model = ChatOpenAI(
    model="gpt-4-vision-preview",
    max_tokens=1024,
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
_query_model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    # model="gpt-4-turbo-preview",
    max_tokens=1024,
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

_summarize_text_prompt = ChatPromptTemplate.from_template(SUMMARIZE_TEXT_PROMPT_TEXT)
_summarize_table_prompt = ChatPromptTemplate.from_template(SUMMARIZE_TABLE_PROMPT_TEXT)

_summarize_text_chain = (
    {"element": lambda x: x} | _summarize_text_prompt | _text_model | StrOutputParser()
)
_summarize_table_chain = (
    {"element": lambda x: x} | _summarize_table_prompt | _text_model | StrOutputParser()
)

# Logging


def _redis_log_sink(event):
    record = event.record
    channel_name = f"log-{PROJECT_NAME}-{record['module']}"
    output = {
        "type": record["level"].name,
        "module": record["name"],
        "function": record["function"],
        "line_number": record["line"],
        "message": record["message"],
        "timestamp": record["time"].isoformat(),
    }

    _db.publish(channel_name, json.dumps(output))


logger.add(
    _redis_log_sink,
    serialize=True,
)


# Helpers


def _get_image_summarization_chain(
    ext: str,
    image: str,
) -> dict:
    image_type = ext.replace(".", "")
    messages = ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_type};base64,{image}"},
                    },
                ]
            )
        ]
    )
    chain = messages | _vision_model | StrOutputParser()
    return chain


def _get_pdf_paths(
    pdf_filename: str,
) -> tuple[str, str, str]:
    output_dir = os.path.join(DATA_DIR, pdf_filename)
    index_output_dir = os.path.join(output_dir, "index")
    storage_output_dir = os.path.join(output_dir, "storage")
    return output_dir, index_output_dir, storage_output_dir


def _get_or_create_retriever(
    pdf_filename: str,
    index_output_dir: str,
    storage_output_dir: str,
) -> MultiVectorRetriever:
    vectorstore = Chroma(
        collection_name=pdf_filename,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=index_output_dir,
    )

    store = LocalFileStore(storage_output_dir)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=ID_KEY,
    )

    return retriever


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _add_documents_to_retriever(vectorstore, docstore, elements, summaries) -> None:
    if len(elements) == 0:
        return
    if len(elements) != len(summaries):
        raise ValueError("Elements and summaries must be the same length.")

    ids = [str(uuid.uuid4()) for _ in elements]
    docs = [
        Document(page_content=s, metadata={ID_KEY: ids[i]})
        for i, s in enumerate(summaries)
    ]
    vectorstore.add_documents(docs)
    docstore.mset(list(zip(ids, [Document(page_content=s) for s in elements])))


def _ingest_pdf(url: str) -> MultiVectorRetriever:
    logger.debug(f"Processing {url}")
    pdf_filename = url.split("/")[-1]
    logger.debug(f"PDF filename: {pdf_filename}")

    output_dir, index_output_dir, storage_output_dir = _get_pdf_paths(pdf_filename)
    logger.debug(f"Output dir: {output_dir}")
    logger.debug(
        f"Index output dir: {index_output_dir} {os.path.exists(index_output_dir)}"
    )
    logger.debug(
        f"Storage output dir: {storage_output_dir} {os.path.exists(storage_output_dir)}"
    )
    # Have we already processed this document?
    if os.path.exists(index_output_dir) and os.path.exists(storage_output_dir):
        logger.debug("Already processed.")
        return _get_or_create_retriever(
            pdf_filename, index_output_dir, storage_output_dir
        )
    logger.debug("Not yet processed.")

    # Pick up wherever we left off (in case of previous failure)
    pdf_input_path = os.path.join(output_dir, pdf_filename)

    # Does the pdf already exist?
    if not os.path.exists(pdf_input_path):
        logger.debug("Downloading PDF.")
        r = requests.get(url, allow_redirects=True)
        if r.status_code != 200:
            raise ValueError(f"Could not download PDF: {r.status_code}")
        os.makedirs(output_dir, exist_ok=True)
        open(pdf_input_path, "wb").write(r.content)
    else:
        logger.debug("PDF already downloaded.")

    text_elements_output_path = os.path.join(output_dir, "text.json")
    table_elements_output_path = os.path.join(output_dir, "table.json")
    image_elements_output_path = os.path.join(output_dir, "image.json")

    if not os.path.exists(text_elements_output_path):
        logger.debug("Partitioning PDF.")
        raw_pdf_elements = partition_pdf(
            filename=pdf_input_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=output_dir,
            extract_image_block_output_dir=output_dir,
        )

        text_elements = []
        table_elements = []
        image_elements = []

        for element in raw_pdf_elements:
            element_type = str(type(element))
            if "CompositeElement" in element_type:
                text_elements.append(element)
            elif "Table" in element_type:
                table_elements.append(element)

        table_elements = [i.text for i in table_elements]
        text_elements = [i.text for i in text_elements]

        for file in os.listdir(output_dir):
            if file.endswith((".png", ".jpg", ".jpeg")):
                ext = os.path.splitext(file)[1]
                image_path = os.path.join(output_dir, file)
                encoded_image = _encode_image(image_path)
                image_elements.append({"ext": ext, "image": encoded_image})

        with open(text_elements_output_path, "w") as f:
            json.dump(text_elements, f)
        with open(table_elements_output_path, "w") as f:
            json.dump(table_elements, f)
        with open(image_elements_output_path, "w") as f:
            json.dump(image_elements, f)
    else:
        logger.debug("Loading elements from disk.")
        with open(text_elements_output_path, "r") as f:
            text_elements = json.load(f)
        with open(table_elements_output_path, "r") as f:
            table_elements = json.load(f)
        with open(image_elements_output_path, "r") as f:
            image_elements = json.load(f)

    logger.debug(f"Text elements: {len(text_elements)}")
    logger.debug(f"Table elements: {len(table_elements)}")
    logger.debug(f"Image elements: {len(image_elements)}")

    text_summaries_output_path = os.path.join(output_dir, "text_summaries.json")
    table_summaries_output_path = os.path.join(output_dir, "table_summaries.json")
    image_summaries_output_path = os.path.join(output_dir, "image_summaries.json")

    retriever = _get_or_create_retriever(
        pdf_filename, index_output_dir, storage_output_dir
    )

    if len(text_elements) > 0:
        if not os.path.exists(text_summaries_output_path):
            logger.debug("Summarizing text elements.")
            text_summaries = _summarize_text_chain.batch(
                text_elements, {"max_concurrency": 5}
            )
            with open(text_summaries_output_path, "w") as f:
                json.dump(text_summaries, f)
        else:
            logger.debug("Loading summaries from disk.")
            with open(text_summaries_output_path, "r") as f:
                text_summaries = json.load(f)

        logger.debug("Adding text elements to retriever.")
        _add_documents_to_retriever(
            retriever.vectorstore, retriever.docstore, text_elements, text_summaries
        )
    else:
        logger.debug("No text elements.")

    if len(table_elements) > 0:
        if not os.path.exists(table_summaries_output_path):
            logger.debug("Summarizing table elements.")
            table_summaries = _summarize_table_chain.batch(
                table_elements, {"max_concurrency": 5}
            )
            with open(table_summaries_output_path, "w") as f:
                json.dump(table_summaries, f)
        else:
            logger.debug("Loading summaries from disk.")
            with open(table_summaries_output_path, "r") as f:
                table_summaries = json.load(f)

        logger.debug("Adding table elements to retriever.")
        _add_documents_to_retriever(
            retriever.vectorstore, retriever.docstore, table_elements, table_summaries
        )
    else:
        logger.debug("No table elements.")

    if len(image_elements) > 0:
        if not os.path.exists(image_summaries_output_path):
            logger.debug("Summarizing image elements.")
            image_summaries = []
            for image_element in image_elements:
                chain = _get_image_summarization_chain(**image_element)
                summary = chain.invoke(input={})
                image_summaries.append(summary)
            with open(image_summaries_output_path, "w") as f:
                json.dump(image_summaries, f)
        else:
            logger.debug("Loading image summaries from disk.")
            with open(image_summaries_output_path, "r") as f:
                image_summaries = json.load(f)

        logger.debug("Adding image elements to retriever.")
        _add_documents_to_retriever(
            retriever.vectorstore, retriever.docstore, image_summaries, image_summaries
        )
    else:
        logger.debug("No image elements.")

    return retriever


def question_pdf(
    url: str,
    question: str,
) -> str:
    logger.debug(f"Questioning {url}: {question}")
    retriever = _ingest_pdf(url)
    query_prompt_text = """Answer the question below based only on the following context,
    which can include text, tables, and images:
    {context}
    Question: {question}
    Answer:
    """
    query_prompt = ChatPromptTemplate.from_template(query_prompt_text)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | query_prompt
        | _query_model
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    logger.debug(f"Answer: {answer}")
    return answer


if __name__ == "__main__":
    path = "/Users/dothomps/src/gai/li/rfp/rag/data/2401.01325v1.pdf/figure-3-1.jpg"
    image = _encode_image(path)
    ext = os.path.splitext(path)[1]
    chain = _get_image_summarization_chain(ext, image)
    print(chain.invoke(input={}))
