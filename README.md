# PDF RAG

## Installation

In your virtual environment of choice, install the required packages:

```bash
poetry install
```

## Setup LangSmith (Optional)

LangSmith will help us trace, monitor and debug LangChain applications.
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/).

Add your keys and enable LangSmith in your project using the .env file.

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
