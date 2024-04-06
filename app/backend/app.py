import io
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import AsyncGenerator

# scrapping
import requests
from bs4 import BeautifulSoup
import pandas as pd
# Data preprocess
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
#Inference
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
    
import aiohttp
import openai
from azure.core.exceptions import ResourceNotFoundError
from azure.identity.aio import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.search.documents.aio import SearchClient
from azure.storage.blob.aio import BlobServiceClient
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from quart import (
    Blueprint,
    Quart,
    abort,
    current_app,
    jsonify,
    make_response,
    request,
    send_file,
    send_from_directory,
)
from quart_cors import cors

from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.retrievethenread import RetrieveThenReadApproach
from core.authentication import AuthenticationHelper

CONFIG_OPENAI_TOKEN = "openai_token"
CONFIG_CREDENTIAL = "azure_credential"
CONFIG_ASK_APPROACH = "ask_approach"
CONFIG_CHAT_APPROACH = "chat_approach"
CONFIG_BLOB_CONTAINER_CLIENT = "blob_container_client"
CONFIG_AUTH_CLIENT = "auth_client"
CONFIG_SEARCH_CLIENT = "search_client"
ERROR_MESSAGE = """The app encountered an error processing your request.
If you are an administrator of the app, view the full error in the logs. See aka.ms/appservice-logs for more information.
Error type: {error_type}
"""
ERROR_MESSAGE_FILTER = """Your message contains content that was flagged by the OpenAI content filter."""

bp = Blueprint("routes", __name__, static_folder="static")
# Fix Windows registry issue with mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

dataframe = None
vectordb = None

@bp.route("/")
async def index():
    return await bp.send_static_file("index.html")


# Empty page is recommended for login redirect to work.
# See https://github.com/AzureAD/microsoft-authentication-library-for-js/blob/dev/lib/msal-browser/docs/initialization.md#redirecturi-considerations for more information
@bp.route("/redirect")
async def redirect():
    return ""


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory(Path(__file__).resolve().parent / "static" / "assets", path)


# Serve content files from blob storage from within the app to keep the example self-contained.
# *** NOTE *** this assumes that the content files are public, or at least that all users of the app
# can access all the files. This is also slow and memory hungry.
@bp.route("/content/<path>")
async def content_file(path: str):
    # Remove page number from path, filename-1.txt -> filename.txt
    if path.find("#page=") > 0:
        path_parts = path.rsplit("#page=", 1)
        path = path_parts[0]
    logging.info("Opening file %s at page %s", path)
    blob_container_client = current_app.config[CONFIG_BLOB_CONTAINER_CLIENT]
    try:
        blob = await blob_container_client.get_blob_client(path).download_blob()
    except ResourceNotFoundError:
        logging.exception("Path not found: %s", path)
        abort(404)
    if not blob.properties or not blob.properties.has_key("content_settings"):
        abort(404)
    mime_type = blob.properties["content_settings"]["content_type"]
    if mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    blob_file = io.BytesIO()
    await blob.readinto(blob_file)
    blob_file.seek(0)
    return await send_file(blob_file, mimetype=mime_type, as_attachment=False, attachment_filename=path)


def error_dict(error: Exception) -> dict:
    if isinstance(error, openai.error.InvalidRequestError) and error.code == "content_filter":
        return {"error": ERROR_MESSAGE_FILTER}
    return {"error": ERROR_MESSAGE.format(error_type=type(error))}


def error_response(error: Exception, route: str, status_code: int = 500):
    logging.exception("Exception in %s: %s", route, error)
    if isinstance(error, openai.error.InvalidRequestError) and error.code == "content_filter":
        status_code = 400
    return jsonify(error_dict(error)), status_code


@bp.route("/ask", methods=["POST"])
async def ask():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    context = request_json.get("context", {})
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    context["auth_claims"] = await auth_helper.get_auth_claims_if_enabled(request.headers)
    try:
        approach = current_app.config[CONFIG_ASK_APPROACH]
        # Workaround for: https://github.com/openai/openai-python/issues/371
        async with aiohttp.ClientSession() as s:
            openai.aiosession.set(s)
            r = await approach.run(
                request_json["messages"], context=context, session_state=request_json.get("session_state")
            )
        return jsonify(r)
    except Exception as error:
        return error_response(error, "/ask")


async def format_as_ndjson(r: AsyncGenerator[dict, None]) -> AsyncGenerator[str, None]:
    try:
        async for event in r:
            yield json.dumps(event, ensure_ascii=False) + "\n"
    except Exception as e:
        logging.exception("Exception while generating response stream: %s", e)
        yield json.dumps(error_dict(e))

# Send MSAL.js settings to the client UI
@bp.route("/auth_setup", methods=["GET"])
def auth_setup():
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    return jsonify(auth_helper.get_auth_setup_for_client())


# @bp.before_request
# async def ensure_openai_token():
#     if openai.api_type != "azure_ad":
#         return
#     openai_token = current_app.config[CONFIG_OPENAI_TOKEN]
#     if openai_token.expires_on < time.time() + 60:
#         openai_token = await current_app.config[CONFIG_CREDENTIAL].get_token(
#             "https://cognitiveservices.azure.com/.default"
#         )
#         current_app.config[CONFIG_OPENAI_TOKEN] = openai_token
#         openai.api_key = openai_token.token


def scraping(index_url):
    def fetch_page(url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the response was an error
            return response.text
        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def parse_links(html, base_url):
        soup = BeautifulSoup(html, 'html.parser')
        links = [base_url + a['href'] for a in soup.find_all('a', href=True)]
        print(links)
        return links

    def extract_information(html):
        return str(html)

    def scrape_subpages(links):
        data = []
        for link in links:
            print("Fetching:", link)
            html = fetch_page(link)
            if html:
                # Your parsing logic here, e.g., find specific information within the subpage
                # print(html[:200])
                print("Html get **************")
                info = extract_information(html)  # Implement this function based on your needs
                data.append({'URL': link, 'Information': info})
        return pd.DataFrame(data)

    # Start by fetching the index page
    # index_url = 'https://www.harvard.edu'
    index_html = fetch_page(index_url)

    # Assume the base URL is known (for appending to relative links)
    base_url = ''

    # Parse the index page to find links to subpages
    subpage_links = parse_links(index_html, base_url)

    # Scrape each subpage for information
    dataframe = scrape_subpages(subpage_links)
    return dataframe
    
def preprocess(data_path):
    df = pd.read_csv(data_path)
    docs_ls = []
    for i in range(len(df)):
        file = open("websites/helper.html", "w")
        file.write(df.iloc[i].loc['Information'])
        file.close()
        loader = UnstructuredHTMLLoader('websites/helper.html')    
        docs_ls.append(loader.load()[0]) 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 150,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    splits = text_splitter.split_documents(docs_ls)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = 'vectorstores/chroma/'
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return vectordb

def get_all_source_url(result):
    def get_source_url(source_documents):
        source_str = source_documents.metadata['source']
        pattern = r'/(?P<number>\d+)\.html'

        # Use re.search to find the pattern in the string
        match = re.search(pattern, source_str)

        # Check if a match is found
        if match:
            # Extract the number using group() method
            extracted_number = match.group('number')
            url = df.iloc[int(extracted_number)].loc['URL']
            return url
        else:
            print("No match found.")
            return None
        
    source_urls = set()
    df = pd.read_csv('websites/scraped_data.csv')
    for i in result["source_documents"]:
        source_url = get_source_url(i)
        source_urls.add(source_url)
    return source_urls


def openai_setup(secret_path: str):
    """
    Load OpenAI API key from the secrets file
    """
    with open(secret_path) as f:
        secrets = json.load(f)
    os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']
    openai.api_key = os.environ['OPENAI_API_KEY']

# OpenAI setup
openai_setup('./secrets/openai_secret.json')

# Load vector database
persist_directory = './vectorstores/chroma/'
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Load OpenAI model
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Load retrieval QA model
retriever=vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
)


@bp.before_app_serving
async def setup_clients():
    print("Init backend")

    # NOTE: chatbot initialization is done outside to keep it a global variable.

    # # Data scraping
    # dataframe = scraping('https://www.harvard.edu')
    # dataframe.to_csv('scraped_data.csv', index=False)

    # # Data preprocess
    # vectordb = preprocess('./websites/scraped_data.csv')

    
@bp.route("/chat", methods=["POST"])
async def chat():
    print("Received chat post request")

    # Get data from the request
    data = await request.get_json()
    query = data["messages"][0]["content"]
    result = qa_chain({"query": query})
    reference = get_all_source_url(result)

    # # Debug
    # print("request:", data)
    # print("response:", result)
    # print("reference:", reference)
    
    answer = {
        "answer": result["result"],
        "reference": list(reference)
    }
    return jsonify(answer)


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.asgi_app = OpenTelemetryMiddleware(app.asgi_app)  # type: ignore[method-assign]

    # Level should be one of https://docs.python.org/3/library/logging.html#logging-levels
    default_level = "INFO"  # In development, log more verbosely
    if os.getenv("WEBSITE_HOSTNAME"):  # In production, don't log as heavily
        default_level = "WARNING"
    logging.basicConfig(level=os.getenv("APP_LOG_LEVEL", default_level))

    if allowed_origin := os.getenv("ALLOWED_ORIGIN"):
        app.logger.info("CORS enabled for %s", allowed_origin)
        cors(app, allow_origin=allowed_origin, allow_methods=["GET", "POST"])
    cors(app, allow_origin="http://localhost:5173")

    return app
