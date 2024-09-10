{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Running Test\n",
        "This notebook does the following:\n",
        "- Requests that you set API keys for\n",
        "    - Pinecone\n",
        "    - OpenAI\n",
        "    - GroundX\n",
        "- Passes the documents to Pinecone, LlamaIndex, and GroundX for parsing and storage\n",
        "- Constructs RAG from LangChain/PineCone, LlamaIndex, and GroundX\n",
        "- Runs all three RAG approaches against the test questions"
      ],
      "metadata": {
        "id": "K9aQK-OBsGMY"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Setting Keys\n",
        "There are three keys required to run this notebook:\n",
        " - A GroundX API key, You'll need a GroundX account. Sign up [here](https://www.groundx.ai/account/signup)\n",
        " - A PineCone API key. You'll need a GroundX account. Sign up [here](https://www.pinecone.io/).\n",
        " - An OpenAI API key. You'll need a GroundX account. Sign up [here](https://openai.com/).\n",
        "\n",
        "This notebook will use those keys to do the following:\n",
        " - The GroundX API key will be used to create a bucket and upload the test documents to that bucket. This GroundX bucket will be searched to satisfy the retrevial component of RAG.\n",
        " - The PineCone API key will be used to create an index and upload documents to that index. LangChain will use that index to satisfy the retreival component of RAG\n",
        " - The PineCone API requires vector embeddings, which will be done using the OpenAI API key.\n",
        "\n",
        "This notebook does not save your keys programatically, though it is recommended to use Colab's built in secerets management system rather than hard code all credentials.\n"
      ],
      "metadata": {
        "id": "6UJPa5k8Brar"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "z-Fim_JCBrar"
      },
      "outputs": [],
      "source": [
        "#set this to false if you're hard-coding your secerets\n",
        "using_secerets_management = True\n",
        "\n",
        "if using_secerets_management:\n",
        "    #uses secerets from colabs built in secerets management system\n",
        "    from google.colab import userdata\n",
        "    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')\n",
        "    PINECONE_API_KEY = userdata.get('PINECONE_API_KEY')\n",
        "    GROUNDX_API_KEY = userdata.get('GROUNDX_API_KEY')\n",
        "else:\n",
        "    #enter your hard coded secerets here\n",
        "    OPENAI_API_KEY = \"xxxxxxx\"\n",
        "    PINECONE_API_KEY = \"xxxxxxx\"\n",
        "    GROUNDX_API_KEY = \"xxxxxxx\""
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Downloading Dependencies\n",
        "If you experience any issues with dependencies, you may consider upgrading the versions. Current versions can be found on [pypi.org](https://pypi.org\n",
        "). Optionally, specific version specifications can be removed to install the most up-to-date version."
      ],
      "metadata": {
        "id": "2DG79issIM4q"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#general dependencies\n",
        "!pip install langChain==0.1.7\n",
        "!pip install openai==1.12.0\n",
        "!pip install pinecone-client==3.1.0\n",
        "!pip install llama-index==0.10.5\n",
        "!pip install groundx-python-sdk==1.3.14\n",
        "\n",
        "#for tokenizing text\n",
        "!pip install tiktoken==0.6.0\n",
        "\n",
        "#for langchain to load docs (this one might take a few minutes, and\n",
        "#you may need to re-start your session for this install to take effect)\n",
        "!pip install unstructured[pdf]"
      ],
      "metadata": {
        "id": "KrnrZdUVIX4i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Configuring\n",
        "Setting secerets in their proper spots"
      ],
      "metadata": {
        "id": "33smVvyMfxio"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "os.environ[\"OPENAI_API_KEY\"] = OPENAI_API_KEY\n",
        "os.environ[\"PINECONE_API_KEY\"] = PINECONE_API_KEY\n",
        "\n",
        "from groundx import Groundx\n",
        "groundx = Groundx(\n",
        "    api_key=GROUNDX_API_KEY,\n",
        ")"
      ],
      "metadata": {
        "id": "81wON9LugvF_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Downloading Files Locally\n",
        "This notebook downloads test files from a publicly exposed S3 bucket. Downloading those files locally so that they can be passed to PineCone and LlamaIndex."
      ],
      "metadata": {
        "id": "BgOPmZDiCI4u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#unpacking to directory\n",
        "import zipfile\n",
        "import os\n",
        "import requests\n",
        "\n",
        "if not os.path.exists('TestFiles'):\n",
        "    os.mkdir('TestFiles')\n",
        "\n",
        "    r = requests.get('https://cdn.eyelevel.ai/demo/rag/test-files.zip', allow_redirects=True)\n",
        "    open('test-files.zip', 'wb').write(r.content)\n",
        "\n",
        "    with zipfile.ZipFile('test-files.zip', 'r') as zip_ref:\n",
        "        zip_ref.extractall('TestFiles')\n",
        "else:\n",
        "    print('file directory already found')"
      ],
      "metadata": {
        "id": "-KgvlRcUSBdk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Creating LlamaIndex\n",
        "This creates a LlamaIndex using `llama_index.core`'s `VectorStoreIndex` and `SimpleDirectoryReader` and saves it to a subdirectory called \"LlamaIndex\""
      ],
      "metadata": {
        "id": "BB0GOqpcH184"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from llama_index.core import VectorStoreIndex, SimpleDirectoryReader\n",
        "from llama_index.llms.openai import OpenAI\n",
        "import time\n",
        "from llama_index.core import StorageContext, load_index_from_storage\n",
        "\n",
        "#making a directory to store the index, then making the index\n",
        "if not os.path.exists('LlamaIndex'):\n",
        "    os.makedirs('LlamaIndex')\n",
        "\n",
        "    #making LlamaIndex and saving it locally.\n",
        "    start = time.process_time()\n",
        "    documents = SimpleDirectoryReader('TestFiles/all/').load_data()\n",
        "    li_index = VectorStoreIndex.from_documents(documents)\n",
        "    end = time.process_time()\n",
        "    print(f'elapsed: {end-start}')\n",
        "    li_index.storage_context.persist(persist_dir=\"LlamaIndex\")\n",
        "else:\n",
        "    print('LlamaIndex already made. Loading...')\n",
        "    storage_context = StorageContext.from_defaults(persist_dir='LlamaIndex')\n",
        "    li_index = load_index_from_storage(storage_context)"
      ],
      "metadata": {
        "id": "1ujEnEcrGAzH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Creating Pinecone vector store\n",
        "this creates a new vector store in Pinecone and uploads all documents to it via LangChain. Or, it uses a pre-made pinecone index by the correct name."
      ],
      "metadata": {
        "id": "8KTLsnuLBMS_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from pinecone import Pinecone, ServerlessSpec\n",
        "\n",
        "pc = Pinecone(api_key=PINECONE_API_KEY)\n",
        "\n",
        "#creating a pinecone index compatible with OpenAIs text-embedding-ada-002\n",
        "pinecone_setup = False\n",
        "pinecone_index_name = \"rag-comparison-test\"\n",
        "try:\n",
        "    pc.create_index(\n",
        "        name=pinecone_index_name,\n",
        "        dimension=1536,\n",
        "        metric=\"cosine\",\n",
        "        spec=ServerlessSpec(\n",
        "            cloud='aws',\n",
        "            region='us-west-2'\n",
        "        )\n",
        "    )\n",
        "except Exception as e:\n",
        "    if e.reason == 'Conflict':\n",
        "        print('index already exists')\n",
        "        pinecone_setup = True\n",
        "    else:\n",
        "        raise e\n",
        "\n",
        "from langchain_community.document_loaders import DirectoryLoader\n",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
        "from langchain.embeddings.openai import OpenAIEmbeddings\n",
        "from langchain.vectorstores import Pinecone\n",
        "\n",
        "if not pinecone_setup:\n",
        "    print('this might take a bit. Go grab lunch...')\n",
        "\n",
        "    #loading files from directory\n",
        "    docs = DirectoryLoader('TestFiles/all/').load()\n",
        "\n",
        "    #parsing and splitting documents\n",
        "    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\n",
        "    texts = text_splitter.split_documents(docs)\n",
        "\n",
        "    #setting up the final docsearch object\n",
        "    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)\n",
        "    pc_docsearch = Pinecone.from_documents(texts, embeddings, index_name=pinecone_index_name)\n",
        "else:\n",
        "\n",
        "    print('assuming documents are already uploaded. Getting index...')\n",
        "    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)\n",
        "    pc_docsearch = Pinecone.from_existing_index(pinecone_index_name, embeddings)\n",
        "\n",
        "print('done')"
      ],
      "metadata": {
        "id": "h1c2IK2oBMBe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Creating GroundX Bucket\n",
        "while GroundX has functionality for uploading local documents, it's designed for smaller sets of documents. For this, we'll use GroundX's `upload_remote` function to upload remotely hosted documents."
      ],
      "metadata": {
        "id": "CUyHuj7lGbDG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "bucket_name = \"rag-comparison-test\"\n",
        "\n",
        "# Checking if the bucket exists\n",
        "response = groundx.buckets.list(n=100)\n",
        "bucket_exists = False\n",
        "bucket_id = None\n",
        "for bucket in response.body['buckets']:\n",
        "    if bucket['name'] == bucket_name:\n",
        "        print('bucket already created')\n",
        "        print(bucket)\n",
        "        bucket_id = bucket['bucketId']\n",
        "        bucket_exists = True\n",
        "        break\n",
        "\n",
        "#Creating a bucket\n",
        "if not bucket_exists:\n",
        "    print('Creating a bucket')\n",
        "\n",
        "    response = groundx.buckets.create(\n",
        "        name=bucket_name\n",
        "    )\n",
        "    bucket_id = response.body['bucket']['bucketId']\n",
        "\n",
        "bucket_exists = False\n",
        "\n",
        "#uploading content to bucket\n",
        "if not bucket_exists:\n",
        "    print('Uploading content to the bucket')\n",
        "    print('Note, GroundX frontloads substantial pre-processing on ingest.')\n",
        "    print('This process may take some time...')\n",
        "\n",
        "    files = os.listdir('TestFiles/all/')\n",
        "    processes = []\n",
        "    print('scheduling files for upload...')\n",
        "    for i, file in enumerate(files):\n",
        "        if i == 0:\n",
        "            continue\n",
        "        print(file)\n",
        "        response = groundx.documents.upload_remote(\n",
        "            documents=[{\n",
        "                    \"bucketId\": bucket_id,\n",
        "                    \"fileName\": file,\n",
        "                    \"fileType\": os.path.splitext(file)[1][1:],\n",
        "                    \"sourceUrl\": 'https://cdn.eyelevel.ai/demo/rag/'+file\n",
        "                }])\n",
        "        processes.append(response.body['ingest']['processId'])"
      ],
      "metadata": {
        "id": "i__6vHGcGa3y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import copy\n",
        "\n",
        "if not bucket_exists:\n",
        "    print('waiting for upload to complete. This includes all GroundX')\n",
        "    print('pre-processing, so it might take a while ()...')\n",
        "\n",
        "    checking = copy.deepcopy(processes)\n",
        "\n",
        "    # Checking if all uploads are completed\n",
        "    while True:\n",
        "        print('checking uploads...')\n",
        "\n",
        "        for process_id in checking:\n",
        "            response = groundx.documents.get_processing_status_by_id(process_id)\n",
        "            # print(f\"{process_id}: {response.body['ingest']['status']}\")\n",
        "            if response.body['ingest']['status'] != 'complete' or response.body['ingest']['status'] != 'error':\n",
        "                print('still in progress...')\n",
        "                break\n",
        "\n",
        "        else:\n",
        "            break\n",
        "\n",
        "        time.sleep(30)"
      ],
      "metadata": {
        "id": "IyKi8lWVk3cH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Building RAG\n",
        "now that all the approaches are defined, we can build functions which abstract them into RAG."
      ],
      "metadata": {
        "id": "IaXStne6waUr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\"\"\"Defining RAG for LlamaIndex\n",
        "\"\"\"\n",
        "\n",
        "from llama_index.llms.openai import OpenAI as LI_OpenAI\n",
        "\n",
        "def define_query_engine(model):\n",
        "    llm = LI_OpenAI(temperature=0.0, model=model)\n",
        "    return li_index.as_query_engine(llm=llm)\n",
        "\n",
        "li_rag = define_query_engine('gpt-4-1106-preview')"
      ],
      "metadata": {
        "id": "X_JsQ9dewamv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\"\"\"Defining RAG for LangChain/PineCone\n",
        "\"\"\"\n",
        "\n",
        "from langchain.chains.question_answering import load_qa_chain\n",
        "from langchain.chains import RetrievalQA\n",
        "from langchain.chat_models import ChatOpenAI\n",
        "\n",
        "retriever = pc_docsearch.as_retriever(include_metadata=True, metadata_key = 'source')\n",
        "\n",
        "def make_chain(model_name):\n",
        "    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)\n",
        "    qa_chain = RetrievalQA.from_chain_type(llm=llm,\n",
        "                                    chain_type=\"stuff\",\n",
        "                                    retriever=retriever,\n",
        "                                    return_source_documents=True)\n",
        "    return qa_chain\n",
        "\n",
        "lcpc_rag = make_chain('gpt-4-1106-preview')"
      ],
      "metadata": {
        "id": "exrtbtQoyTnI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\"\"\"Defining RAG for GroundX\n",
        "note: we're planning on rolling out an abstraction which makes this a\n",
        "bit less verbose.\n",
        "\"\"\"\n",
        "\n",
        "system_prompt = \"\"\"\n",
        "Your task is to use the document excerpts to provide answers to employees. Answer questions precisely and succinctly but provide full lists of steps and stipulations if the response requires it.\n",
        "\n",
        "Format your answers into list format when appropriate, but make sure they are complete. Long lists are ok if the information is specifically addressing the question.\n",
        "Format your answers in easy to read tables when asked to do so.\n",
        "\n",
        "Answer questions with as much detail as possible. Think logically and take it step by step. If the content explains where to find more information, please include that in your answer.\n",
        "\n",
        "DO NOT tell the USER to go to the a website.\n",
        "DO NOT recommend contacting customer service since you are providing information to employees.\n",
        "DO NOT reference files names unless specifically asked to do so.\n",
        "DO NOT provide information other than what you have in the CONTENT.\n",
        "\n",
        "If you cannot find an answer to the question from the document excerpts you are provided, ask the USER to limit their questions to the documents you have ingested for this proof of concept.\n",
        "\n",
        "If your answer is short, please ask the USER whether they need more detailed information.\n",
        "\n",
        "Answer questions in whatever language the USER is speaking in.\n",
        "\n",
        "Remember, USERS are employees and ARE NOT customers. DO NOT refer to the USER as a customer.\n",
        "\n",
        "If you are asked to create questions from specific documents, create the questions using the document excerpts provided to you. They should include the filenames they are from.\n",
        "\"\"\"\n",
        "\n",
        "def gx_rag(query, model='gpt-4-1106-preview', bucket=bucket_id, system_prompt = system_prompt):\n",
        "    from openai import OpenAI\n",
        "    #getting information from GroundX\n",
        "\n",
        "    for _ in range(3):\n",
        "        try:\n",
        "            context = groundx.search.content(\n",
        "                id=bucket,\n",
        "                query=query\n",
        "            ).body[\"search\"]\n",
        "\n",
        "            print(context)\n",
        "            context = context[\"text\"]\n",
        "            break\n",
        "        except:\n",
        "            print('error on GroundX Retreival, re-trying')\n",
        "            time.sleep(5)\n",
        "    else:\n",
        "        print('GroundX failed, setting the context as failed to retreive.')\n",
        "        context = 'failed to retreive information from GroundX. API error on groundx.search.content'\n",
        "\n",
        "    #ensuring the context is of reasonable size\n",
        "    if model == 'gpt-3.5-turbo':\n",
        "        if len(context) > 4000 * 3:\n",
        "            context = context[:4000*3]\n",
        "    elif model == 'gpt-4-0613':\n",
        "        if len(context) > 6000 * 3:\n",
        "            context = context[:6000*3]\n",
        "    else:\n",
        "        if len(context) > 6000 * 3:\n",
        "            context = context[:6000*3]\n",
        "\n",
        "    context_query_template = \"\"\"\n",
        "    Content:\n",
        "    {}\n",
        "\n",
        "    Answer the following question using the content above:\n",
        "    {}\n",
        "    \"\"\"\n",
        "\n",
        "    in_context = [{'role': 'user', 'content': 'Content:\\nDerbent, a city in Dagestan, Russia, claims to be the oldest city in Russia. Archaeological excavations have confirmed that Derbent has been continuously inhabited for nearly 2,000 years. Historical documentation dates back to the 8th century BC, making it one of the oldest continuously inhabited cities in the world.\\n\\nAnswer the following question using the content above:\\nHow old is the oldest city in Russia?'},\n",
        "                  {'role': 'assistant', 'content':'The oldest city in Russia, Derbent, is nearly 2,000 years old.'},\n",
        "                  {'role': 'user', 'content': 'Content:\\nJoan is 42 and John is 55\\n\\nAnswer the following question using the content above:\\nWhat is the age difference between Joan and John?'},\n",
        "                  {'role': 'assistant', 'content':'The age difference between Joan and John is 13 years.'},\n",
        "                  {'role': 'user', 'content': 'Content:\\nThe High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\\n\\nAnswer the following question using the content above:\\nHow high do the High Plains go?'},\n",
        "                  {'role': 'assistant', 'content':'The High Plains reach an elevation of up to 7,000ft '}]\n",
        "\n",
        "    #Augmenting users query with the system prompt and context from GroundX\n",
        "    messages = [{\n",
        "        \"role\": \"system\",\n",
        "        \"content\": system_prompt}] + in_context + [{\n",
        "        \"role\": \"user\",\n",
        "        \"content\": context_query_template.format(context, query)\n",
        "         }]\n",
        "\n",
        "    #Generating\n",
        "    client = OpenAI()\n",
        "    return client.chat.completions.create(model=model,messages=messages).choices[0].message.content"
      ],
      "metadata": {
        "id": "YUnqEaewymFx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Testing all RAG questions"
      ],
      "metadata": {
        "id": "zJml7tA8z9Yl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def run_rag(query):\n",
        "    \"\"\"Runs all RAG approaches against a query\n",
        "    \"\"\"\n",
        "\n",
        "    return {'lcpc_res_4': lcpc_rag(query)['result'],\n",
        "            'gx_res_4': gx_rag(query),\n",
        "            'li_res_4':li_rag.query(query).response}\n",
        "\n",
        "res = run_rag('What is the branch rate in Zimbabwe?')\n",
        "print(f'---\\nresults:')\n",
        "print(res)"
      ],
      "metadata": {
        "id": "knG0Yb42z9O4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Downloading Questions"
      ],
      "metadata": {
        "id": "wAxp4abSskg_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "df_questions = pd.read_csv('https://cdn.eyelevel.ai/demo/rag/tests.tsv', sep='\\t')\n",
        "df_questions"
      ],
      "metadata": {
        "id": "zhxg3uW7snqL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Running test"
      ],
      "metadata": {
        "id": "7iXUVoNx2esm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import time\n",
        "\n",
        "res = []\n",
        "\n",
        "for i, question_set in df_questions.iterrows():\n",
        "    print('==============================================')\n",
        "    print('Question {}/{}'.format(i, len(df_questions.index)))\n",
        "    print('==============================================')\n",
        "    print('__________Question Details__________')\n",
        "    print('Question: {}'.format(question_set['query']))\n",
        "    print('Answer: {}'.format(question_set['expected_response']))\n",
        "    print('-------answers------')\n",
        "    for _ in range(3):\n",
        "        try:\n",
        "            v = run_rag(question_set['query'])\n",
        "            v['question'] = question_set['query']\n",
        "            v['expected answer'] = question_set['expected_response']\n",
        "            v['context_file'] = question_set['context_file']\n",
        "            v['problem_type'] = question_set['problem_type']\n",
        "            res.append(v)\n",
        "            break\n",
        "        except Exception as e:\n",
        "            print('error -------------------')\n",
        "            print(e)\n",
        "            print('error -------------------')\n",
        "            print('issue, retrying all')\n",
        "    else:\n",
        "        raise ValueError('The api messed up')\n",
        "\n",
        "    print('\\n')\n",
        "\n",
        "    #sleeping 20 seconds, to be conservative, for openAI rate limit\n",
        "    time.sleep(5)"
      ],
      "metadata": {
        "id": "VH5jgrShswgO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pd.DataFrame(res).to_csv('test_results.csv')"
      ],
      "metadata": {
        "id": "Wtl0cyzM2rUb"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}