import os
from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.astradb import AstraDB

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl

from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_VECTOR_ENDPOINT = os.environ["ASTRA_VECTOR_ENDPOINT"]
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]
ASTRA_DB_COLLECTION = os.environ["ASTRA_DB_COLLECTION"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"


#
# Define LLM Prompt
#
system_template = """You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in the user's language:"""

user_template = """Question:
{question}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]
prompt = ChatPromptTemplate.from_messages( messages )


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Connecting to database ...", disable_feedback=True)
    await msg.send()

    #
    # define Embedding model
    #
    embeddings = OpenAIEmbeddings()

    #
    # define AstraDB vector store
    #
    vectorstore = AstraDB(
        embedding=embeddings,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
    )

    retriever = vectorstore.as_retriever()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    #
    # Create a chain that uses the Astra vector store
    #
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )



    # Let the user know that the system is ready
    msg.content = "You can now ask questions!"
    await msg.send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    # activate the chain
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    # add the source documents to the answer
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            # remove all words after the | character
            title = source_doc.metadata["title"].split(" | ")[0]

            # create a Text element for the source
            text_elements.append(
                cl.Text(content=source_doc.metadata["source"], name=title)
            )

        source_names = [text_el.name for text_el in text_elements]

        # remove duplicates in source_names
        source_names = list(dict.fromkeys(source_names))

        if source_names:
            answer += "\n\nSources:"
            for source in source_names:
                answer += f"\n{source}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
