# %%
##### Dependencies

#pip install langchain langchain-community langchain_openai
#pip install pandas
#pip install pinecone
#pip install sentence-transformers
#pip install -U "langchain" "langchain-core" "langchain-openai"
#pip install langchain-core
#pip install langchain-classic
###

# %%
# Importing libraries 

from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from RAG_system import ask 
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# %% [markdown]
# ## Defining RAG Tools 

# %%
@tool
def ask_rag_tool(question: str, previous_answer: str = "") -> str:
    """
    Use the RAG system (Pinecone + course transcripts) as the primary source,
    and let the model extend the answer when the context is not enough.
    If previous_answer is provided, treat this as a follow-up and include that
    answer as context when forming the query.
    """
    if previous_answer:
        full_question = f"{previous_answer}\n\nUser follow up: {question}"
    else:
        full_question = question
    return ask(full_question, top_k=15)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9,
    max_tokens=2000,
)

tools = [ask_rag_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Security+ SY0 701 tutor.\n"
            "You must always answer by calling the tool ask_rag_tool exactly once.\n"
            "When you receive the output from ask_rag_tool, your final answer to the user "
            "must be exactly that output, verbatim, without shortening, summarizing, or "
            "dropping any part of it. Do not rewrite or compress the tool output. "
            "Just return it as the answer.\n"
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

memory = ConversationBufferMemory(return_messages=True)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
)




# %%
## Defining the funtion for the RAG tool retrieval 

# %%
def chat(query: str) -> str:
    """
    Call the agent_executor and always return the full RAG answer string.
    Instead of using result["output"], take the tool output from intermediate_steps.
    """
    result = agent_executor.invoke({"input": query})

    # LangChain stores (AgentAction, tool_output) pairs here
    steps = result.get("intermediate_steps", [])

    if steps:
        # Take the last tool call output
        last_action, last_output = steps[-1]
        answer = last_output
    else:
        # Fallback to the agent final output if no tool was called
        answer = result.get("output", "")

    # Normalize to string
    if isinstance(answer, str):
        text = answer
    elif hasattr(answer, "content"):
        text = str(answer.content)
    else:
        text = str(answer)

    text = text.strip()

    print("Raw Answer from chat():", repr(text))
    return text

# %%

if __name__ == "__main__":
    print("RAG chatbot with tools and memory. Type 'exit' to quit.\n")
    while True:
        q = input("Ask question or write quit: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        answer = chat(q)
        print(f"Bot: {answer}\n")



