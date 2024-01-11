from langchain.chains import create_tagging_chain
from langchain.chat_models import ChatOpenAI

def fact_check(input: str) -> str: #NOTE make llm as input parameter
    schema = {
        "properties": {
            "fact_label": {
                "type": "string",
                "enum": ['True', 'Mostly True', 'Half True', "Mostly False", "False", "Pants on Fire"],
                "description": """Please classify the given statement into one of the following categories:
        True - The statement is accurate and there is nothing significant missing.
        Mostly True - The statement is accurate but needs clarification or additional information.
        Half True - The statement is partially accurate but leaves out important details or takes things out of context.
        Mostly False - The statement contains an element of truth but ignores critical facts that would give a different impression.
        False - The statement is not accurate.
        Pants on Fire - The statement is not accurate and makes a ridiculous claim including a conspiracy.
        The statement will have the following format:
        statement|author|date
        e.g.:'Last year alone, natural disasters in America caused $178 billion in damages.|Barack Obama|2013-02-13'"""
            },
        },
        "required": ["fact_label"],
    }
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    chain = create_tagging_chain(schema, llm)
    return chain.run(input)

fact_check("Last year alone, natural disasters in America caused $178 billion in damages.|Barack Obama|2013-02-13")