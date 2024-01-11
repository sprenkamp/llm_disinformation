from langchain.schema import HumanMessage, SystemMessage
from langchain.base_language import BaseLanguageModel

def fact_check(statement: str, author: str, date:str, llm: BaseLanguageModel, search: str = None) -> str:
        prompt = [ #TODO better support 0, 1 seems to be working well
            SystemMessage(
                content=f"""Please classify the given statement into one of the following labels, use your own extensive knowledge and the information provided in the statement:
        true:The statement is accurate and there is nothing significant missing.
        mostly-true:The statement is accurate but needs clarification or additional information.
        half-true:The statement is partially accurate but leaves out important details or takes things out of context.
        barely-true:The statement contains an element of truth but ignores critical facts that would give a different impression.
        false:The statement is not accurate.
        pants-fire:The statement is not accurate and makes a ridiculous claim including a conspiracy.
        The statement will have the following format:
        statement|author
        e.g.:'Last year alone, natural disasters in America caused $178 billion in damages.|Barack Obama|2013-02-13'
        All of these statements have been made in real life the date even if it lies after your knowledge cut has already been passed.
        Further I want you to reason about your decision and write a short explanation. The output should be a string of the form.
        <Label> -> <reason why you chose this label>
        """
            ),
            HumanMessage(
                content=f"{statement}|{author}|{date}"
            ),
        ]
        #llm = ChatOpenAI(temperature=0, model="gpt-4", timeout=60)
        # print(input_str)
        output = llm.invoke(prompt)
        return output.content #== "1"