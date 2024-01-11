class DisInformationClassification:
    def __init__(self,input_text,date,source):
        self.input_text = input_text
        self.date = date
        self.entity = entity
    def __str__(self):
        return("The stance of entity {} toward the statement {} given text {} is {}".format(self.entity,self.statement,self.input_text,self.stance))
    def detect_stance(self, llm_setup):
        out = llm_setup.apply(self)
        self.llm_setup = llm_setup
        self.reasoning = out["analysis"]["analysis"]
        self.stance = out["summarization"]["stance"]
        print("Detected stance: " + str(self.stance))