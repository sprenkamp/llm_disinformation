
class StanceClassification:
    def __init__(self,input_text,statement,entity):
        self.input_text = input_text
        self.statement = statement
        self.entity = entity
        self.stance = None
        self.reasoning = None
    def __str__(self):
        return("The stance of entity {} toward the statement {} given text {} is {}".format(self.entity,self.statement,self.input_text,self.stance))
    def detect_stance(self, llm_setup):
        out = llm_setup.apply(self)
        self.llm_setup = llm_setup
        self.reasoning = out["analysis"]["analysis"]
        self.stance = out["summarization"]["stance"]
        print("Detected stance: " + str(self.stance))

class StanceDetectionMethod:
    def __init__(self,fun,lm):
        self.fun = fun
        self.lm = lm
    def apply(self, task: StanceClassification):
        out = self.fun(task, lm = self.lm)
        if not all([key in out.keys() for key in ["analysis","summarization"]]):
            return "error"
        else:
            return(out)