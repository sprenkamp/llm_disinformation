from stance_llm import *

from guidance import models, gen, select, user, system, assistant

from huggingface_hub import hf_hub_download

# setting model paths to downloaded local models

# modelpath_em_german_70b = hf_hub_download(repo_id='TheBloke/em_german_70b_v01-GGUF',filename="em_german_70b_v01.Q4_K_M.gguf",cache_dir = "../../../../../mnt")
# modelpath_leo_7b_mistral=hf_hub_download(repo_id='TheBloke/Leo-Mistral-Hessianai-7B-Chat-GGUF',filename="leo-mistral-hessianai-7b-chat.Q6_K.gguf")
# modelpath_leo_13b=hf_hub_download(repo_id='TheBloke/leo-hessianai-13B-chat-GGUF', filename="leo-hessianai-13b-chat.Q6_K.gguf")

# # models

# leo7b = models.LlamaCppChat(modelpath_leo_7b_mistral, n_gpu_layers=10)
# leo13b = models.LlamaCppChat(modelpath_leo_13b, n_gpu_layers = 20)

# basic test

lm_test = models.OpenAI("gpt-3.5-turbo") + f'Antworte mir. Wo ist der baum?' + gen(name = 'baum')

print(lm_test)
print(lm_test['baum'])

# classes



def no_roles_chain(task: StanceClassification, lm):
    analysis = lm + 'Du bist ein Experte in politischer Kommunikation.'
    analysis += f"Analysiere für mich diesen Text: \n{task.input_text}\n"
    analysis += f'Ich hätte gerne eine kurze Analyse zu folgender Frage: \n Wie steht {task.entity} zu folgender Aussage "{task.statement}" basierend auf dem Text, den ich dir zur Analyse gegeben habe'
    analysis += f"Dies ist meine Analyse: \n" + gen(max_tokens = 60, name = 'analysis')
    summarization = lm + 'Du bist ein Assistent, der bestehende Analysen zusammenfasst.'
    summarization += f'Betrachte die folgende Analyse\n' + f'{str(analysis["analysis"])}\n'
    summarization += f'Die Haltung von {task.entity} basierend auf dieser Analyse lässt sich so kategorisieren: ' + select(['Zustimmung','Ablehnung','Neutral'], name = 'stance')
    out = {}
    out["analysis"] = analysis
    out["summarization"] = summarization
    return out

def roles_chain(task: StanceClassification, lm):
    with system():
        analysis = lm + 'Du bist ein Experte in politischer Kommunikation.'
    with user():
        analysis += f"Analysiere für mich diesen Text: \n{task.input_text}\n"
        analysis += f'Ich hätte gerne eine kurze Analyse zu folgender Frage: \n Wie steht {task.entity} zu folgender Aussage "{task.statement}" basierend auf dem Text, den ich dir zur Analyse gegeben habe'
    with assistant():
        analysis += f"Dies ist meine Analyse: \n" + gen(max_tokens = 60, name = 'analysis')
    with system():
        summarization = lm + 'Du bist ein Assistent, der bestehende Analysen zusammenfasst.'
    with user():
        summarization += f'Betrachte die folgende Analyse\n' + f'{str(analysis["analysis"])}\n'
    with assistant():
        summarization += f'Wie {task.entity} zur Aussage "{task.statement}" basierend auf dieser Analyse steht lässt sich so kategorisieren: ' + select(['Zustimmung','Ablehnung','Neutral'], name = 'stance')
    out = {}
    out["analysis"] = analysis
    out["summarization"] = summarization
    return out


task1 = StanceClassification(input_text = """Peter mag keine Äpfel.
                Er findet Äpfel sind mega doof.""",
                statement = "Äpfel sind super",
                entity = "Peter")

task2 = StanceClassification(input_text = """Peter mag keine Äpfel.
                Er findet Äpfel sind mega doof.""",
                statement = "Äpfel sind wiederlich",
                entity = "Peter")

llm_setup1 = StanceDetectionMethod(lm = leo13b, fun = no_roles_chain)
llm_setup2 = StanceDetectionMethod(lm = leo7b, fun = no_roles_chain)

# illustration

task1.detect_stance(llm_setup1)

task2.detect_stance(llm_setup1)

task1.detect_stance(llm_setup2)
