import copy
from llmtuner import ChatModel
from tqdm import tqdm
import argparse
import json


parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--model_name_or_path", default="/mnt2/pretrained_model/LLM/gemma-7b", type=str,
                    help="Model name or path")
parser.add_argument("--adapter_name_or_path", default="output/checkpoint-900", type=str,
                    help="Checkpoint directory")
parser.add_argument("--template", default="gemma", type=str, help="Template type")
parser.add_argument("--finetuning_type", default="lora", type=str, help="Fine-tuning type")


args = parser.parse_args()
args_dict = vars(args)
chat_model = ChatModel(args_dict)
history = []

prompt1 = "You are a genetic disease expert. In this Gene-Disease relation extraction task, you need to follow 3 steps. You need to extract the [gene, function change, disease] triplet from the text, such as: [SHROOM3, LOF, Neural tube defects]. The second element in the triple means the regulation that the gene produces to the disease. Types of regulations are: LOF and GOF, which indicate loss or gain of function; REG, which indicates a general regulatory relationship; COM, which indicates that the functional change between genes and diseases is more complex, and it is difficult to determine whether the functional change is LOF or GOF. Please return all the relations extracted from the text in ternary format [[GENE, FUNCTION, DISEASE]]."
prompt2 = "You are a biologist AI. I'll give you the abstract of literature. Please identify all the [[compound,disease]] relations in the abstract, and just give me a list of all relations you recognized"
prompt3 = "You are a medicinal chemist. Now you need to identify all the drug-drug interactions from the text I provide to you, and please only write down all the drug-drug interactions in the format of [[drug, interaction, drug]]. "


fin = open("data/submission.jsonl", "r", encoding="utf8")
fout = open("submission.jsonl", "w", encoding="utf-8")
for idx, i in tqdm(enumerate(fin)):
    i = i.strip()
    if i == "":
        continue
    data = json.loads(i)
    task = data['task']
    if task == 2:
        text = data['abstract']
    else:
        text = data['text']
    if task == 1:
        prompt = copy.deepcopy(prompt1)
    elif task == 2:
        prompt = copy.deepcopy(prompt2)
    else:
        prompt = copy.deepcopy(prompt3)

    input_text = prompt+"\n"+text
    messages = [{"role": "user", "content": input_text}]
    response = chat_model.chat(messages, max_length=1536, history=[], do_sample=False)
    try:
        output = eval(response[0].response_text)
    except:
        output = []

    relations = []
    for one in output:
        if task == 1:
            if len(one) == 3 and one[1] in ["REG", "LOF", "GOF", "COM"]:
                relations.append((one[0], one[1], one[2]))

        elif task == 2:
            if len(one) == 2 and "," not in one[0] and "," not in one[1]:
                relations.append((one[0], one[1]))

        elif task == 3:
            if len(one) == 3 and one[1] in ["effect", "advise", "mechanism", "int"]:
                relations.append((one[0], one[1], one[2]))


    relations = list(set(relations))
    new_relations = []
    punc = [",", "(", ")"]
    for rel in relations:
        flag = False
        for p in punc:
            if p in rel[0] or p in rel[-1]:
                flag = True
        if not flag:
            new_relations.append(rel)

    value = ", ".join([f'({", ".join(match)})' for match in new_relations])
    print(value)
    if task == 1:
        line = {"text": text, "ideal": {"GENE, FUNCTION, DISEASE": value}, "task": task}
        fout.write(json.dumps(line, ensure_ascii=False) + "\n")
    elif task == 2:
        line = {"abstract": text, "ideal": {"chemical, disease": value}, "task": task}
        fout.write(json.dumps(line, ensure_ascii=False) + "\n")
    elif task == 3:
        line = {"text": text, "ideal": {"DDI-triples": value}, "task": task}
        fout.write(json.dumps(line, ensure_ascii=False) + "\n")

fout.close()